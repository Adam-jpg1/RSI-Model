import time
import traceback
from datetime import datetime
from typing import Dict, List
from data.fetcher import get_price_data, get_stock_news
from data.indicators import add_all_indicators
from sentiment.analyzer import analyze_headlines
from bot.alpaca_client import AlpacaClient
from bot.strategy import generate_signal, compute_position_size
from ml.model import RSIModel
from config import CONFIG


class LiveTrader:
    def __init__(self, models: Dict[str, RSIModel], tickers: List[str] = None):
        self.models = models
        self.tickers = tickers or list(models.keys())
        self.client = AlpacaClient()
        self.last_signals: Dict[str, Dict] = {}

    def run(self):
        print(f"\n[trader] Starting live trader — paper={CONFIG.bot.paper_trading}")
        print(f"[trader] Watching: {self.tickers}")

        while True:
            try:
                if not self.client.is_market_open():
                    print(f"[trader] Market closed. Sleeping 60s...")
                    time.sleep(60)
                    continue

                self._tick()

            except KeyboardInterrupt:
                print("\n[trader] Stopped by user.")
                break
            except Exception:
                traceback.print_exc()

            time.sleep(CONFIG.bot.poll_interval_seconds)

    def _tick(self):
        acct = self.client.get_account()
        portfolio_value = acct["portfolio_value"]
        positions = self.client.get_positions()
        ts = datetime.now().strftime("%H:%M:%S")

        for ticker in self.tickers:
            model = self.models.get(ticker)
            if model is None or not model.is_trained:
                continue

            df = get_price_data(ticker, period="5d", interval="5m")
            df = add_all_indicators(df)

            headlines = get_stock_news(ticker)
            sentiment = analyze_headlines(headlines)
            sentiment_score = sentiment["score"]

            sig = generate_signal(df, model, sentiment_score)
            if sig is None:
                continue

            signal = sig["signal"]
            confidence = sig["confidence"]
            prev = self.last_signals.get(ticker, {})
            self.last_signals[ticker] = sig

            print(
                f"[{ts}] {ticker:6s}  {signal:4s}  conf={confidence:.3f}  "
                f"sent={sentiment_score:+.3f}  rsi={sig.get('rsi_14', 0):.1f}  "
                f"price=${sig['price']:.2f}"
            )

            in_position = ticker in positions

            if in_position:
                pos = positions[ticker]
                pnl_pct = pos["unrealized_pnl_pct"] / 100
                stop = -CONFIG.bot.stop_loss_pct
                take = CONFIG.bot.take_profit_pct

                should_exit = (
                    pnl_pct <= stop
                    or pnl_pct >= take
                    or (signal == "DOWN" and confidence >= CONFIG.bot.min_confidence)
                )
                if should_exit:
                    reason = "stop" if pnl_pct <= stop else ("take" if pnl_pct >= take else "signal")
                    print(f"[{ts}] {ticker:6s}  CLOSE ({reason})  pnl={pnl_pct*100:+.2f}%")
                    self.client.close_position(ticker)

            elif signal == "UP" and confidence >= CONFIG.bot.min_confidence:
                alloc = compute_position_size(confidence, sentiment_score, portfolio_value, signal)
                if alloc < 1.0:
                    continue
                price = sig["price"]
                qty = alloc / price
                if qty < 0.01:
                    continue
                print(f"[{ts}] {ticker:6s}  BUY  qty={qty:.3f}  alloc=${alloc:.2f}  conf={confidence:.3f}")
                self.client.place_market_order(ticker, qty, "BUY")

    def close_all(self):
        positions = self.client.get_positions()
        for ticker in positions:
            print(f"[trader] Closing {ticker}...")
            self.client.close_position(ticker)
        print("[trader] All positions closed.")
