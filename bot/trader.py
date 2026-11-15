import logging
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional

from config import CONFIG
from data.fetcher import get_price_data, get_daily_data, get_stock_news
from data.indicators import add_all_indicators
from data.universe import get_trained_tickers, get_model_age_hours
from sentiment.analyzer import analyze_headlines
from bot.alpaca_client import AlpacaClient
from bot.strategy import generate_signal, compute_position_size, get_dynamic_confidence
from data.indicators import (add_peer_features, add_macro_event_features,
                              add_calendar_features, add_cross_sectional_features,
                              add_microstructure_features, add_regime_features,
                              add_earnings_features)
from data.fetcher import get_earnings_dates
from ml.model import RSIModel
from ml.optimizer import load_best_params
from portfolio.manager import PortfolioManager
from screening.scanner import scan_universe

logger = logging.getLogger(__name__)


class LiveTrader:
    def __init__(self, models: Dict[str, RSIModel] = None, tickers: List[str] = None):
        self.models = models or {}
        self.tickers = tickers or list(self.models.keys())
        self.client = AlpacaClient()
        self.portfolio = PortfolioManager()
        self.last_signals: Dict[str, Dict] = {}
        self._last_retrain: float = 0.0
        self._order_ids: set = set()  # duplicate-order guard
        # Cache Optuna-tuned ADX and confidence thresholds — keyed by ticker, None = no saved params
        self._optuna_adx: Dict[str, Optional[float]] = self._load_optuna_adx()
        self._optuna_conf: Dict[str, Optional[float]] = self._load_optuna_conf()

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_optuna_adx() -> Dict[str, Optional[float]]:
        """Load saved Optuna ADX thresholds for all trained tickers."""
        from data.universe import get_trained_tickers as _gtt
        result: Dict[str, Optional[float]] = {}
        for ticker in _gtt():
            best = load_best_params(ticker)
            result[ticker] = best["adx_filter_min"] if best else None
        return result

    @staticmethod
    def _load_optuna_conf() -> Dict[str, Optional[float]]:
        """Load saved per-ticker min_confidence from param files."""
        from data.universe import get_trained_tickers as _gtt
        result: Dict[str, Optional[float]] = {}
        for ticker in _gtt():
            best = load_best_params(ticker)
            result[ticker] = best.get("min_confidence") if best else None
        return result

    # ------------------------------------------------------------------ #
    # Constructors                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def load_all_trained(cls) -> "LiveTrader":
        """Load every model from disk whose age is within staleness threshold."""
        tickers = get_trained_tickers()
        models: Dict[str, RSIModel] = {}
        skipped = []
        for ticker in tickers:
            age = get_model_age_hours(ticker)
            if age and age > CONFIG.bot.model_staleness_hours:
                skipped.append(f"{ticker}({age:.0f}h)")
                continue
            m = RSIModel()
            try:
                m.load(ticker=ticker)
                models[ticker] = m
            except Exception as e:
                logger.warning(f"[trader] Failed to load {ticker}: {e}")
        if skipped:
            print(f"[trader] Skipped stale models: {', '.join(skipped)}")
        print(f"[trader] Loaded {len(models)} models: {list(models.keys())}")
        return cls(models, tickers=list(models.keys()))

    # ------------------------------------------------------------------ #
    # Public entry points                                                 #
    # ------------------------------------------------------------------ #

    def run(self):
        """Manual mode: trade only the tickers in self.tickers."""
        print(f"\n[trader] Manual mode — paper={CONFIG.bot.paper_trading}  tickers={self.tickers}")
        self._main_loop(autonomous=False)

    def run_autonomous(self):
        """
        Autonomous mode: scan full universe every poll cycle, discover and trade
        opportunities across all tickers that have trained models.
        """
        print(f"\n[trader] Autonomous mode — paper={CONFIG.bot.paper_trading}")
        print(f"[trader] {len(self.tickers)} models loaded")
        self._main_loop(autonomous=True)

    # ------------------------------------------------------------------ #
    # Main loop                                                           #
    # ------------------------------------------------------------------ #

    def _main_loop(self, autonomous: bool):
        while True:
            try:
                if not self.client.is_market_open():
                    self._maybe_retrain()
                    print(f"[trader] Market closed — sleeping 60s...")
                    time.sleep(60)
                    continue

                acct = self.client.get_account()
                portfolio_value = acct["portfolio_value"]
                positions = self.client.get_positions()
                self.portfolio.update_from_alpaca(positions)

                # 1. Manage existing positions (stop/take/signal exits)
                self._manage_positions(positions, portfolio_value)

                # 2. Find and enter new opportunities
                if autonomous:
                    opps = scan_universe(max_workers=CONFIG.screening.max_scan_workers,
                                        top_n=CONFIG.screening.top_n_opportunities)
                else:
                    opps = self._signals_for_tickers(portfolio_value)

                self._enter_opportunities(opps, positions, portfolio_value)
                self._print_status(acct, positions, opps[:5])

            except KeyboardInterrupt:
                print("\n[trader] Stopped by user.")
                break
            except Exception:
                traceback.print_exc()

            time.sleep(CONFIG.bot.poll_interval_seconds)

    # ------------------------------------------------------------------ #
    # Opportunity generation (manual mode)                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fetch_peer_data() -> dict:
        """Fetch hourly NVDA, XLK, ^VIX data once per poll cycle for cross-asset RSI features."""
        peer_data = {}
        for sym in ["NVDA", "XLK", "^VIX"]:
            try:
                peer_data[sym] = get_price_data(sym, period="30d", interval="1h")
            except Exception:
                peer_data[sym] = None
        return peer_data

    def _fetch_peer_close(self) -> dict:
        """Fetch Close series for all portfolio tickers for cross-sectional rank features."""
        result = {}
        for sym in self.tickers:
            try:
                df = get_price_data(sym, period="30d", interval="1h")
                result[sym] = df["Close"].rename(sym)
            except Exception:
                result[sym] = None
        return result

    def _signals_for_tickers(self, portfolio_value: float) -> List[Dict]:
        opps = []
        peer_data = self._fetch_peer_data()
        peer_close = self._fetch_peer_close()
        for ticker in self.tickers:
            model = self.models.get(ticker)
            if model is None or not model.is_trained:
                continue
            try:
                df = get_price_data(ticker, period="30d", interval="1h")
                df = add_all_indicators(df)
                df = add_peer_features(
                    df,
                    nvda_hourly_df=peer_data.get("NVDA") if ticker != "NVDA" else None,
                    xlk_hourly_df=peer_data.get("XLK"),
                    vix_hourly_df=peer_data.get("^VIX"),
                )
                df = add_macro_event_features(df)
                df = add_calendar_features(df)
                others = {t: s for t, s in peer_close.items() if t != ticker and s is not None}
                if others:
                    df = add_cross_sectional_features(df, ticker, others)
                df = add_microstructure_features(df)
                df = add_regime_features(df)
                try:
                    edates = get_earnings_dates(ticker)
                    if edates:
                        df = add_earnings_features(df, edates)
                except Exception:
                    pass
                daily_df = None
                try:
                    daily_df = get_daily_data(ticker, period="1y")
                except Exception:
                    pass
                headlines = get_stock_news(ticker)
                sentiment_score = analyze_headlines(headlines)["score"]
                sig = generate_signal(df, model, sentiment_score, daily_df=daily_df)
                if sig:
                    opps.append({
                        "ticker": ticker,
                        "signal": sig["signal"],
                        "confidence": sig["confidence"],
                        "adx": sig.get("adx") or 0.0,
                        "rsi": sig.get("rsi_14") or 50.0,
                        "vol_ratio": 1.0,
                        "d_trend": sig.get("d_trend", 0.0),
                        "price": sig["price"],
                        "sentiment_score": sig["sentiment_score"],
                    })
            except Exception:
                logger.debug(traceback.format_exc())
        return opps

    # ------------------------------------------------------------------ #
    # Position management                                                 #
    # ------------------------------------------------------------------ #

    def _manage_positions(self, positions: Dict, portfolio_value: float):
        ts = datetime.now().strftime("%H:%M:%S")
        peer_data = self._fetch_peer_data()
        peer_close = self._fetch_peer_close()
        for ticker, pos in list(positions.items()):
            model = self.models.get(ticker)
            if model is None:
                continue
            try:
                df = get_price_data(ticker, period="30d", interval="1h")
                df = add_all_indicators(df)
                df = add_peer_features(
                    df,
                    nvda_hourly_df=peer_data.get("NVDA") if ticker != "NVDA" else None,
                    xlk_hourly_df=peer_data.get("XLK"),
                    vix_hourly_df=peer_data.get("^VIX"),
                )
                df = add_macro_event_features(df)
                df = add_calendar_features(df)
                others = {t: s for t, s in peer_close.items() if t != ticker and s is not None}
                if others:
                    df = add_cross_sectional_features(df, ticker, others)
                df = add_microstructure_features(df)
                df = add_regime_features(df)
                try:
                    edates = get_earnings_dates(ticker)
                    if edates:
                        df = add_earnings_features(df, edates)
                except Exception:
                    pass
                daily_df = None
                try:
                    daily_df = get_daily_data(ticker, period="1y")
                except Exception:
                    pass
                headlines = get_stock_news(ticker)
                sentiment_score = analyze_headlines(headlines)["score"]
                sig = generate_signal(df, model, sentiment_score, daily_df=daily_df)
                if sig:
                    self._check_exit(ticker, pos, sig["signal"], sig["confidence"], ts)
            except Exception:
                logger.debug(traceback.format_exc())

    def _check_exit(self, ticker: str, pos: Dict, signal: str, confidence: float, ts: str):
        pnl_pct = pos["unrealized_pnl_pct"] / 100.0
        qty = pos["qty"]
        side = "long" if qty >= 0 else "short"
        stop = CONFIG.bot.stop_loss_pct
        take = CONFIG.bot.take_profit_pct

        exit_conf = get_dynamic_confidence()
        if side == "long":
            should_exit = (
                pnl_pct <= -stop
                or pnl_pct >= take
                or (signal == "DOWN" and confidence >= exit_conf)
            )
        else:  # short
            should_exit = (
                pnl_pct <= -stop
                or pnl_pct >= take
                or (signal == "UP" and confidence >= exit_conf)
            )

        if should_exit:
            reason = "stop" if pnl_pct <= -stop else ("take" if pnl_pct >= take else "signal")
            print(f"[{ts}] {ticker:6s}  CLOSE-{side.upper()} ({reason})  pnl={pnl_pct*100:+.2f}%")
            self.client.close_position(ticker)
            self.portfolio.close_position(ticker)
            self.portfolio.record_trade(ticker, pnl_pct * 100)

    # ------------------------------------------------------------------ #
    # New entry execution                                                 #
    # ------------------------------------------------------------------ #

    def _enter_opportunities(self, opps: List[Dict], positions: Dict, portfolio_value: float):
        ts = datetime.now().strftime("%H:%M:%S")
        for opp in opps:
            ticker = opp["ticker"]
            signal = opp["signal"]
            confidence = opp["confidence"]
            adx_val = opp.get("adx", 0.0) or 0.0
            sentiment_score = opp.get("sentiment_score", 0.0)

            if ticker in positions:
                continue
            if signal not in ("UP", "DOWN"):
                continue
            dyn_conf = get_dynamic_confidence(base_conf=self._optuna_conf.get(ticker))
            if confidence < dyn_conf:
                continue
            if not self.portfolio.can_enter(ticker, signal, portfolio_value):
                continue

            # Prefer Optuna-tuned ADX threshold over config default
            optuna_adx = self._optuna_adx.get(ticker)
            adx_min = (optuna_adx if optuna_adx is not None
                       else CONFIG.bot.per_ticker_adx.get(ticker, CONFIG.bot.adx_filter_min))
            if adx_min > 0 and adx_val < adx_min:
                continue

            alloc = self.portfolio.position_size_dollars(
                confidence, sentiment_score, portfolio_value, signal, ticker=ticker
            )
            if alloc < 1.0:
                continue

            price = opp["price"]
            qty = alloc / price
            if qty < 0.01:
                continue

            if signal == "UP":
                self._place_long(ticker, qty, price, confidence, adx_val, ts)
            elif signal == "DOWN" and CONFIG.bot.allow_short:
                self._place_short(ticker, qty, price, confidence, adx_val, ts)

    def _place_long(self, ticker: str, qty: float, price: float, conf: float, adx: float, ts: str):
        print(f"[{ts}] {ticker:6s}  LONG   qty={qty:.3f}  ${qty*price:,.0f}  conf={conf:.3f}  adx={adx:.1f}")
        result = self.client.place_market_order(ticker, qty, "BUY")
        if result and result.get("id") not in self._order_ids:
            self._order_ids.add(result.get("id", ""))
            self.portfolio.open_position(ticker, "long", qty, price)

    def _place_short(self, ticker: str, qty: float, price: float, conf: float, adx: float, ts: str):
        print(f"[{ts}] {ticker:6s}  SHORT  qty={qty:.3f}  ${qty*price:,.0f}  conf={conf:.3f}  adx={adx:.1f}")
        result = self.client.place_market_order(ticker, qty, "SELL")
        if result and result.get("id") not in self._order_ids:
            self._order_ids.add(result.get("id", ""))
            self.portfolio.open_position(ticker, "short", qty, price)

    # ------------------------------------------------------------------ #
    # Scheduled retraining                                                #
    # ------------------------------------------------------------------ #

    def _maybe_retrain(self):
        interval_secs = CONFIG.bot.retrain_interval_hours * 3600
        if time.time() - self._last_retrain < interval_secs:
            return
        print("[trader] Starting scheduled retraining...")
        from ml.trainer import train_on_ticker
        for ticker in get_trained_tickers():
            age = get_model_age_hours(ticker)
            if not (age and age > CONFIG.bot.retrain_interval_hours):
                continue
            try:
                result = train_on_ticker(ticker)
                self.models[ticker] = result["model"]
                if ticker not in self.tickers:
                    self.tickers.append(ticker)
                up_prec = result["report"].get("UP", {}).get("precision", 0.0)
                print(f"[trainer] {ticker} retrained  UP_prec={up_prec:.3f}")
            except Exception as e:
                print(f"[trainer] {ticker} failed: {e}")
        self._last_retrain = time.time()
        self._optuna_adx = self._load_optuna_adx()   # refresh after retraining
        self._optuna_conf = self._load_optuna_conf()

    # ------------------------------------------------------------------ #
    # Utilities                                                           #
    # ------------------------------------------------------------------ #

    def _print_status(self, acct: Dict, positions: Dict, top_opps: List[Dict]):
        ts = datetime.now().strftime("%H:%M:%S")
        pv = acct["portfolio_value"]
        cash = acct["cash"]
        summary = self.portfolio.summary(pv)
        print(
            f"\n[{ts}] ${pv:,.0f}  cash ${cash:,.0f}  "
            f"pos {summary['open_positions']}/{summary['max_positions']}  "
            f"exposure {summary['total_exposure_pct']:.1f}%/{summary['max_exposure_pct']:.0f}%"
        )
        for t, p in list(positions.items())[:6]:
            print(f"  {t:6s}  ${p['market_value']:,.0f}  pnl {p['unrealized_pnl_pct']:+.2f}%")
        if top_opps:
            sigs = "  ".join(f"{o['ticker']}={o['signal']}({o['confidence']:.2f})" for o in top_opps)
            print(f"  signals: {sigs}")

    def close_all(self):
        positions = self.client.get_positions()
        for ticker in list(positions.keys()):
            print(f"[trader] Closing {ticker}...")
            self.client.close_position(ticker)
            self.portfolio.close_position(ticker)
        print("[trader] All positions closed.")
