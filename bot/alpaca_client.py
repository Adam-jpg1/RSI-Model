from typing import Optional, Dict, List
from config import CONFIG

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    raise ImportError("Run: pip install alpaca-py")


class AlpacaClient:
    def __init__(self):
        cfg = CONFIG.bot
        self.trading = TradingClient(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
            paper=cfg.paper_trading,
        )
        self.data = StockHistoricalDataClient(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
        )

    def get_account(self) -> Dict:
        acct = self.trading.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
        }

    def get_positions(self) -> Dict[str, Dict]:
        positions = {}
        for p in self.trading.get_all_positions():
            positions[p.symbol] = {
                "qty": float(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pnl": float(p.unrealized_pl),
                "unrealized_pnl_pct": float(p.unrealized_plpc) * 100,
            }
        return positions

    def place_market_order(self, ticker: str, qty: float, side: str) -> Dict:
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=ticker,
            qty=round(qty, 2),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading.submit_order(req)
        return {"id": str(order.id), "status": str(order.status), "symbol": ticker, "qty": qty, "side": side}

    def close_position(self, ticker: str) -> Optional[Dict]:
        try:
            order = self.trading.close_position(ticker)
            return {"id": str(order.id), "status": str(order.status), "symbol": ticker}
        except Exception as e:
            print(f"[alpaca] close_position {ticker} failed: {e}")
            return None

    def get_latest_price(self, ticker: str) -> Optional[float]:
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quote = self.data.get_stock_latest_quote(req)
            return float(quote[ticker].ask_price)
        except Exception:
            return None

    def is_market_open(self) -> bool:
        clock = self.trading.get_clock()
        return clock.is_open
