import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import CONFIG

logger = logging.getLogger(__name__)

# Sector lookup cache — populated lazily via yfinance info calls
_sector_cache: Dict[str, str] = {}


def _fetch_sector(ticker: str) -> str:
    """Return the sector string for a ticker, with an in-process cache."""
    if ticker in _sector_cache:
        return _sector_cache[ticker]
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        sector = info.get("sector") or "Unknown"
    except Exception:
        sector = "Unknown"
    _sector_cache[ticker] = sector
    return sector


class PortfolioManager:
    """
    Tracks open positions and enforces portfolio-level risk constraints:
      - Max positions open simultaneously
      - Max total portfolio exposure
      - Max per-position size
      - Max single-sector exposure (prevents tech concentration)
      - Dynamic Kelly sizing scaled by rolling model win rate
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        self._positions: Dict[str, Dict] = {}
        # Rolling trade outcomes per ticker for dynamic sizing
        self._trade_history: Dict[str, deque] = {}

    # ------------------------------------------------------------------ #
    # Sync from live source                                               #
    # ------------------------------------------------------------------ #

    def update_from_alpaca(self, positions: Dict[str, Dict]):
        """Overwrite internal state from an Alpaca get_positions() response."""
        self._positions = {}
        for ticker, pos in positions.items():
            qty = pos["qty"]
            self._positions[ticker] = {
                "side": "long" if qty >= 0 else "short",
                "qty": abs(qty),
                "entry_price": pos["avg_entry"],
                "market_value": abs(pos["market_value"]),
                "unrealized_pnl_pct": pos["unrealized_pnl_pct"],
                "entry_time": time.time(),
            }

    # ------------------------------------------------------------------ #
    # Risk checks                                                         #
    # ------------------------------------------------------------------ #

    def total_exposure(self, portfolio_value: float) -> float:
        total_mv = sum(p.get("market_value", 0.0) for p in self._positions.values())
        return total_mv / portfolio_value if portfolio_value > 0 else 0.0

    def sector_exposure(self, portfolio_value: float) -> Dict[str, float]:
        """Return fraction of portfolio in each sector across open positions."""
        by_sector: Dict[str, float] = {}
        for ticker, pos in self._positions.items():
            sector = _fetch_sector(ticker)
            mv = pos.get("market_value", 0.0)
            by_sector[sector] = by_sector.get(sector, 0.0) + mv
        return {s: mv / portfolio_value for s, mv in by_sector.items()} if portfolio_value else {}

    def can_enter(self, ticker: str, signal: str, portfolio_value: float) -> bool:
        if ticker in self._positions:
            return False
        if len(self._positions) >= CONFIG.bot.max_positions:
            return False
        if self.total_exposure(portfolio_value) >= CONFIG.bot.max_portfolio_exposure:
            return False
        if signal == "DOWN" and not CONFIG.bot.allow_short:
            return False
        # Sector concentration check
        sector = _fetch_sector(ticker)
        if sector != "Unknown":
            sec_exp = self.sector_exposure(portfolio_value)
            if sec_exp.get(sector, 0.0) >= CONFIG.bot.max_sector_exposure:
                logger.debug(f"[portfolio] {ticker} blocked — {sector} sector at "
                             f"{sec_exp.get(sector,0)*100:.1f}% (max {CONFIG.bot.max_sector_exposure*100:.0f}%)")
                return False
        return True

    # ------------------------------------------------------------------ #
    # Position sizing                                                     #
    # ------------------------------------------------------------------ #

    def rolling_win_rate(self, ticker: str) -> Optional[float]:
        """Return rolling win rate for the last N trades, or None if too few."""
        history = self._trade_history.get(ticker)
        if not history or len(history) < 3:
            return None
        return float(np.mean([1.0 if pnl > 0 else 0.0 for pnl in history]))

    def dynamic_kelly_multiplier(self, ticker: str) -> float:
        """
        Scale position size between dynamic_min_scale and dynamic_max_scale
        based on rolling win rate. Returns 1.0 when insufficient history.
        """
        if not CONFIG.bot.dynamic_sizing:
            return 1.0
        wr = self.rolling_win_rate(ticker)
        if wr is None:
            return 1.0
        # Linear ramp: 40% win rate → min_scale, 70% win rate → max_scale
        lo, hi = 0.40, 0.70
        t = (wr - lo) / (hi - lo)
        t = float(np.clip(t, 0.0, 1.0))
        return CONFIG.bot.dynamic_min_scale + t * (CONFIG.bot.dynamic_max_scale - CONFIG.bot.dynamic_min_scale)

    def position_size_dollars(
        self,
        confidence: float,
        sentiment_score: float,
        portfolio_value: float,
        signal: str = "UP",
        ticker: str = "",
    ) -> float:
        """Dollar allocation for one trade, respecting all risk limits."""
        base_pct = confidence * CONFIG.bot.max_position_pct
        sentiment_adj = sentiment_score * CONFIG.bot.sentiment_weight

        if signal == "DOWN":
            base_pct *= CONFIG.bot.short_size_factor

        pct = float(np.clip(base_pct + sentiment_adj, 0.0, CONFIG.bot.max_position_pct))

        # Dynamic Kelly scaling
        pct *= self.dynamic_kelly_multiplier(ticker)
        pct = min(pct, CONFIG.bot.max_position_pct)

        # Never exceed remaining exposure budget
        remaining = CONFIG.bot.max_portfolio_exposure - self.total_exposure(portfolio_value)
        pct = min(pct, max(remaining, 0.0))

        return portfolio_value * pct

    # ------------------------------------------------------------------ #
    # Position lifecycle                                                  #
    # ------------------------------------------------------------------ #

    def open_position(self, ticker: str, side: str, qty: float, entry_price: float):
        self._positions[ticker] = {
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "entry_time": time.time(),
            "market_value": qty * entry_price,
            "unrealized_pnl_pct": 0.0,
        }

    def close_position(self, ticker: str):
        self._positions.pop(ticker, None)

    def record_trade(self, ticker: str, pnl_pct: float):
        """Record the outcome of a closed trade for dynamic sizing."""
        if ticker not in self._trade_history:
            self._trade_history[ticker] = deque(maxlen=CONFIG.bot.dynamic_sizing_window)
        self._trade_history[ticker].append(pnl_pct)

    def get_position(self, ticker: str) -> Optional[Dict]:
        return self._positions.get(ticker)

    def get_all_positions(self) -> Dict[str, Dict]:
        return dict(self._positions)

    def open_count(self) -> int:
        return len(self._positions)

    # ------------------------------------------------------------------ #
    # Reporting                                                           #
    # ------------------------------------------------------------------ #

    def summary(self, portfolio_value: float) -> Dict:
        sec_exp = self.sector_exposure(portfolio_value)
        top_sector = max(sec_exp, key=sec_exp.get) if sec_exp else "—"
        top_sector_pct = sec_exp.get(top_sector, 0.0) * 100
        return {
            "open_positions": self.open_count(),
            "max_positions": CONFIG.bot.max_positions,
            "total_exposure_pct": round(self.total_exposure(portfolio_value) * 100, 1),
            "max_exposure_pct": CONFIG.bot.max_portfolio_exposure * 100,
            "top_sector": top_sector,
            "top_sector_pct": round(top_sector_pct, 1),
        }

    def model_performance_summary(self) -> Dict[str, Dict]:
        """Return rolling win rate and Kelly multiplier for every tracked ticker."""
        out = {}
        for ticker, history in self._trade_history.items():
            if len(history) < 3:
                continue
            trades = list(history)
            wr = float(np.mean([1.0 if p > 0 else 0.0 for p in trades]))
            out[ticker] = {
                "trades": len(trades),
                "win_rate": round(wr, 3),
                "kelly_mult": round(self.dynamic_kelly_multiplier(ticker), 2),
                "avg_pnl_pct": round(float(np.mean(trades)), 3),
            }
        return out
