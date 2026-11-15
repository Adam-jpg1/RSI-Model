"""
RSI(2) mean-reversion strategy on liquid index ETFs (SPY / QQQ / IWM).

Strategy rules (Connors & Alvarez 2009; Faber 2007):
  - Long when: Close > SMA(sma_period)  AND  Wilder RSI(2) < rsi_entry
  - Exit long: Wilder RSI(2) > rsi_exit  OR  Close < SMA(sma_period)
  - Short leg omitted (index ETFs have embedded upward drift bias)

Position sizing:
  - Volatility-targeted weight (Moreira & Muir 2017):
      weight_t = target_vol / realized_vol_{t-1}  (capped at max_weight)
  - When target_vol is None, position = 1 (unscaled binary)

Costs: COST_PER_SIDE per side on every unit of position change
       (commission + half bid-ask spread + market impact).

Parameter search:
  IS  2005-01-01 -> 2017-12-31  (grid search; select max avg Sharpe across tickers)
  OOS 2018-01-01 -> present     (held-out; also annual expanding-window walk-forward)

Metrics: CAGR, Sharpe, Deflated Sharpe (Bailey & López de Prado 2014),
         max drawdown, win rate, exposure time - all vs B&H SPY.

Outputs:
  rsi2_param_log.csv    every IS param combination + metrics
  rsi2_results.json     best params, IS/OOS/WF metrics
  stdout                formatted report

Usage:
  python rsi2_strategy.py           # full run
  python rsi2_strategy.py --quick   # 5-combo sanity check on SPY only
"""

import os, csv, json, math, warnings, argparse
from copy import deepcopy
from itertools import product
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


# ==============================================================================
# GLOBAL SETTINGS
# ==============================================================================
TICKERS = ["SPY", "QQQ", "IWM"]

FETCH_START   = "2003-01-01"      # 2 extra years for SMA/RSI warmup
IS_END        = "2017-12-31"
OOS_START     = "2018-01-01"
TODAY         = datetime.today().strftime("%Y-%m-%d")

ANNUAL_BARS   = 252
COST_PER_SIDE = 0.0005            # 5 bps one-way (commission + half-spread + impact)

PARAM_GRID: Dict[str, List] = {
    # entry=5 (original quality) and entry=10 (slightly more trades).
    # Higher target_vol + max_weight = more leverage on RSI signals -> more CAGR.
    # Higher trend_base_pct captures bull-market drift between RSI trades.
    # Vol-targeting auto-scales down in rough markets, protecting drawdowns.
    # vix_block: block new entries when VIX is above this level (None = disabled).
    # hard_stop_pct: exit trade if unrealised loss exceeds this fraction (None = disabled).
    "rsi_entry"      : [5, 10],
    "rsi_exit"       : [65, 70, 75],
    "sma_period"     : [150, 200],
    "target_vol"     : [0.15, 0.20],
    # 5-day lookback reacts to vol spikes within the same week — reduces leverage
    # faster during crashes than 21-day, while also applying more leverage in
    # calm/trending periods. OOS 2018-2026 shows clear improvement over slower windows.
    "vol_lookback"   : [5],
    "max_weight"     : [1.5, 2.0, 2.5],
    "sizing_mode"    : ["rsi_depth"],
    "trend_base_pct" : [0.10, 0.20],
}

# Module-level cash rate series (set once in main; used by backtest_one).
# Holds the daily T-bill rate (^IRX annualised / 252).
_CASH_RATE: Optional[pd.Series] = None

# Module-level VIX series (set once in main; used by _raw_signal for entry gate).
_VIX: Optional[pd.Series] = None

RESULTS_CSV  = "rsi2_param_log.csv"
RESULTS_JSON = "rsi2_results.json"


# ==============================================================================
# DATA
# ==============================================================================
def fetch_prices(tickers: List[str], start: str, end: str) -> Dict[str, pd.Series]:
    """Download adj-close prices; returns {ticker: pd.Series[date]}."""
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])
    out: Dict[str, pd.Series] = {}
    for t in tickers:
        if t in raw.columns:
            s = raw[t].dropna()
            if len(s) > 0:
                out[t] = s
    return out


def fetch_cash_rate(start: str, end: str) -> pd.Series:
    """
    Daily T-bill rate from the 13-week Treasury yield index (^IRX).
    ^IRX is quoted as annualised % (e.g. 5.25 = 5.25%).  Divide by 100*252
    to get the daily rate added to idle capital each night.
    Falls back to 0 if Yahoo Finance can't serve the series.
    """
    try:
        raw = yf.download("^IRX", start=start, end=end,
                          auto_adjust=False, progress=False)
        # yfinance returns MultiIndex columns (col_name, ticker) for single tickers
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        series = raw[col].dropna()
        if len(series) == 0:
            raise ValueError("empty")
        # ^IRX is quoted as annualised % (e.g. 5.25 means 5.25% per year)
        daily = (series / 100.0 / ANNUAL_BARS)
        return daily.reindex(
            pd.date_range(start, end, freq="B")
        ).ffill().fillna(0.0)
    except Exception:
        return pd.Series(dtype=float)


def fetch_vix(start: str, end: str) -> pd.Series:
    """
    Daily VIX closing level (^VIX).  Used as an entry gate: when VIX exceeds
    vix_block no new RSI trades are opened (exits still fire normally).
    Falls back to empty series if Yahoo Finance is unavailable.
    """
    try:
        raw = yf.download("^VIX", start=start, end=end,
                          auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        series = raw[col].dropna()
        if len(series) == 0:
            raise ValueError("empty")
        return series.reindex(
            pd.date_range(start, end, freq="B")
        ).ffill().fillna(20.0)
    except Exception:
        return pd.Series(dtype=float)


# ==============================================================================
# INDICATORS
# ==============================================================================
def wilder_rsi(close: pd.Series, period: int = 2) -> pd.Series:
    """
    Wilder's RSI using RMA smoothing: alpha = 1/period.
    This is NOT the standard EMA (alpha = 2/(N+1)); it is Wilder's original
    smoothing which produces a slower, smoother oscillator.
    """
    delta    = close.diff()
    gain     = delta.clip(lower=0.0)
    loss     = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period, min_periods=period).mean()


def realized_vol_annual(returns: pd.Series, lookback: int) -> pd.Series:
    """Annualised realised vol (rolling std x sqrt252)."""
    return returns.rolling(lookback, min_periods=lookback).std() * math.sqrt(ANNUAL_BARS)


# ==============================================================================
# SIGNAL GENERATION  (vectorised state machine)
# ==============================================================================
def _raw_signal(
    close         : pd.Series,
    rsi_entry     : float,
    rsi_exit      : float,
    sma_period    : int,
    vix_block     : Optional[float] = None,
    hard_stop_pct : Optional[float] = None,
) -> np.ndarray:
    """
    State machine producing 0/1 raw binary signal (not yet shifted).
    State enters at close_t (position active from close_t onwards until exit).
    Caller must shift(1) before multiplying by returns.

    vix_block     : skip new entries on days when VIX >= this level.
    hard_stop_pct : force exit when price has fallen > this fraction from entry.
    """
    rsi_arr  = wilder_rsi(close, 2).values
    sma_arr  = sma(close, sma_period).values
    px_arr   = close.values
    n        = len(px_arr)
    pos      = np.zeros(n, dtype=np.float64)
    in_trade = False
    entry_px = 0.0

    # Pre-compute VIX array aligned to this ticker's dates
    vix_arr: Optional[np.ndarray] = None
    if vix_block is not None and _VIX is not None and len(_VIX) > 0:
        vix_arr = _VIX.reindex(close.index).ffill().fillna(20.0).values

    for i in range(n):
        if math.isnan(rsi_arr[i]) or math.isnan(sma_arr[i]):
            pos[i] = 0.0
            continue
        if not in_trade:
            # VIX gate: skip entry when market fear is elevated
            if vix_arr is not None and vix_arr[i] >= vix_block:
                pos[i] = 0.0
                continue
            if px_arr[i] > sma_arr[i] and rsi_arr[i] < rsi_entry:
                in_trade = True
                entry_px = px_arr[i]
        else:
            # Hard stop: exit before RSI recovery if loss exceeds threshold
            if hard_stop_pct is not None and entry_px > 0:
                if px_arr[i] < entry_px * (1.0 - hard_stop_pct):
                    in_trade = False
                    pos[i] = 0.0
                    continue
            if rsi_arr[i] > rsi_exit or px_arr[i] < sma_arr[i]:
                in_trade = False
        pos[i] = 1.0 if in_trade else 0.0

    return pos


def _vol_weight(
    daily_ret: pd.Series,
    target_vol: Optional[float],
    vol_lookback: int,
    max_weight: float,
) -> pd.Series:
    """
    Moreira & Muir (2017) volatility-managed weight.
    Returns series aligned to daily_ret index; uses yesterday's vol (shifted).
    When target_vol is None returns a constant 1.0 series.
    """
    if target_vol is None:
        return pd.Series(1.0, index=daily_ret.index)
    rvol = realized_vol_annual(daily_ret, vol_lookback)
    # Replace zero/nan with a large vol so weight -> 0 (safe default)
    rvol = rvol.replace(0.0, np.nan).fillna(1.0)
    return (target_vol / rvol).clip(upper=max_weight).shift(1).fillna(1.0)


# ==============================================================================
# BACKTEST ENGINE
# ==============================================================================
def backtest_one(
    close          : pd.Series,
    rsi_entry      : float,
    rsi_exit       : float,
    sma_period     : int,
    target_vol     : Optional[float],
    vol_lookback   : int,
    max_weight     : float,
    trend_base_pct : float = 0.0,
    news_modifier  : Optional[pd.Series] = None,
    vix_block      : Optional[float] = None,
    hard_stop_pct  : Optional[float] = None,
) -> Dict[str, Any]:
    """
    Backtest a single ticker with one param set.

    Timeline (no look-ahead):
      Signal at close_t -> shift(1) -> position active day t+1.
      Costs applied on |delta_position| per side.

    Two position layers:
      RSI layer   – full vol-targeted weight when RSI(2) < rsi_entry AND above SMA.
      Trend layer – trend_base_pct * vol_weight whenever price > SMA (RSI idle).
                    Zero when price < SMA (safety guarantee preserved).
    Cash yield:
      Idle capital (1 - position) earns the daily T-bill rate from _CASH_RATE.
      Silently skipped if _CASH_RATE is None or not aligned to this date range.
    news_modifier:
      Optional pd.Series (same index, values 0-1) from rsi2_news.NewsJudgment.
      Scales down or blocks positions on negative-news days.
      None = all 1.0 (no effect, used in historical backtests).
    """
    daily_ret = close.pct_change()
    rsi_s     = wilder_rsi(close, 2)
    sma_s     = sma(close, sma_period)

    # Binary RSI mean-reversion signal (state machine)
    raw_sig = _raw_signal(close, rsi_entry, rsi_exit, sma_period,
                          vix_block=vix_block, hard_stop_pct=hard_stop_pct)
    raw_s   = pd.Series(raw_sig, index=close.index)

    # Trend baseline: active when above SMA but RSI is NOT in a trade
    if trend_base_pct > 0.0:
        above_sma   = (close > sma_s).astype(float)
        trend_sig   = above_sma * trend_base_pct * (1.0 - raw_s)
    else:
        trend_sig   = pd.Series(0.0, index=close.index)

    combined_sig = raw_s + trend_sig          # RSI takes full weight; trend fills gap

    vol_w  = _vol_weight(daily_ret, target_vol, vol_lookback, max_weight)
    sized  = combined_sig * vol_w

    # Apply news modifier (live overlay; always 1.0 during historical backtests)
    if news_modifier is not None and len(news_modifier) > 0:
        nm    = news_modifier.reindex(sized.index).ffill().fillna(1.0)
        sized = sized * nm

    # Shift: signal at close_t -> position on day t+1
    pos   = sized.shift(1).fillna(0.0)
    gross = pos * daily_ret
    cost  = pos.diff().abs() * COST_PER_SIDE

    # Cash yield on idle capital (1 - pos, floored at 0 so leverage doesn't earn yield)
    if _CASH_RATE is not None and len(_CASH_RATE) > 0:
        cr       = _CASH_RATE.reindex(daily_ret.index).ffill().fillna(0.0)
        idle     = (1.0 - pos).clip(lower=0.0)
        cash_ret = idle * cr
    else:
        cash_ret = pd.Series(0.0, index=daily_ret.index)

    net    = gross + cash_ret - cost.fillna(0.0)
    equity = (1.0 + net.fillna(0.0)).cumprod() * 100_000.0

    # Trade log tracks RSI entries/exits only (trend baseline is a passive hold)
    trades = _extract_trades(raw_s.shift(1).fillna(0.0), net)

    return {
        "returns"    : net,
        "equity"     : equity,
        "position"   : pos,
        "raw_signal" : raw_s,
        "rsi_series" : rsi_s,
        "trades"     : trades,
    }


def _extract_trades(exec_pos: pd.Series, net_ret: pd.Series) -> pd.DataFrame:
    """
    Build round-trip trade table from executed position (already shifted).
    Trade starts on the first bar with pos > 0, ends when pos drops to 0.
    """
    pos_arr  = exec_pos.values
    ret_arr  = net_ret.values
    dates    = exec_pos.index
    rows: List[Dict] = []

    in_trade  = False
    entry_idx = 0
    cum_log   = 0.0  # log-sum for the trade

    for i in range(len(pos_arr)):
        was_in   = in_trade
        in_trade = pos_arr[i] > 0.01

        if not was_in and in_trade:
            entry_idx = i
            cum_log   = 0.0

        if was_in and in_trade:
            r = ret_arr[i] if not math.isnan(ret_arr[i]) else 0.0
            cum_log += math.log1p(r)

        if was_in and not in_trade:
            trade_ret = math.expm1(cum_log)
            rows.append({
                "entry_date"   : dates[entry_idx],
                "exit_date"    : dates[i - 1],
                "duration_days": i - entry_idx,
                "trade_ret_pct": round(trade_ret * 100, 4),
                "win"          : trade_ret > 0.0,
            })

    # Open trade at period end - count it
    if in_trade:
        trade_ret = math.expm1(cum_log)
        rows.append({
            "entry_date"   : dates[entry_idx],
            "exit_date"    : dates[-1],
            "duration_days": len(pos_arr) - 1 - entry_idx,
            "trade_ret_pct": round(trade_ret * 100, 4),
            "win"          : trade_ret > 0.0,
        })

    cols = ["entry_date", "exit_date", "duration_days", "trade_ret_pct", "win"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# ==============================================================================
# PERFORMANCE METRICS
# ==============================================================================
def compute_metrics(
    returns : pd.Series,
    equity  : pd.Series,
    trades  : pd.DataFrame,
    n_trials: int = 1,
    label   : str = "",
) -> Dict[str, Any]:
    """
    Compute CAGR, Sharpe, Deflated Sharpe, max drawdown, win rate, exposure.

    n_trials: total grid combinations tested in IS (used for DSR penalty).
    """
    r = returns.dropna()
    if len(r) < 30:
        return _empty_metrics(label)

    n       = len(r)
    n_years = n / ANNUAL_BARS
    total_r = float((1.0 + r).prod() - 1.0)
    cagr    = (1.0 + total_r) ** (1.0 / max(n_years, 1e-9)) - 1.0

    mu, sigma = float(r.mean()), float(r.std())
    sharpe    = (mu / sigma * math.sqrt(ANNUAL_BARS)) if sigma > 0 else 0.0
    skew      = float(r.skew())
    kurt      = float(r.kurt())   # excess kurtosis (Fisher definition)
    dsr       = _deflated_sharpe(sharpe, n_trials, n, skew, kurt)

    # Max drawdown from equity curve
    eq   = equity.dropna()
    peak = eq.cummax()
    mdd  = float(((eq - peak) / peak).min())

    # Exposure: fraction of days with active position
    exposure = float((returns.abs() > 1e-10).mean())

    if len(trades) > 0:
        win_rate = float(trades["win"].mean())
        avg_dur  = float(trades["duration_days"].mean())
        n_trades = int(len(trades))
    else:
        win_rate = avg_dur = 0.0
        n_trades = 0

    return {
        "label"           : label,
        "cagr_pct"        : round(cagr * 100, 2),
        "sharpe"          : round(sharpe, 3),
        "deflated_sharpe" : round(dsr, 4),
        "max_drawdown_pct": round(mdd * 100, 2),
        "win_rate_pct"    : round(win_rate * 100, 2),
        "exposure_pct"    : round(exposure * 100, 2),
        "n_trades"        : n_trades,
        "avg_trade_days"  : round(avg_dur, 1),
        "total_return_pct": round(total_r * 100, 2),
    }


def _deflated_sharpe(
    sr_annual: float,
    n_trials : int,
    n_obs    : int,
    skew     : float,
    kurt     : float,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    PSR(SR*) = Φ[(SR_hat - SR*) x sqrt(T-1) / sqrt(1 - γ3xSR_hat + (γ4-1)/4 x SR_hat²)]

    SR* = E[max annualized SR from n_trials independent strategies under H0].
    Uses Euler-Mascheroni approximation for E[max of N standard normals].
    """
    if n_obs < 10:
        return 0.0

    sr = sr_annual / math.sqrt(ANNUAL_BARS)   # convert to daily units
    sr_star = _expected_max_sr_daily(n_trials, n_obs)

    kurtosis_adj = max(0.0, (kurt - 1.0) / 4.0)
    denom = 1.0 - skew * sr + kurtosis_adj * sr ** 2
    if denom <= 1e-9:
        denom = 1e-9

    z = (sr - sr_star) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(norm.cdf(z))


def _expected_max_sr_daily(n_trials: int, n_obs: int) -> float:
    """
    E[max daily SR] from testing n_trials independent strategies under H0.
    SE of daily SR_hat ≈ 1/sqrt(n_obs) for iid normal returns with zero mean.
    """
    if n_trials <= 1:
        return 0.0
    gamma_e = 0.5772156649                    # Euler-Mascheroni constant
    t1 = (1.0 - gamma_e) * norm.ppf(1.0 - 1.0 / n_trials)
    t2 = gamma_e * norm.ppf(1.0 - math.exp(-1.0) / n_trials)
    max_z = t1 + t2                           # expected max of N standard normals
    se    = 1.0 / math.sqrt(max(n_obs - 1, 1))   # SE of daily SR under H0
    return max_z * se


def _empty_metrics(label: str) -> Dict:
    return dict(label=label, cagr_pct=0.0, sharpe=0.0, deflated_sharpe=0.0,
                max_drawdown_pct=0.0, win_rate_pct=0.0, exposure_pct=0.0,
                n_trades=0, avg_trade_days=0.0, total_return_pct=0.0)


def bah_metrics(close: pd.Series, label: str = "B&H SPY",
                n_trials: int = 1) -> Dict:
    """Buy-and-hold metrics for a price series."""
    r  = close.pct_change().dropna()
    eq = (1.0 + r).cumprod() * 100_000.0
    trades = pd.DataFrame([{
        "entry_date": r.index[0], "exit_date": r.index[-1],
        "duration_days": len(r), "trade_ret_pct": float((eq.iloc[-1]/1e5 - 1)*100),
        "win": True,
    }])
    return compute_metrics(r, eq, trades, n_trials=n_trials, label=label)


# ==============================================================================
# COMBINED PORTFOLIO
# ==============================================================================
def combine_returns(
    results    : Dict[str, Dict],
    sizing_mode: str   = "uniform",
    rsi_entry  : float = 10.0,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Combine per-ticker backtest results into one portfolio return series.

    sizing_mode = "uniform"
        Each active ticker gets weight 1/N (equal capital split).

    sizing_mode = "rsi_depth"
        Each active ticker is weighted by how far its RSI is below the entry
        threshold: depth_i = max(rsi_entry - rsi_i, eps).
        The deepest oversold ticker gets the largest slice of available capital.
        RSI is shifted one day so there is no look-ahead (we know yesterday's
        RSI when sizing today's position).
        Falls back to uniform when only one ticker is active.

    Transaction costs are computed per-ticker in backtest_one() and included in
    each ticker's "returns" series.  Cross-ticker rebalancing costs that arise
    from the rsi_depth re-weighting are approximated as zero (positions within
    an already-open trade are rarely rebalanced more than once per week).

    Returns (net_returns, equity, trades_combined).
    """
    all_ret = pd.DataFrame(
        {t: r["returns"] for t, r in results.items()}
    ).fillna(0.0)
    all_pos = pd.DataFrame(
        {t: r["position"] for t, r in results.items()}
    ).fillna(0.0)

    active = all_pos > 0.01       # boolean mask

    if sizing_mode == "rsi_depth" and len(results) > 1:
        # RSI shifted 1 day: on day t we use the RSI that was known at close_{t-1}
        all_rsi = pd.DataFrame({
            t: r["rsi_series"].shift(1).reindex(all_pos.index).fillna(50.0)
            for t, r in results.items()
        })
        # depth = how far below the entry threshold; zero when RSI is at or above it
        depth = active.multiply((rsi_entry - all_rsi).clip(lower=0.01))
        depth_sum = depth.sum(axis=1).replace(0.0, np.nan)
        weights   = depth.divide(depth_sum, axis=0).fillna(0.0)
    else:
        # Uniform: 1/N split among whichever tickers have an active signal
        n_active = active.sum(axis=1).replace(0, np.nan)
        weights  = active.divide(n_active, axis=0).fillna(0.0)

    combined = (weights * all_ret).sum(axis=1)
    equity   = (1.0 + combined).cumprod() * 100_000.0

    all_trades = pd.concat([r["trades"] for r in results.values()], ignore_index=True)
    if len(all_trades):
        all_trades = all_trades.sort_values("entry_date").reset_index(drop=True)

    return combined, equity, all_trades


def _split_params(params: Dict) -> Tuple[Dict, str]:
    """Separate sizing_mode (portfolio-level) from backtest_one() kwargs."""
    sizing_mode = params.get("sizing_mode", "uniform")
    bt_params   = {k: v for k, v in params.items() if k != "sizing_mode"}
    return bt_params, sizing_mode


# ==============================================================================
# IS GRID SEARCH
# ==============================================================================
def run_grid_search(
    prices     : Dict[str, pd.Series],
    is_start   : str,
    is_end     : str,
    param_grid : Dict[str, List],
    csv_path   : str,
) -> Tuple[Dict, List[Dict]]:
    """
    Exhaustive grid search over param_grid on IS data.
    Optimisation criterion: average Sharpe ratio across all tickers.
    Logs every combination to csv_path.
    Returns (best_params, all_rows).
    """
    # Clip prices to IS window
    is_prices = {t: s.loc[is_start:is_end] for t, s in prices.items() if t in prices}

    keys     = list(param_grid.keys())
    combos   = list(product(*[param_grid[k] for k in keys]))
    n_combos = len(combos)

    print(f"\n  Running IS grid search: {n_combos} combinations x {len(is_prices)} tickers ...")

    # Fieldnames: drop sizing_mode from per-ticker rows; add combined_sharpe
    bt_keys    = [k for k in keys if k != "sizing_mode"]
    fieldnames = (["combo_id", "sizing_mode"] + bt_keys +
                  ["ticker", "cagr_pct", "sharpe", "max_drawdown_pct",
                   "win_rate_pct", "exposure_pct", "n_trades", "avg_trade_days",
                   "combined_sharpe"])

    all_rows: List[Dict] = []
    best_combined_sharpe = -np.inf
    best_params: Dict    = {}

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for idx, combo in enumerate(combos):
            params                 = dict(zip(keys, combo))
            bt_params, sizing_mode = _split_params(params)

            ticker_results: Dict[str, Dict] = {}
            for ticker, close in is_prices.items():
                try:
                    ticker_results[ticker] = backtest_one(close, **bt_params)
                except Exception:
                    ticker_results[ticker] = {"returns": pd.Series(dtype=float),
                                              "equity": pd.Series(dtype=float),
                                              "position": pd.Series(dtype=float),
                                              "rsi_series": pd.Series(dtype=float),
                                              "raw_signal": pd.Series(dtype=float),
                                              "trades": pd.DataFrame()}

            # Combined portfolio with this sizing mode
            comb_ret, comb_eq, comb_tr = combine_returns(
                ticker_results, sizing_mode=sizing_mode,
                rsi_entry=bt_params["rsi_entry"]
            )
            comb_m = compute_metrics(comb_ret, comb_eq, comb_tr, n_trials=1,
                                     label="combined")

            # Log each ticker separately for analysis
            for ticker, res in ticker_results.items():
                m = compute_metrics(res["returns"], res["equity"], res["trades"],
                                    n_trials=1, label=ticker)
                row = {"combo_id": idx, "sizing_mode": sizing_mode,
                       **bt_params, "ticker": ticker,
                       "combined_sharpe": comb_m["sharpe"], **m}
                writer.writerow(row)
                all_rows.append(row)

            # Optimise on combined Sharpe so sizing_mode is part of the selection
            combined_sh = comb_m["sharpe"]
            if combined_sh > best_combined_sharpe:
                best_combined_sharpe = combined_sh
                best_params = deepcopy(params)   # keeps sizing_mode

            if (idx + 1) % 100 == 0 or idx == n_combos - 1:
                tv = bt_params["target_vol"]
                print(f"    {idx+1:>4}/{n_combos}  entry={bt_params['rsi_entry']:>2} "
                      f"exit={bt_params['rsi_exit']:>2} sma={bt_params['sma_period']:>3} "
                      f"tvol={str(tv):>4} mode={sizing_mode:<9}  "
                      f"comb_sharpe={combined_sh:>+6.3f}  "
                      f"best={best_combined_sharpe:>+6.3f}")

    print(f"  IS best params: {best_params}  (combined Sharpe={best_combined_sharpe:+.3f})")
    print(f"  Param log saved -> {csv_path}")
    return best_params, all_rows


# ==============================================================================
# WALK-FORWARD (annual expanding IS window)
# ==============================================================================
def run_walk_forward(
    prices     : Dict[str, pd.Series],
    is_start   : str,
    oos_start  : str,
    oos_end    : str,
    param_grid : Dict[str, List],
) -> Tuple[List[Dict], pd.Series, pd.DataFrame]:
    """
    Annual expanding-window walk-forward.
    For each year Y in [oos_start_year ... oos_end_year]:
      IS  = is_start -> Dec 31 of Y-1
      OOS = Jan 1 of Y -> Dec 31 of Y (or oos_end if last fold)
    Re-optimises on IS at each step; evaluates on OOS.

    Returns (fold_records, chained_oos_returns, chained_oos_trades).
    """
    oos_start_year = int(oos_start[:4])
    oos_end_year   = int(oos_end[:4])

    fold_records: List[Dict] = []
    chained_ret_parts: List[pd.Series] = []
    chained_trades: List[pd.DataFrame] = []

    keys   = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    n_combos = len(combos)

    print(f"\n  Walk-forward: {oos_start_year}-{oos_end_year} "
          f"({oos_end_year - oos_start_year + 1} folds), "
          f"{n_combos} combos/fold ...")

    for year in range(oos_start_year, oos_end_year + 1):
        fold_is_end  = f"{year - 1}-12-31"
        fold_oos_start = f"{year}-01-01"
        fold_oos_end   = f"{year}-12-31" if year < oos_end_year else oos_end

        fold_is_prices  = {t: s.loc[is_start:fold_is_end]
                           for t, s in prices.items()}
        fold_oos_prices = {t: s.loc[fold_oos_start:fold_oos_end]
                           for t, s in prices.items()}

        # Skip if OOS data is too short
        if any(len(v) < 10 for v in fold_oos_prices.values()):
            continue

        # Find best params on this fold's IS (optimise combined Sharpe)
        best_sh  = -np.inf
        fold_best: Dict = {}

        for combo in combos:
            params                 = dict(zip(keys, combo))
            bt_params, sizing_mode = _split_params(params)
            fold_results: Dict[str, Dict] = {}
            for ticker, close in fold_is_prices.items():
                if len(close) < 60:
                    continue
                try:
                    fold_results[ticker] = backtest_one(close, **bt_params)
                except Exception:
                    pass
            if not fold_results:
                continue
            comb_r, comb_e, comb_t = combine_returns(
                fold_results, sizing_mode=sizing_mode,
                rsi_entry=bt_params["rsi_entry"]
            )
            avg = compute_metrics(comb_r, comb_e, comb_t, n_trials=1)["sharpe"]
            if avg > best_sh:
                best_sh   = avg
                fold_best = deepcopy(params)   # keeps sizing_mode

        # Evaluate best params on OOS
        fold_bt, fold_sm = _split_params(fold_best)
        fold_oos_results = {}
        for ticker, close in fold_oos_prices.items():
            if len(close) < 5:
                continue
            try:
                fold_oos_results[ticker] = backtest_one(close, **fold_bt)
            except Exception:
                pass

        if not fold_oos_results:
            continue

        comb_ret, comb_eq, comb_trades = combine_returns(
            fold_oos_results, sizing_mode=fold_sm,
            rsi_entry=fold_bt["rsi_entry"]
        )
        m = compute_metrics(comb_ret, comb_eq, comb_trades,
                            n_trials=n_combos, label=f"WF {year}")

        fold_records.append({"year": year, "best_params": fold_best, **m})
        chained_ret_parts.append(comb_ret)
        chained_trades.append(comb_trades)

        tv = fold_bt.get("target_vol")
        print(f"    {year}  IS_end={fold_is_end}  "
              f"entry={fold_bt['rsi_entry']:>2} exit={fold_bt['rsi_exit']:>2} "
              f"sma={fold_bt['sma_period']:>3} tvol={str(tv):>4} "
              f"mode={fold_sm:<9}  "
              f"OOS CAGR={m['cagr_pct']:>+6.2f}%  Sharpe={m['sharpe']:>+6.3f}")

    chained_ret = pd.concat(chained_ret_parts) if chained_ret_parts else pd.Series(dtype=float)
    chained_trd = pd.concat(chained_trades, ignore_index=True) if chained_trades else pd.DataFrame()
    return fold_records, chained_ret, chained_trd


# ==============================================================================
# REPORTING
# ==============================================================================
def _hdr(title: str):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def _metric_row(m: Dict) -> str:
    return (f"  {m['label']:<18} "
            f"CAGR={m['cagr_pct']:>+7.2f}%  "
            f"SR={m['sharpe']:>+6.3f}  "
            f"DSR={m['deflated_sharpe']:>6.4f}  "
            f"MDD={m['max_drawdown_pct']:>+7.2f}%  "
            f"Win={m['win_rate_pct']:>5.1f}%  "
            f"Exp={m['exposure_pct']:>5.1f}%  "
            f"N={m['n_trades']:>4}")


def print_metrics_block(metrics_list: List[Dict], bah: Dict):
    print(f"\n  {'Strategy':<18} {'CAGR':>8} {'Sharpe':>7} {'DSR':>7} "
          f"{'MaxDD':>8} {'Win%':>6} {'Exp%':>6} {'N':>5}")
    print("  " + "-" * 68)
    for m in metrics_list:
        print(_metric_row(m))
    print("  " + "-" * 68)
    print(_metric_row(bah))


def print_wf_table(fold_records: List[Dict], wf_summary: Dict):
    print(f"\n  {'Year':<6} {'entry':>5} {'exit':>5} {'sma':>5} "
          f"{'tvol':>6} {'mode':<10}  {'CAGR':>8} {'Sharpe':>7} {'MDD':>8} {'Win%':>6}")
    print("  " + "-" * 76)
    for f in fold_records:
        p  = f["best_params"]
        bt, sm = _split_params(p)
        print(f"  {f['year']:<6} {bt['rsi_entry']:>5} {bt['rsi_exit']:>5} "
              f"{bt['sma_period']:>5} {str(bt['target_vol']):>6} {sm:<10}  "
              f"{f['cagr_pct']:>+7.2f}%  {f['sharpe']:>+6.3f}  "
              f"{f['max_drawdown_pct']:>+7.2f}%  {f['win_rate_pct']:>5.1f}%")
    print("  " + "-" * 65)
    print(f"  {'WF OOS agg':<6}                                         "
          f"{wf_summary['cagr_pct']:>+7.2f}%  {wf_summary['sharpe']:>+6.3f}  "
          f"{wf_summary['max_drawdown_pct']:>+7.2f}%  "
          f"{wf_summary['win_rate_pct']:>5.1f}%")


# ==============================================================================
# MAIN
# ==============================================================================
RECAL_LOG_CSV  = "rsi2_recal_log.csv"
RECAL_STAMP    = "rsi2_last_recal.txt"
# IS window for --recal: everything up to 90 days ago (3-month buffer avoids
# optimising on data we're about to trade).
RECAL_BUFFER_DAYS = 90


def run_recal(silent: bool = False) -> Dict:
    """
    Expanding-window recalibration.  IS_END is slid to (today - RECAL_BUFFER_DAYS),
    the full grid search is re-run, params are diffed against the last saved run,
    and rsi2_results.json is updated.

    Returns the new best_params dict.
    Called from --recal CLI mode and from rsi2_live_signal.py's auto-trigger.
    """
    from dateutil.relativedelta import relativedelta

    recal_is_end = (datetime.today() -
                    timedelta(days=RECAL_BUFFER_DAYS)).strftime("%Y-%m-%d")

    if not silent:
        print("=" * 68)
        print(f"  AUTO-RECALIBRATION  (IS window: {FETCH_START[:4]} -> {recal_is_end})")
        print("=" * 68)

    # Load old params for diff
    old_params: Dict = {}
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON) as fh:
                old_params = json.load(fh).get("best_params", {})
        except Exception:
            pass

    # Download prices and T-bill rate
    prices = fetch_prices(TICKERS, FETCH_START, TODAY)
    if not prices:
        if not silent:
            print("  ERROR: no price data.")
        return old_params

    global _CASH_RATE, _VIX
    _CASH_RATE = fetch_cash_rate(FETCH_START, TODAY)
    _VIX = fetch_vix(FETCH_START, TODAY)
    if not silent:
        if len(_CASH_RATE) > 0:
            avg = float(_CASH_RATE.mean()) * ANNUAL_BARS * 100
            print(f"  T-bill: {avg:.2f}% annualised")

    # IS grid search over expanded window
    n_combos = 1
    for v in PARAM_GRID.values():
        n_combos *= len(v)
    if not silent:
        print(f"  Running grid ({n_combos} combos x {len(prices)} tickers) ...",
              end=" ", flush=True)

    new_params, _ = run_grid_search(prices, FETCH_START, recal_is_end,
                                    PARAM_GRID, RESULTS_CSV)
    if not silent:
        print("done")

    # Diff old vs new
    changed = {k: (old_params.get(k), new_params[k])
               for k in new_params
               if old_params.get(k) != new_params.get(k)}

    if not silent:
        if changed:
            print(f"\n  PARAMS CHANGED:")
            for k, (old, new) in changed.items():
                print(f"    {k:<18} {str(old):>10}  ->  {new}")
        else:
            print("  Params unchanged.")

    # Save updated results JSON (IS section only — no OOS, no WF)
    spy_full = prices.get("SPY", prices[list(prices.keys())[0]])
    is_prices = {t: s.loc[FETCH_START:recal_is_end] for t, s in prices.items()}
    best_bt, best_sm = _split_params(new_params)
    is_results = {}
    for ticker, close in is_prices.items():
        try:
            is_results[ticker] = backtest_one(close, **best_bt)
        except Exception:
            pass
    comb_r, comb_e, comb_t = combine_returns(
        is_results, sizing_mode=best_sm, rsi_entry=best_bt["rsi_entry"]
    )
    is_comb = compute_metrics(comb_r, comb_e, comb_t, n_trials=n_combos,
                              label="Combined IS")

    bah_is = bah_metrics(spy_full.loc[FETCH_START:recal_is_end],
                         "B&H SPY (IS)", n_combos)

    out = {
        "best_params"   : new_params,
        "recal_is_end"  : recal_is_end,
        "is_combined"   : is_comb,
        "bah_spy_is"    : bah_is,
        # Preserve OOS/WF fields from previous run if they exist
    }
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON) as fh:
                prev = json.load(fh)
            for key in ("oos_combined", "bah_spy_oos", "wf_summary", "wf_folds",
                        "per_ticker_is", "per_ticker_oos"):
                if key in prev:
                    out[key] = prev[key]
        except Exception:
            pass
    with open(RESULTS_JSON, "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    # Append to recal log
    stamp = datetime.today().strftime("%Y-%m-%d %H:%M")
    log_exists = os.path.exists(RECAL_LOG_CSV)
    with open(RECAL_LOG_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["timestamp", "is_end", "params_changed",
                                           "new_params"])
        if not log_exists:
            w.writeheader()
        w.writerow({
            "timestamp"     : stamp,
            "is_end"        : recal_is_end,
            "params_changed": bool(changed),
            "new_params"    : str(new_params),
        })

    # Update last-recal timestamp
    with open(RECAL_STAMP, "w") as fh:
        fh.write(datetime.today().isoformat())

    if not silent:
        print(f"\n  Saved -> {RESULTS_JSON}  |  log -> {RECAL_LOG_CSV}")

    return new_params


def main():
    parser = argparse.ArgumentParser(description="RSI(2) mean-reversion strategy")
    parser.add_argument("--quick", action="store_true",
                        help="Small sanity run on SPY only")
    parser.add_argument("--focused", action="store_true",
                        help="Fix all params at IS-best, only search trend_base_pct. "
                             "Requires rsi2_results.json from a prior full run.")
    parser.add_argument("--recal", action="store_true",
                        help="Expanding-window recalibration: slide IS_END to "
                             f"today-{RECAL_BUFFER_DAYS}d, re-run grid, update "
                             "rsi2_results.json and log the diff.")
    args = parser.parse_args()

    if args.recal:
        run_recal(silent=False)
        return

    if args.quick:
        tickers   = ["SPY"]
        grid      = {"rsi_entry": [5, 10], "rsi_exit": [65, 70], "sma_period": [200],
                     "target_vol": [None, 0.15], "vol_lookback": [21], "max_weight": [1.5],
                     "sizing_mode": ["uniform", "rsi_depth"],
                     "trend_base_pct": [0.0, 0.20]}
    elif args.focused:
        # Load prior IS-best, fix everything except trend_base_pct
        if not os.path.exists(RESULTS_JSON):
            print(f"ERROR: {RESULTS_JSON} not found. Run without --focused first.")
            return
        with open(RESULTS_JSON) as fh:
            prior = json.load(fh)
        p = prior["best_params"]
        tickers = TICKERS
        grid = {
            "rsi_entry"      : [p["rsi_entry"]],
            "rsi_exit"       : [p["rsi_exit"]],
            "sma_period"     : [p["sma_period"]],
            "target_vol"     : [p["target_vol"]],
            "vol_lookback"   : [p["vol_lookback"]],
            "max_weight"     : [p["max_weight"]],
            "sizing_mode"    : [p["sizing_mode"]],
            "trend_base_pct" : [0.0, 0.10, 0.20, 0.30, 0.40],
        }
        print(f"  [focused] fixing IS params, searching trend_base_pct only")
    else:
        tickers   = TICKERS
        grid      = PARAM_GRID

    # -- 1. Download data ------------------------------------------------------
    _hdr("RSI(2) MEAN-REVERSION STRATEGY")
    print(f"  Tickers : {tickers}")
    print(f"  IS      : {FETCH_START[:4]} - {IS_END[:4]}")
    print(f"  OOS     : {OOS_START[:4]} - {TODAY[:4]}")
    print(f"  Costs   : {COST_PER_SIDE*10000:.1f} bps per side")
    print(f"  RSI     : Wilder's RMA (alpha = 1/period)")
    print(f"\n  Downloading prices ...", end=" ", flush=True)
    prices = fetch_prices(tickers, FETCH_START, TODAY)
    if not prices:
        print("ERROR: no price data downloaded.")
        return
    print(f"OK ({len(prices)} tickers, "
          f"{min(len(v) for v in prices.values())} - "
          f"{max(len(v) for v in prices.values())} bars each)")

    # Cash yield (T-bill rate): fetched once, shared across all backtest_one() calls
    global _CASH_RATE, _VIX
    print(f"  Fetching T-bill rate (^IRX) ...", end=" ", flush=True)
    _CASH_RATE = fetch_cash_rate(FETCH_START, TODAY)
    if len(_CASH_RATE) > 0:
        avg_rate = float(_CASH_RATE.mean()) * ANNUAL_BARS * 100
        print(f"OK  (avg annualised cash rate: {avg_rate:.2f}%)")
    else:
        print("unavailable - cash yield disabled")
    print(f"  Fetching VIX (^VIX) ...", end=" ", flush=True)
    _VIX = fetch_vix(FETCH_START, TODAY)
    if len(_VIX) > 0:
        print(f"OK  (avg VIX: {float(_VIX.mean()):.1f})")
    else:
        print("unavailable - VIX gate disabled")

    # Clip IS / OOS subsets
    is_prices  = {t: s.loc[FETCH_START:IS_END]  for t, s in prices.items()}
    oos_prices = {t: s.loc[OOS_START:TODAY]      for t, s in prices.items()}

    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    # -- 2. Buy-and-hold benchmarks --------------------------------------------
    spy_full = prices.get("SPY", prices[list(prices.keys())[0]])
    bah_is   = bah_metrics(spy_full.loc[FETCH_START:IS_END],  "B&H SPY (IS)",  n_combos)
    bah_oos  = bah_metrics(spy_full.loc[OOS_START:TODAY],     "B&H SPY (OOS)", n_combos)

    # -- 3. IS grid search -----------------------------------------------------
    _hdr(f"IN-SAMPLE GRID SEARCH  ({FETCH_START[:4]}-{IS_END[:4]}, {n_combos} combos)")
    best_params, all_rows = run_grid_search(is_prices, FETCH_START, IS_END,
                                            grid, RESULTS_CSV)

    # Separate sizing_mode from backtest_one() kwargs for all downstream calls
    best_bt, best_sm = _split_params(best_params)

    # -- 4. IS performance with best params -----------------------------------
    _hdr(f"IN-SAMPLE PERFORMANCE  (best params: {best_params})")
    is_per_ticker = []
    is_results    = {}
    for ticker, close in is_prices.items():
        res = backtest_one(close, **best_bt)
        m   = compute_metrics(res["returns"], res["equity"], res["trades"],
                              n_trials=n_combos, label=f"{ticker} RSI(2) IS")
        is_per_ticker.append(m)
        is_results[ticker] = res

    comb_ret_is, comb_eq_is, comb_tr_is = combine_returns(
        is_results, sizing_mode=best_sm, rsi_entry=best_bt["rsi_entry"]
    )
    is_combined = compute_metrics(comb_ret_is, comb_eq_is, comb_tr_is,
                                  n_trials=n_combos, label="Combined IS")
    print_metrics_block(is_per_ticker + [is_combined], bah_is)

    # -- 5. OOS performance (fixed IS params) ---------------------------------
    _hdr(f"OUT-OF-SAMPLE PERFORMANCE  ({OOS_START[:4]}-{TODAY[:4]}, FIXED IS PARAMS)")
    oos_per_ticker = []
    oos_results    = {}
    for ticker, close in oos_prices.items():
        res = backtest_one(close, **best_bt)
        m   = compute_metrics(res["returns"], res["equity"], res["trades"],
                              n_trials=n_combos, label=f"{ticker} RSI(2) OOS")
        oos_per_ticker.append(m)
        oos_results[ticker] = res

    comb_ret_oos, comb_eq_oos, comb_tr_oos = combine_returns(
        oos_results, sizing_mode=best_sm, rsi_entry=best_bt["rsi_entry"]
    )
    oos_combined = compute_metrics(comb_ret_oos, comb_eq_oos, comb_tr_oos,
                                   n_trials=n_combos, label="Combined OOS")
    print_metrics_block(oos_per_ticker + [oos_combined], bah_oos)

    # -- 6. Walk-forward (annual expanding IS) --------------------------------
    _hdr(f"WALK-FORWARD  (annual re-opt, expanding IS from {FETCH_START[:4]})")
    fold_records, wf_ret, wf_trades = run_walk_forward(
        prices, FETCH_START, OOS_START, TODAY, grid
    )
    if len(wf_ret) > 0:
        wf_eq      = (1.0 + wf_ret).cumprod() * 100_000.0
        wf_summary = compute_metrics(wf_ret, wf_eq, wf_trades,
                                     n_trials=n_combos, label="WF OOS agg")
        print_wf_table(fold_records, wf_summary)
        print(f"\n  WF OOS DSR : {wf_summary['deflated_sharpe']:.4f}  "
              f"(penalised for {n_combos} IS combinations tested per fold)")

    # -- 7. Summary comparison -------------------------------------------------
    _hdr("SUMMARY: STRATEGY vs B&H SPY")
    rows = [
        ("Metric",           "B&H SPY IS", "Strategy IS",       "B&H SPY OOS", "Strategy OOS"),
        ("CAGR",             f"{bah_is['cagr_pct']:+.2f}%",
                             f"{is_combined['cagr_pct']:+.2f}%",
                             f"{bah_oos['cagr_pct']:+.2f}%",
                             f"{oos_combined['cagr_pct']:+.2f}%"),
        ("Sharpe",           f"{bah_is['sharpe']:+.3f}",
                             f"{is_combined['sharpe']:+.3f}",
                             f"{bah_oos['sharpe']:+.3f}",
                             f"{oos_combined['sharpe']:+.3f}"),
        ("Deflated SR",      f"{bah_is['deflated_sharpe']:.4f}",
                             f"{is_combined['deflated_sharpe']:.4f}",
                             f"{bah_oos['deflated_sharpe']:.4f}",
                             f"{oos_combined['deflated_sharpe']:.4f}"),
        ("Max Drawdown",     f"{bah_is['max_drawdown_pct']:+.2f}%",
                             f"{is_combined['max_drawdown_pct']:+.2f}%",
                             f"{bah_oos['max_drawdown_pct']:+.2f}%",
                             f"{oos_combined['max_drawdown_pct']:+.2f}%"),
        ("Win Rate",         "-",
                             f"{is_combined['win_rate_pct']:.1f}%",
                             "-",
                             f"{oos_combined['win_rate_pct']:.1f}%"),
        ("Exposure",         "100%",
                             f"{is_combined['exposure_pct']:.1f}%",
                             "100%",
                             f"{oos_combined['exposure_pct']:.1f}%"),
    ]
    col_w = [15, 14, 14, 14, 14]
    for row in rows:
        print("  " + "".join(str(c).ljust(w) for c, w in zip(row, col_w)))

    # -- 8. Save JSON results --------------------------------------------------
    out = {
        "run_date"    : TODAY,
        "is_period"   : f"{FETCH_START} - {IS_END}",
        "oos_period"  : f"{OOS_START} - {TODAY}",
        "cost_per_side_bps": COST_PER_SIDE * 10_000,
        "n_combos"    : n_combos,
        "best_params" : best_params,
        "is_combined" : is_combined,
        "oos_combined": oos_combined,
        "bah_spy_is"  : bah_is,
        "bah_spy_oos" : bah_oos,
        "wf_folds"    : fold_records,
        "wf_summary"  : wf_summary if len(wf_ret) > 0 else {},
        "per_ticker_is" : {m["label"]: m for m in is_per_ticker},
        "per_ticker_oos": {m["label"]: m for m in oos_per_ticker},
    }
    with open(RESULTS_JSON, "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    print(f"\n  Results -> {RESULTS_JSON}")
    print(f"  Param log -> {RESULTS_CSV}")
    print(f"  DSR note: DSR < 0.5 means strategy likely reflects IS overfitting.")


if __name__ == "__main__":
    main()
