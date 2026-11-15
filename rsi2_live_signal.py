"""
RSI(2) daily live signal - run each morning before the open.

Loads best_params from rsi2_results.json, downloads the latest prices,
computes current RSI(2) / SMA values, scores today's news via FinBERT,
and prints a clear action table: BUY / HOLD / EXIT / BLOCKED.

Usage:
  python rsi2_live_signal.py              # uses today's date
  python rsi2_live_signal.py --date 2026-11-07
"""

import os, sys, json, argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import rsi2_strategy as _strat
from rsi2_strategy import (
    TICKERS, ANNUAL_BARS, RESULTS_JSON, COST_PER_SIDE,
    RECAL_STAMP, fetch_prices, fetch_cash_rate, fetch_vix, wilder_rsi, sma,
    _split_params, run_recal,
)
from rsi2_news import NewsJudgment

TODAY = datetime.today().strftime("%Y-%m-%d")

# ── auto-recalibration ────────────────────────────────────────────────────────
RECAL_INTERVAL_DAYS = 14   # re-run grid search every 2 weeks


def maybe_recalibrate(force: bool = False) -> bool:
    """
    Check whether a recalibration is due.  If so, run it and return True.
    Skipped (returns False) if the last run was < RECAL_INTERVAL_DAYS ago
    and force=False.
    """
    days_since = _days_since_recal()

    if not force and days_since is not None and days_since < RECAL_INTERVAL_DAYS:
        print(f"  Params current (last recal {days_since}d ago, "
              f"next in {RECAL_INTERVAL_DAYS - days_since}d)")
        return False

    if days_since is None:
        print("  No prior recalibration found - running initial calibration ...")
    else:
        print(f"  Recalibration due ({days_since}d since last run) - updating params ...")

    run_recal(silent=False)
    return True


def _days_since_recal() -> int | None:
    """Return days since last recal, or None if stamp file is missing."""
    if not os.path.exists(RECAL_STAMP):
        return None
    try:
        last = datetime.fromisoformat(open(RECAL_STAMP).read().strip())
        return (datetime.today() - last).days
    except Exception:
        return None

# ── helpers ───────────────────────────────────────────────────────────────────
def load_best_params(json_path: str) -> dict:
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"{json_path} not found - run rsi2_strategy.py first."
        )
    with open(json_path) as fh:
        data = json.load(fh)
    return data["best_params"]


def _bar(value: float, lo: float, hi: float, width: int = 20) -> str:
    t = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(t * width + 0.5)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _rsi_state(rsi_val: float, entry: float, exit_: float) -> str:
    if rsi_val < entry:
        return f"OVERSOLD  (< {entry})"
    elif rsi_val > exit_:
        return f"OVERBOUGHT (> {exit_})"
    else:
        return f"NEUTRAL   ({entry}-{exit_})"


# ── signal logic (matches backtest_one, single-bar version) ──────────────────
def compute_live_signal(
    close         : pd.Series,
    rsi_entry     : float,
    rsi_exit      : float,
    sma_period    : int,
    target_vol    : float,
    vol_lookback  : int,
    max_weight    : float,
    trend_base_pct: float,
    news_modifier : float,
    vix_block     : float = None,
    hard_stop_pct : float = None,
) -> dict:
    """
    Compute today's signal for a single ticker.
    Returns a dict with current indicator values and recommended action.
    """
    daily_ret = close.pct_change()
    rsi_s = wilder_rsi(close, 2)
    sma_s = sma(close, sma_period)

    last_close = float(close.iloc[-1])
    last_rsi   = float(rsi_s.iloc[-1])
    last_sma   = float(sma_s.iloc[-1])
    above_sma  = last_close > last_sma

    # Realised vol
    if target_vol is not None:
        rvol_ann = float(daily_ret.rolling(vol_lookback).std().iloc[-1]) * np.sqrt(ANNUAL_BARS)
        vol_w    = min(target_vol / max(rvol_ann, 1e-6), max_weight) if rvol_ann > 0 else 1.0
    else:
        vol_w = 1.0

    # RSI signal (look at yesterday's close as the signal - today's position)
    prev_rsi   = float(rsi_s.iloc[-2]) if len(rsi_s) >= 2 else last_rsi
    prev_close = float(close.iloc[-2])  if len(close) >= 2 else last_close
    prev_sma   = float(sma_s.iloc[-2])  if len(sma_s) >= 2 else last_sma

    # RSI signal: were we in a trade coming into today?
    # Reconstruct last bar of _raw_signal (including VIX gate and hard stop)
    rsi_arr   = rsi_s.values
    sma_arr   = sma_s.values
    close_arr = close.values

    # VIX series aligned to this ticker
    vix_arr = None
    if vix_block is not None and _strat._VIX is not None and len(_strat._VIX) > 0:
        vix_arr = _strat._VIX.reindex(close.index).ffill().fillna(20.0).values

    state    = False
    entry_px = 0.0
    for i in range(len(rsi_arr)):
        c, r, sm = close_arr[i], rsi_arr[i], sma_arr[i]
        if not state:
            if vix_arr is not None and vix_arr[i] >= vix_block:
                continue
            if r < rsi_entry and c > sm:
                state    = True
                entry_px = c
        else:
            if hard_stop_pct is not None and entry_px > 0:
                if c < entry_px * (1.0 - hard_stop_pct):
                    state = False
                    continue
            if r > rsi_exit or c < sm:
                state = False
    rsi_signal = state   # True = currently in an RSI trade
    raw_sig    = 1.0 if rsi_signal else 0.0

    # Trend baseline signal for today
    trend_sig = (trend_base_pct * (1.0 - raw_sig)) if above_sma else 0.0

    combined = raw_sig + trend_sig
    sized    = combined * vol_w * news_modifier

    # Action
    if not above_sma:
        action = "FLAT"
        reason = f"Price below SMA({sma_period}) - trend filter OFF"
    elif news_modifier == 0.0:
        action = "BLOCKED"
        reason = "News sentiment too negative - no new entries"
    elif rsi_signal:
        action = "HOLD/BUY"
        reason = f"RSI(2)={last_rsi:.1f} < {rsi_entry} - in mean-reversion trade"
    elif trend_base_pct > 0.0 and above_sma:
        action = "TREND"
        reason = f"RSI(2)={last_rsi:.1f} neutral - holding trend baseline {trend_base_pct:.0%}"
    else:
        action = "FLAT"
        reason = f"RSI(2)={last_rsi:.1f} neutral, no baseline"

    return {
        "close"       : last_close,
        "rsi"         : last_rsi,
        "sma"         : last_sma,
        "above_sma"   : above_sma,
        "vol_weight"  : vol_w,
        "news_modifier": news_modifier,
        "raw_signal"  : rsi_signal,
        "final_weight": sized,
        "action"      : action,
        "reason"      : reason,
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RSI(2) live signal")
    parser.add_argument("--date",         default=TODAY,         help="Signal date (default: today)")
    parser.add_argument("--json",         default=RESULTS_JSON,  help="Path to rsi2_results.json")
    parser.add_argument("--no-news",      action="store_true",   help="Skip FinBERT news scoring")
    parser.add_argument("--force-recal",  action="store_true",   help="Force recalibration now regardless of schedule")
    parser.add_argument("--skip-recal",   action="store_true",   help="Never recalibrate this run")
    args = parser.parse_args()

    print("=" * 68)
    print(f"  RSI(2) LIVE SIGNAL  -  {args.date}")
    print("=" * 68)

    # ── auto-recalibration check ──────────────────────────────────────────────
    if not args.skip_recal:
        print()
        maybe_recalibrate(force=args.force_recal)
        print()

    best_params = load_best_params(args.json)
    best_bt, best_sm = _split_params(best_params)

    rsi_entry       = best_bt["rsi_entry"]
    rsi_exit        = best_bt["rsi_exit"]
    sma_period      = best_bt["sma_period"]
    target_vol      = best_bt["target_vol"]
    vol_lookback    = best_bt["vol_lookback"]
    max_weight      = best_bt["max_weight"]
    trend_base_pct  = best_bt.get("trend_base_pct", 0.0)
    vix_block       = best_bt.get("vix_block", None)
    hard_stop_pct   = best_bt.get("hard_stop_pct", None)

    print(f"\n  Params: entry={rsi_entry} exit={rsi_exit} "
          f"sma={sma_period} tvol={target_vol} trend={trend_base_pct} "
          f"vix_block={vix_block} stop={hard_stop_pct}")

    # Download enough history for warmup (SMA needs sma_period bars)
    warmup_start = (datetime.strptime(args.date, "%Y-%m-%d")
                    - relativedelta(months=14)).strftime("%Y-%m-%d")
    print(f"\n  Downloading prices from {warmup_start} ...", end=" ", flush=True)
    prices = fetch_prices(TICKERS, warmup_start, args.date)
    if not prices:
        print("ERROR: no price data.")
        return
    print(f"OK ({len(prices)} tickers)")

    # T-bill cash rate and VIX
    _strat._CASH_RATE = fetch_cash_rate(warmup_start, args.date)
    _strat._VIX = fetch_vix(warmup_start, args.date)

    # ── News sentiment ────────────────────────────────────────────────────────
    news_modifiers: dict = {}
    if not args.no_news:
        print(f"\n  Scoring news sentiment via FinBERT ...")
        nj = NewsJudgment(use_finbert=True)
        nj.report(TICKERS)
        for t in TICKERS:
            mod, lbl, dtl = nj.get_modifier(t)
            news_modifiers[t] = mod
    else:
        print("\n  News scoring disabled (--no-news).")
        for t in TICKERS:
            news_modifiers[t] = 1.0

    # ── Per-ticker signals ────────────────────────────────────────────────────
    print("=" * 68)
    print("  SIGNAL TABLE")
    print("=" * 68)
    print(f"\n  {'Ticker':<7} {'Close':>8} {'RSI(2)':>8} {'SMA':>8} "
          f"{'VolW':>6} {'NewsM':>6} {'Weight':>7}  Action")
    print("  " + "-" * 68)

    signals = {}
    for ticker, close in prices.items():
        nm   = news_modifiers.get(ticker, 1.0)
        sig  = compute_live_signal(
            close, rsi_entry, rsi_exit, sma_period,
            target_vol, vol_lookback, max_weight, trend_base_pct, nm,
            vix_block=vix_block, hard_stop_pct=hard_stop_pct,
        )
        signals[ticker] = sig
        trend_flag = "^" if sig["above_sma"] else "v"
        print(f"  {ticker:<7} {sig['close']:>8.2f} {sig['rsi']:>8.1f} "
              f"{sig['sma']:>8.2f} {sig['vol_weight']:>6.2f} "
              f"{sig['news_modifier']:>6.2f} {sig['final_weight']:>7.3f}  "
              f"{sig['action']:<10} {trend_flag}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  {'Ticker':<7}  Reason")
    print("  " + "-" * 60)
    for ticker, sig in signals.items():
        print(f"  {ticker:<7}  {sig['reason']}")

    # Overall portfolio allocation
    total_w = sum(s["final_weight"] for s in signals.values())
    print(f"\n  Total portfolio weight : {total_w:.3f}  "
          f"({'leveraged' if total_w > 1.0 else 'cash remaining: '}"
          f"{'' if total_w > 1.0 else f'{(1-total_w)*100:.1f}%'})")

    # News panic check
    panic_tickers = [t for t in TICKERS
                     if not args.no_news and nj.get_score(t) < -0.75]
    if panic_tickers:
        print(f"\n  !! PANIC signal on: {', '.join(panic_tickers)}")
        print("     Consider exiting existing positions on these tickers.")

    print()


if __name__ == "__main__":
    main()
