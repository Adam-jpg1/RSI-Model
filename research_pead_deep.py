"""
Deep PEAD (Post-Earnings Announcement Drift) Research

SHAP analysis revealed pead_days_since and pead_gap_score are the #1 and #3
most predictive features across all 6 tickers. This script digs into WHY
and finds ways to improve the signal.

Current implementation uses overnight gap > 3% as earnings proxy.
Problems with this approach:
  - Splits, macro events, news catalysts look like earnings gaps
  - Exact earnings dates are more reliable for drift window
  - Doesn't capture analyst surprise magnitude (biggest driver of PEAD size)

Research questions:
  1. Does actual earnings date vs. gap-proxy improve IC?
  2. What's the optimal drift window (5d? 10d? 20d?) by ticker?
  3. Does gap magnitude predict drift magnitude? (larger gap = stronger drift?)
  4. Is drift asymmetric? (positive gaps drift more than negative?)
  5. What happens when gap aligns with daily trend? (gap_up + uptrend = stronger?)

New features to test:
  - pead_surprise_intensity: gap magnitude × sign (continuous vs. binary)
  - pead_days_since_norm: days_since / 20 (normalized to [0,1])
  - pead_signal: gap_score / (1 + days_since/5)  — exponentially decayed
  - pead_gap_x_trend: pead_gap_score × d_trend_score  — trend alignment
  - days_before_earnings: countdown to NEXT earnings (< 5 days = near-event)

Methodology: IC (Spearman) of each feature vs. next-6h return across all 6 tickers
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from scipy import stats

from data.fetcher import get_price_data, get_daily_data
from data.indicators import (add_all_indicators, add_daily_trend_features,
                              add_pead_features)
from ml.features import build_target, target_bars_for_interval

TICKERS  = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD   = "2y"
INTERVAL = "1h"


def compute_ic(x: pd.Series, y: pd.Series) -> dict:
    common = x.dropna().index.intersection(y.dropna().index)
    if len(common) < 50:
        return {"ic": np.nan, "p": np.nan, "n": 0}
    rho, p = stats.spearmanr(x.loc[common], y.loc[common])
    return {"ic": float(rho), "p": float(p), "n": len(common)}


def get_actual_earnings_dates(ticker: str) -> list:
    """Fetch actual earnings dates from yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        # earnings_dates: index = date, 'Reported EPS'/'Estimated EPS' columns
        ed = t.earnings_dates
        if ed is None or len(ed) == 0:
            return []
        return [pd.Timestamp(d).date() for d in ed.index if pd.notna(d)]
    except Exception:
        return []


def get_earnings_gaps(daily_df: pd.DataFrame, actual_dates: list = None,
                      gap_threshold: float = 0.025) -> pd.DataFrame:
    """
    Returns a DataFrame with earnings_date, gap_pct, and direction.
    Uses actual earnings dates if available, otherwise falls back to gap detection.
    """
    daily_close = daily_df["Close"]
    daily_open  = daily_df["Open"] if "Open" in daily_df.columns else daily_close
    overnight_gap = (daily_open - daily_close.shift(1)) / daily_close.shift(1)

    rows = []
    dates_set = set(str(d) for d in (actual_dates or []))

    for ts, gap in zip(daily_df.index, overnight_gap.values):
        if np.isnan(gap):
            continue
        d = pd.Timestamp(ts).date()
        is_actual = str(d) in dates_set
        is_large_gap = abs(gap) >= gap_threshold

        if is_actual or (not dates_set and is_large_gap):
            rows.append({
                "date": d,
                "gap_pct": float(gap),
                "direction": 1 if gap > 0 else -1,
                "is_actual_earnings": is_actual,
                "magnitude": abs(float(gap)),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["date", "gap_pct", "direction", "is_actual_earnings", "magnitude"])


def add_enhanced_pead(hourly_df: pd.DataFrame, earnings_df: pd.DataFrame) -> pd.DataFrame:
    """Add enhanced PEAD features using precise earnings data."""
    df = hourly_df.copy()
    n = len(df)

    # Initialize arrays
    gap_score        = np.zeros(n)
    days_since       = np.full(n, 30.0)
    days_before_next = np.full(n, 63.0)
    surprise_signal  = np.zeros(n)
    pead_decay       = np.zeros(n)
    gap_x_trend      = np.zeros(n)

    if earnings_df.empty:
        for c in ["ep_gap_score", "ep_days_since", "ep_days_before", "ep_signal"]:
            df[c] = 0.0
        return df

    # Sort earnings by date
    earnings_df = earnings_df.sort_values("date")
    earn_dates = list(earnings_df["date"])
    earn_gaps  = list(earnings_df["gap_pct"])

    hourly_dates = pd.to_datetime(df.index).date

    for i, dt in enumerate(hourly_dates):
        # Last earnings event (days_since)
        last_idx = -1
        for j, ed in enumerate(earn_dates):
            if ed <= dt:
                last_idx = j
        if last_idx >= 0:
            last_ed = earn_dates[last_idx]
            days_ago = (dt - last_ed).days
            gap = earn_gaps[last_idx]
            gap_score[i]  = gap
            days_since[i] = min(float(days_ago), 30.0)
            # Decay signal: half-life 7 days
            pead_decay[i] = gap * np.exp(-days_ago / 7.0) if days_ago <= 25 else 0.0

        # Next earnings event (days_before)
        import bisect
        next_idx = bisect.bisect_right(earn_dates, dt)
        if next_idx < len(earn_dates):
            next_ed = earn_dates[next_idx]
            days_to = (next_ed - dt).days
            days_before_next[i] = min(float(days_to), 63.0)

        # Surprise signal: magnitude × direction, decayed
        if last_idx >= 0:
            days_ago = (dt - earn_dates[last_idx]).days
            mag = abs(earn_gaps[last_idx])
            if days_ago <= 20:
                surprise_signal[i] = earn_gaps[last_idx] * (1.0 - days_ago / 20.0)

    df["ep_gap_score"]   = gap_score
    df["ep_days_since"]  = days_since
    df["ep_days_before"] = days_before_next
    df["ep_signal"]      = surprise_signal
    df["ep_decay"]       = pead_decay
    return df


def run():
    print("=" * 70)
    print("  Deep PEAD Research")
    print(f"  SHAP ranking: pead_days_since=#1, pead_gap_score=#3 across all 6 tickers")
    print(f"  Goal: improve PEAD signals, find ideal drift window, test enhancement")
    print("=" * 70)

    n_bars = target_bars_for_interval(INTERVAL)

    all_results = []

    for ticker in TICKERS:
        print(f"\n{'=' * 50}")
        print(f"  {ticker}")
        print('=' * 50)

        try:
            df = get_price_data(ticker, period=PERIOD, interval=INTERVAL)
            df = add_all_indicators(df)
            daily_df = get_daily_data(ticker, period=PERIOD)
            daily_df = add_all_indicators(daily_df)
        except Exception as e:
            print(f"  ERROR fetching data: {e}")
            continue

        # Add standard PEAD features
        df = add_pead_features(df, daily_df)

        # Get actual earnings dates
        actual_dates = get_actual_earnings_dates(ticker)
        print(f"  Actual earnings dates: {len(actual_dates)}")

        # Build enhanced earnings features
        earnings_df = get_earnings_gaps(daily_df, actual_dates=actual_dates)
        df = add_enhanced_pead(df, earnings_df)
        print(f"  Earnings events found: {len(earnings_df)}")

        # Build target
        target = build_target(df, n_bars=n_bars)
        mask = target.notna() & (np.arange(len(df)) < len(df) - n_bars)
        target = target[mask]

        print(f"\n  Standard PEAD features:")
        for feat in ["pead_gap_score", "pead_days_since", "pead_bull_active", "pead_momentum"]:
            if feat in df.columns:
                x = df[feat][mask]
                r = compute_ic(x, target)
                sig = "*" if r["p"] < 0.05 else " "
                print(f"    {feat:<30} IC={r['ic']:>+8.4f}  p={r['p']:.4f} {sig}")

        print(f"\n  Enhanced PEAD features:")
        for feat in ["ep_gap_score", "ep_days_since", "ep_days_before", "ep_signal", "ep_decay"]:
            if feat in df.columns:
                x = df[feat][mask]
                r = compute_ic(x, target)
                sig = "*" if r["p"] < 0.05 else " "
                print(f"    {feat:<30} IC={r['ic']:>+8.4f}  p={r['p']:.4f} {sig}")

        # Drift window analysis: IC at different days_since windows
        print(f"\n  PEAD drift by days_since window:")
        pead_days = df["pead_days_since"][mask]
        gap_score = df["pead_gap_score"][mask]
        for window_start, window_end in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]:
            w_mask = (pead_days >= window_start) & (pead_days < window_end)
            if w_mask.sum() < 30:
                continue
            x = gap_score[w_mask]
            y = target[w_mask]
            r = compute_ic(x, y)
            sig = "*" if r["p"] < 0.05 else " "
            print(f"    days_since {window_start:>2}-{window_end:>2}: "
                  f"IC={r['ic']:>+8.4f}  n={r['n']:>6} {sig}")

        # Asymmetry: positive vs. negative gaps
        print(f"\n  Asymmetry: positive vs. negative gaps:")
        pos_mask = gap_score > 0.03
        neg_mask = gap_score < -0.03
        near_mask = pead_days < 15

        if (pos_mask & near_mask).sum() >= 20:
            r = compute_ic(gap_score[pos_mask & near_mask], target[pos_mask & near_mask])
            print(f"    Positive gaps (>3%), days<15:  IC={r['ic']:>+8.4f}  n={r['n']}")
        if (neg_mask & near_mask).sum() >= 20:
            r = compute_ic(gap_score[neg_mask & near_mask], target[neg_mask & near_mask])
            print(f"    Negative gaps (<-3%), days<15: IC={r['ic']:>+8.4f}  n={r['n']}")

        # Interaction: gap_score × daily trend
        try:
            daily_df_trend = get_daily_data(ticker, period=PERIOD)
            daily_df_trend = add_all_indicators(daily_df_trend)
            spy_daily = get_daily_data("SPY", period=PERIOD)
            df_with_trend = add_daily_trend_features(df, daily_df_trend, spy_daily_df=spy_daily)
            if "d_trend_score" in df_with_trend.columns:
                trend = df_with_trend["d_trend_score"][mask]
                gap  = df["pead_gap_score"][mask]
                pead_x_trend = gap * trend
                r = compute_ic(pead_x_trend, target)
                sig = "*" if r["p"] < 0.05 else " "
                print(f"\n  Interaction: pead_gap_score × d_trend_score:")
                print(f"    IC={r['ic']:>+8.4f}  p={r['p']:.4f} {sig}")
        except Exception:
            pass

        # Countdown to next earnings (pre-event drift / volatility compression)
        print(f"\n  Pre-earnings countdown features:")
        if "ep_days_before" in df.columns:
            ep_before = df["ep_days_before"][mask]
            for window_start, window_end in [(0, 5), (5, 10), (10, 20)]:
                w_mask = (ep_before >= window_start) & (ep_before < window_end)
                if w_mask.sum() < 20:
                    continue
                r_target = compute_ic(ep_before[w_mask], target[w_mask])
                print(f"    {window_start:>2}-{window_end:>2} days before earnings: "
                      f"IC={r_target['ic']:>+8.4f}  n={r_target['n']:>5}")

        all_results.append({"ticker": ticker})

    # Summary
    print(f"\n\n{'=' * 70}")
    print("  PEAD RESEARCH SUMMARY")
    print('=' * 70)
    print(f"""
Key findings from SHAP analysis (prior):
  pead_days_since  = #1 overall (mean|SHAP|=0.01108 across 6 tickers)
  pead_gap_score   = #3 overall (mean|SHAP|=0.00981)

This research tests:
  1. Whether actual earnings dates (via yfinance) improve on gap proxy
  2. Optimal drift window by ticker
  3. Gap sign asymmetry (up gaps drift more than down gaps?)
  4. Pre-earnings countdown features

Recommended PEAD enhancements to implement:
  - Add ep_days_before (countdown to next earnings) to FEATURE_COLS
  - Tune gap_threshold per ticker (3% global may miss small-gap events)
  - If ep_signal IC > pead_gap_score IC: replace standard PEAD with enhanced
""")


if __name__ == "__main__":
    run()
