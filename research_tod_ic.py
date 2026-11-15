"""
Time-of-Day IC Analysis
Segment hourly data by hour and compute IC per bucket.
Strong IC in specific hours = intraday timing alpha.
"""
import sys
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data.fetcher import get_price_data, get_daily_data
from data.indicators import (add_all_indicators, add_daily_trend_features,
                              add_cross_asset_features, add_pead_features)
from ml.features import get_feature_matrix, target_bars_for_interval
from ml.model import RSIModel
from config import CONFIG

TICKERS = ["APP", "AMD", "SNOW", "TSLA"]
PERIOD = "2y"
FORWARD_BARS = target_bars_for_interval("1h")  # 6h forward return as target


def build_feature_df(ticker):
    df = get_price_data(ticker, period=PERIOD, interval="1h")
    df = add_all_indicators(df)
    try:
        daily_df = get_daily_data(ticker, period=PERIOD)
        spy_df = get_daily_data("SPY", period=PERIOD)
        vix_df = get_daily_data("^VIX", period=PERIOD)
        df = add_daily_trend_features(df, daily_df, compute_weekly=True,
                                      spy_daily_df=spy_df, vix_daily_df=vix_df)
        ca = {}
        for sym in ["TLT", "HYG", "GLD", "UUP"]:
            try:
                ca[sym] = get_daily_data(sym, period=PERIOD)
            except Exception:
                ca[sym] = None
        df = add_cross_asset_features(df, tlt_daily_df=ca.get("TLT"),
                                       hyg_daily_df=ca.get("HYG"),
                                       gld_daily_df=ca.get("GLD"),
                                       uup_daily_df=ca.get("UUP"))
        df = add_pead_features(df, daily_df)
    except Exception as e:
        print(f"  Warning: {e}")
    return df


def compute_tod_ic(ticker):
    print(f"\n[ToD-IC] {ticker}...")
    df = build_feature_df(ticker)

    # Future return target
    future_ret = df["Close"].shift(-FORWARD_BARS) / df["Close"] - 1.0

    # Convert index to US/Eastern to get consistent ET hours
    idx = df.index
    if hasattr(idx, 'tz') and idx.tz is not None:
        hours = idx.tz_convert('America/New_York').hour
    else:
        # Naive index — assume already ET
        hours = idx.hour

    # US market hours in ET: first 1h bar is at 9:30 AM → hour=9; last at 15:30 → hour=15
    market_hours = [9, 10, 11, 12, 13, 14, 15]

    # Build feature matrix (top features by overall IC)
    # Use RSI, MACD, ADX, volume features as summary
    key_features = [c for c in df.columns if any(
        kw in c.lower() for kw in ['rsi', 'macd', 'adx', 'bb_', 'vol_', 'atr', 'tod_']
    )]
    key_features = [c for c in key_features if c in df.columns][:15]

    results = []
    for h in market_hours:
        mask = (hours == h)
        sub = df.loc[mask].copy()
        fr_sub = future_ret.loc[mask].dropna()
        sub = sub.loc[fr_sub.index]

        if len(sub) < 50:
            continue

        ics = []
        for col in key_features:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            common = vals.index.intersection(fr_sub.index)
            if len(common) < 30:
                continue
            ic, pval = spearmanr(vals.loc[common], fr_sub.loc[common], nan_policy='omit')
            if not np.isnan(ic):
                ics.append(abs(ic))

        avg_abs_ic = np.mean(ics) if ics else 0.0
        n = len(fr_sub)
        results.append({'hour_ET': h, 'n_bars': n, 'avg_abs_IC': round(avg_abs_ic, 4)})

    return pd.DataFrame(results).set_index('hour_ET')


def run_all():
    print("=" * 60)
    print("  Time-of-Day IC Analysis")
    print("=" * 60)
    combined = {}
    for ticker in TICKERS:
        df = compute_tod_ic(ticker)
        combined[ticker] = df['avg_abs_IC']

    result = pd.DataFrame(combined)
    result['avg'] = result.mean(axis=1)
    result = result.sort_values('avg', ascending=False)

    print(f"\n{'Hour(ET)':>8} | {'APP':>6} | {'AMD':>6} | {'SNOW':>6} | {'TSLA':>6} | {'AVG':>6}")
    print("-" * 50)
    for h, row in result.iterrows():
        print(f"  {h:02d}:00  | "
              f"{row.get('APP', 0):>5.4f} | "
              f"{row.get('AMD', 0):>5.4f} | "
              f"{row.get('SNOW', 0):>5.4f} | "
              f"{row.get('TSLA', 0):>5.4f} | "
              f"{row['avg']:>5.4f}")

    best = result['avg'].idxmax()
    print(f"\n  Best hour: {best:02d}:00 ET (avg abs IC = {result['avg'].max():.4f})")
    worst = result['avg'].idxmin()
    print(f"  Worst hour: {worst:02d}:00 ET (avg abs IC = {result['avg'].min():.4f})")
    print(f"\n  Ratio best/worst: {result['avg'].max()/result['avg'].min():.1f}x")

    return result


if __name__ == "__main__":
    run_all()
