"""
Cross-Asset Lead/Lag IC Analysis
Test whether returns of related assets at time T predict target ticker return at T+1 to T+6.

Hypothesis: Sector ETF (SMH, QQQ) and closely-correlated peers (AMD→MRVL, AMD→APP)
have leading predictive power beyond what single-stock indicators capture.

If we find IC > 0.03 for a lag-1 or lag-2 relationship, we can add these as features
and trigger a full Optuna re-run.
"""
import sys
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data.fetcher import get_price_data
from data.indicators import add_all_indicators
from ml.features import target_bars_for_interval

TARGETS = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD = "2y"
FORWARD_BARS = target_bars_for_interval("1h")  # 6h forward return

# Potential predictors: sector ETFs + peer stocks
PREDICTORS = {
    "SMH":  "Semiconductor ETF",
    "QQQ":  "Nasdaq-100 ETF",
    "XLK":  "Tech sector ETF",
    "NVDA": "Nvidia (semis peer)",
    "AMD":  "AMD (semis peer)",   # AMD leads MRVL/APP?
    "MSFT": "Microsoft (mega-cap tech)",
    "SPY":  "S&P 500",
    "^VIX": "VIX (risk-off gauge)",
}

LAGS = [1, 2, 3, 6]  # bars ago (1h each)
MIN_OBS = 100


def compute_lagged_return(df: pd.DataFrame, lag: int) -> pd.Series:
    """Return at lag bars ago (shifted forward for alignment)."""
    r = df["Close"].pct_change()
    return r.shift(lag)  # shift(lag) makes past return available at current bar


def build_target(df: pd.DataFrame) -> pd.Series:
    return df["Close"].shift(-FORWARD_BARS) / df["Close"] - 1.0


def analyze_target(ticker: str, predictor_data: dict) -> pd.DataFrame:
    print(f"\n[lead-lag] Target: {ticker}")

    df_target = get_price_data(ticker, period=PERIOD, interval="1h")
    if df_target is None or len(df_target) < 200:
        print(f"  Insufficient data for {ticker}")
        return pd.DataFrame()

    future_ret = build_target(df_target)
    # Align to common index
    idx = df_target.index

    rows = []
    for pred_ticker, pred_name in PREDICTORS.items():
        if pred_ticker == ticker:
            continue  # skip self

        pred_df = predictor_data.get(pred_ticker)
        if pred_df is None or len(pred_df) < 200:
            continue

        # Reindex predictor to target's index
        pred_aligned = pred_df["Close"].reindex(idx, method="ffill")

        for lag in LAGS:
            lag_ret = pred_aligned.pct_change().shift(lag)

            # Also test predictor RSI as a leading signal (mean-reversion: overbought sector → target falls)
            # Compute only if indicators already in pred_df
            if "RSI_14" in pred_df.columns:
                pred_rsi = pred_df["RSI_14"].reindex(idx, method="ffill").shift(lag)
            else:
                pred_rsi = None

            # Merge with target
            common = future_ret.dropna().index.intersection(lag_ret.dropna().index)
            if len(common) < MIN_OBS:
                continue

            fr = future_ret.loc[common]
            lr = lag_ret.loc[common]

            # Drop pairs where either is NaN
            valid = fr.notna() & lr.notna()
            fr_v = fr[valid]
            lr_v = lr[valid]

            if len(fr_v) < MIN_OBS:
                continue

            ic, pval = spearmanr(lr_v, fr_v, nan_policy='omit')
            if np.isnan(ic):
                continue

            row = {
                'predictor': pred_ticker,
                'predictor_name': pred_name,
                'lag_bars': lag,
                'lag_hours': lag,
                'feature': 'return',
                'IC': round(ic, 4),
                'abs_IC': round(abs(ic), 4),
                'pval': round(pval, 4),
                'n_obs': int(len(fr_v)),
                'significant': pval < 0.05,
            }
            rows.append(row)

            # Also test predictor RSI leading indicator
            if pred_rsi is not None:
                common2 = future_ret.dropna().index.intersection(pred_rsi.dropna().index)
                if len(common2) >= MIN_OBS:
                    fr2 = future_ret.loc[common2]
                    rsi2 = pred_rsi.loc[common2]
                    valid2 = fr2.notna() & rsi2.notna()
                    if valid2.sum() >= MIN_OBS:
                        ic2, pv2 = spearmanr(rsi2[valid2], fr2[valid2], nan_policy='omit')
                        if not np.isnan(ic2):
                            rows.append({
                                'predictor': pred_ticker,
                                'predictor_name': pred_name,
                                'lag_bars': lag,
                                'lag_hours': lag,
                                'feature': 'RSI_14',
                                'IC': round(ic2, 4),
                                'abs_IC': round(abs(ic2), 4),
                                'pval': round(pv2, 4),
                                'n_obs': int(valid2.sum()),
                                'significant': pv2 < 0.05,
                            })

    return pd.DataFrame(rows)


def run_all():
    print("=" * 70)
    print("  Cross-Asset Lead/Lag IC Analysis")
    print("  Testing whether sector ETFs / peer returns predict next 6h move")
    print("=" * 70)

    # Pre-fetch all predictor data once
    print("\nFetching predictor data...")
    predictor_data = {}
    for sym in PREDICTORS:
        try:
            df = get_price_data(sym, period=PERIOD, interval="1h")
            if df is not None and len(df) > 200:
                df = add_all_indicators(df)
                predictor_data[sym] = df
                print(f"  {sym:6s}: {len(df)} bars")
            else:
                print(f"  {sym:6s}: SKIP (insufficient data)")
                predictor_data[sym] = None
        except Exception as e:
            print(f"  {sym:6s}: ERROR — {e}")
            predictor_data[sym] = None

    all_results = {}
    for target in TARGETS:
        result_df = analyze_target(target, predictor_data)
        all_results[target] = result_df

    # Print summary: for each target, show top predictors by abs_IC
    print("\n" + "=" * 70)
    print("  RESULTS: Top cross-asset leading predictors (by abs IC)")
    print("=" * 70)

    IC_THRESHOLD = 0.03

    global_top = []
    for ticker, df in all_results.items():
        if df.empty:
            continue
        sig = df[df['significant'] & (df['abs_IC'] >= IC_THRESHOLD)].sort_values('abs_IC', ascending=False)

        print(f"\n{ticker} — significant predictors (p<0.05, |IC|>={IC_THRESHOLD}):")
        if sig.empty:
            print("  (none)")
        else:
            for _, row in sig.head(10).iterrows():
                print(f"  {row['predictor']:6s} lag={row['lag_bars']}h  "
                      f"{row['feature']:10s}  IC={row['IC']:+.4f}  "
                      f"abs={row['abs_IC']:.4f}  p={row['pval']:.4f}  n={row['n_obs']}")
            for _, row in sig.iterrows():
                global_top.append({'target': ticker, **row.to_dict()})

    if global_top:
        top_df = pd.DataFrame(global_top).sort_values('abs_IC', ascending=False)
        print("\n" + "=" * 70)
        print("  GLOBAL TOP CROSS-ASSET SIGNALS (potential new features):")
        print("=" * 70)
        for _, row in top_df.head(20).iterrows():
            print(f"  {row['target']:6s} <- {row['predictor']:6s} lag={row['lag_bars']}h  "
                  f"{row['feature']:10s}  IC={row['IC']:+.4f}  "
                  f"abs={row['abs_IC']:.4f}  p={row['pval']:.4f}")

    return all_results


if __name__ == "__main__":
    run_all()
