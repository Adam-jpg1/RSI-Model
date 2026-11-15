"""
Cross-Asset Expansion Research

SHAP analysis showed cross-asset macro features are among the most predictive:
  ca_tlt_ret5    = #6  (TLT 5-day return — bond market leads equity)
  ca_hyg_tlt_spr = #10 (credit spread — risk-on/off)
  ca_gld_ret5    = #14 (gold 5-day return — safe haven / inflation)
  ca_tlt_vs_ma20 = #19 (TLT trend context)
  ca_hyg_ret5    = #25 (HYG 5-day return)
  ca_uup_ret5    = #37 (dollar 5-day return)

Current features use 5-day returns and 20-day MAs. This research tests:
  1. Additional lookback windows: 1d, 3d, 10d, 20d
  2. Hourly cross-asset returns (same-timeframe as target)
  3. Additional assets: LQD (IG bonds), EMB (EM bonds), XLE (energy), USO (oil)
  4. Cross-asset momentum vs. mean-reversion patterns
  5. TLT/SPY ratio (risk parity signal)

If any new cross-asset features show IC > 0.03, add to indicators.py and FEATURE_COLS.
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from scipy import stats

from data.fetcher import get_price_data, get_daily_data
from data.indicators import add_all_indicators
from ml.features import build_target, target_bars_for_interval

TICKERS  = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD   = "2y"
INTERVAL = "1h"

# Assets to test: current + new
CURRENT_ASSETS = {
    "TLT": "long bonds",
    "HYG": "high-yield bonds",
    "GLD": "gold",
    "UUP": "dollar",
}
NEW_DAILY_ASSETS = {
    "LQD":  "investment-grade bonds",
    "EMB":  "emerging market bonds",
    "XLE":  "energy sector",
    "USO":  "crude oil",
    "^TNX": "10Y treasury yield",
    "VIXY": "VIX short-term",
}
HOURLY_ASSETS = {
    "TLT":  "long bonds hourly",
    "HYG":  "HYG hourly",
    "GLD":  "gold hourly",
    "SPY":  "SPY hourly",
}

LOOKBACK_WINDOWS = [1, 3, 5, 10, 20]


def compute_ic(x: pd.Series, y: pd.Series) -> dict:
    common = x.dropna().index.intersection(y.dropna().index)
    if len(common) < 50:
        return {"ic": np.nan, "p": np.nan, "n": 0}
    rho, p = stats.spearmanr(x.loc[common], y.loc[common])
    return {"ic": float(rho), "p": float(p), "n": len(common)}


def run():
    n_bars = target_bars_for_interval(INTERVAL)

    print("=" * 70)
    print("  Cross-Asset Expansion Research")
    print(f"  Testing {len(CURRENT_ASSETS) + len(NEW_DAILY_ASSETS)} daily + {len(HOURLY_ASSETS)} hourly assets")
    print(f"  Return windows: {LOOKBACK_WINDOWS} days")
    print("=" * 70)

    # Pre-fetch target tickers
    print("\nFetching target tickers...")
    ticker_data = {}
    for sym in TICKERS:
        try:
            df = get_price_data(sym, period=PERIOD, interval=INTERVAL)
            df = add_all_indicators(df)
            tgt = build_target(df, n_bars=n_bars)
            mask = tgt.notna() & (np.arange(len(df)) < len(df) - n_bars)
            ticker_data[sym] = {"df": df, "target": tgt[mask], "mask": mask}
            print(f"  {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym}: ERROR {e}")

    # === SECTION 1: Daily cross-asset features ===
    print("\n\n" + "=" * 70)
    print("  SECTION 1: Daily cross-asset returns at different windows")
    print("=" * 70)

    all_daily_assets = {**CURRENT_ASSETS, **NEW_DAILY_ASSETS}
    results = {}  # (asset, window) -> {ic_per_ticker}

    for sym, label in all_daily_assets.items():
        print(f"\n  {sym} ({label}):")
        try:
            asset_daily = get_daily_data(sym, period=PERIOD)
        except Exception as e:
            print(f"    FETCH ERROR: {e}")
            continue

        for window in LOOKBACK_WINDOWS:
            ret = asset_daily["Close"].pct_change(window)
            # Forward-fill to hourly grid per ticker
            ics = []
            for ticker, td in ticker_data.items():
                tgt = td["target"]
                mask = td["mask"]
                df_tick = td["df"]

                # Align daily to hourly
                ret_aligned = ret.reindex(
                    ret.index.union(df_tick.index)
                ).ffill().reindex(df_tick.index)

                x = ret_aligned[mask]
                r = compute_ic(x, tgt)
                ics.append(r["ic"])

            avg_ic = float(np.nanmean(ics)) if ics else np.nan
            sig = "*" if abs(avg_ic) > 0.03 else (" " if abs(avg_ic) > 0.015 else "")
            print(f"    {sym}_ret{window:>2}d: avg|IC|={avg_ic:>+8.4f} {sig}  "
                  f"[{', '.join(f'{v:+.3f}' if not np.isnan(v) else ' nan' for v in ics)}]")
            results[(sym, window)] = avg_ic

    # === SECTION 2: Hourly cross-asset features ===
    print("\n\n" + "=" * 70)
    print("  SECTION 2: Hourly cross-asset returns (same frequency as target)")
    print("=" * 70)

    for sym, label in HOURLY_ASSETS.items():
        print(f"\n  {sym} hourly ({label}):")
        try:
            asset_hourly = get_price_data(sym, period=PERIOD, interval=INTERVAL)
        except Exception as e:
            print(f"    FETCH ERROR: {e}")
            continue

        for window in [1, 2, 3, 6, 12]:
            ret = asset_hourly["Close"].pct_change(window)
            ics = []
            for ticker, td in ticker_data.items():
                tgt = td["target"]
                mask = td["mask"]
                df_tick = td["df"]
                x = ret.reindex(df_tick.index).ffill()[mask]
                r = compute_ic(x, tgt)
                ics.append(r["ic"])

            avg_ic = float(np.nanmean(ics)) if ics else np.nan
            sig = "*" if abs(avg_ic) > 0.03 else (" " if abs(avg_ic) > 0.015 else "")
            print(f"    {sym}_ret{window:>2}h: avg|IC|={avg_ic:>+8.4f} {sig}  "
                  f"[{', '.join(f'{v:+.3f}' if not np.isnan(v) else ' nan' for v in ics)}]")

    # === SECTION 3: Cross-asset ratios and spreads ===
    print("\n\n" + "=" * 70)
    print("  SECTION 3: Cross-asset ratios and spreads")
    print("=" * 70)

    ratios_to_test = []
    try:
        tlt = get_daily_data("TLT", period=PERIOD)
        spy_d = get_daily_data("SPY", period=PERIOD)
        hyg = get_daily_data("HYG", period=PERIOD)
        lqd = get_daily_data("LQD", period=PERIOD)

        # SPY/TLT ratio (equity vs bonds)
        spy_tlt_ratio = spy_d["Close"] / tlt["Close"]
        ratios_to_test.append(("spy_tlt_ratio", spy_tlt_ratio.pct_change(5)))

        # HYG/LQD ratio (risk appetite within bonds)
        try:
            hyg_lqd = hyg["Close"] / lqd["Close"]
            ratios_to_test.append(("hyg_lqd_ratio", hyg_lqd.pct_change(5)))
        except Exception:
            pass

        # TLT momentum vs. SPY momentum divergence
        tlt_mom = tlt["Close"].pct_change(10)
        spy_mom = spy_d["Close"].pct_change(10)
        tlt_spy_div = tlt_mom - spy_mom
        ratios_to_test.append(("tlt_spy_divg", tlt_spy_div))

    except Exception as e:
        print(f"  Ratio fetch error: {e}")

    for name, series in ratios_to_test:
        ics = []
        for ticker, td in ticker_data.items():
            tgt = td["target"]
            mask = td["mask"]
            df_tick = td["df"]
            x = series.reindex(series.index.union(df_tick.index)).ffill().reindex(df_tick.index)[mask]
            r = compute_ic(x, tgt)
            ics.append(r["ic"])
        avg_ic = float(np.nanmean(ics)) if ics else np.nan
        sig = "*" if abs(avg_ic) > 0.03 else (" " if abs(avg_ic) > 0.015 else "")
        print(f"  {name:<30} avg|IC|={avg_ic:>+8.4f} {sig}")

    # === Summary: Best new features to add ===
    print("\n\n" + "=" * 70)
    print("  SUMMARY: Best new cross-asset features to add")
    print("=" * 70)
    print("\n  Current features (for reference):")
    current_known = [
        ("ca_tlt_ret5",    0.00824),
        ("ca_hyg_tlt_spr", 0.00706),
        ("ca_gld_ret5",    0.00618),
        ("ca_tlt_vs_ma20", 0.00547),
        ("ca_hyg_ret5",    0.00483),
        ("ca_uup_ret5",    0.00363),
    ]
    for name, shap in current_known:
        print(f"    {name:<30} mean|SHAP|={shap:.5f}")

    print("\n  New features with |IC| > 0.03:")
    best_new = [(k, v) for k, v in results.items() if abs(v) > 0.03]
    best_new.sort(key=lambda x: abs(x[1]), reverse=True)
    if best_new:
        for (sym, window), ic in best_new:
            feat_name = f"ca_{sym.lower().replace('^', '').replace('-', '')}_ret{window}d"
            print(f"    {feat_name:<30} avg|IC|={ic:>+8.4f}")
    else:
        print("    None found (threshold: |IC| > 0.03)")

    print("\n  Features with |IC| > 0.015 (worth considering):")
    near = [(k, v) for k, v in results.items() if 0.015 < abs(v) <= 0.03]
    near.sort(key=lambda x: abs(x[1]), reverse=True)
    for (sym, window), ic in near[:10]:
        feat_name = f"ca_{sym.lower().replace('^', '').replace('-', '')}_ret{window}d"
        print(f"    {feat_name:<30} avg|IC|={ic:>+8.4f}")


if __name__ == "__main__":
    run()
