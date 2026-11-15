"""
Regime-Conditional Signal Research

Hypothesis: One XGBoost model trained on all market conditions learns a
            compromise between momentum and mean-reversion signals, resulting
            in weak flat importance across 138 features.

            Reality: Markets alternate between:
              TRENDING regime  - momentum signals dominate
                (high ADX, smooth R², VIX < 18)
              CHOPPY regime    - mean-reversion signals dominate
                (low ADX, noisy R², VIX > 20)

If true: training regime-specific models could double signal strength.

Experiment:
  1. Label each hourly bar as TRENDING / CHOPPY / NEUTRAL using:
       - 20-bar ADX threshold
       - 20-bar R² of linear regression on close
       - Optionally: SPY/VIX state
  2. Compute IC of key signals within each regime separately
  3. Compare regime-conditional IC vs unconditional IC
  4. If regime-specific IC >> overall IC, recommend regime-conditional models

Key signals to test:
  Momentum signals (expected IC positive in TRENDING):
    rsi_14, macd_hist, d_trend_score, momentum_efficiency_10
  Mean-reversion signals (expected IC negative in CHOPPY):
    buying_pressure, tick_imbal_z, bb_pct (price vs BB extreme)
  Calendar signals (should work in all regimes):
    is_fri, opex_monthly
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from scipy import stats

from data.fetcher import get_price_data
from ml.features import build_target, target_bars_for_interval

TICKERS  = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD   = "2y"
INTERVAL = "1h"
N_BARS   = target_bars_for_interval(INTERVAL)

# Regime detection thresholds
ADX_TREND   = 25.0   # ADX > this = trending
ADX_CHOPPY  = 15.0   # ADX < this = choppy
R2_TREND    = 0.65   # R² of 20-bar linear fit
R2_CHOPPY   = 0.30


def compute_r2_rolling(close: pd.Series, window: int = 20) -> pd.Series:
    """R² of linear fit on rolling window."""
    result = np.full(len(close), np.nan)
    x = np.arange(window, dtype=float)
    arr = close.values
    for i in range(window - 1, len(arr)):
        y = arr[i - window + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        slope, intercept, r, p, se = stats.linregress(x, y)
        result[i] = r ** 2
    return pd.Series(result, index=close.index)


def add_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """Add bar microstructure features."""
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    open_ = df["Open"]
    rng   = (high - low).replace(0, np.nan)

    df = df.copy()
    df["buying_pressure"] = (close - low) / rng
    df["bar_efficiency"]  = (close - open_).abs() / rng
    df["wick_imbalance"]  = (
        (np.minimum(open_, close) - low) - (high - np.maximum(open_, close))
    ) / rng

    if "Volume" in df.columns:
        ti = (df["buying_pressure"] - 0.5) * df["Volume"]
        ti_std = ti.rolling(20).std().replace(0, np.nan)
        df["tick_imbal_z"] = ti / ti_std

    abs_moves  = close.diff().abs()
    total_path = abs_moves.rolling(10).sum().replace(0, np.nan)
    net_move   = close.diff(10).abs()
    df["momentum_efficiency_10"] = net_move / total_path
    return df


def label_regimes(df: pd.DataFrame) -> pd.Series:
    """
    Assign regime label to each bar.
    Returns pd.Series with values: 'TRENDING', 'CHOPPY', 'NEUTRAL'
    Uses 20-bar ADX and R² of linear fit on close.
    """
    if "ADX" not in df.columns:
        from data.indicators import compute_atr
        # Simple approximate ADX using smoothed true range
        return pd.Series("NEUTRAL", index=df.index)

    adx = df["ADX"]
    r2  = compute_r2_rolling(df["Close"], 20)

    regime = pd.Series("NEUTRAL", index=df.index, dtype=object)
    trending = (adx > ADX_TREND) & (r2 > R2_TREND)
    choppy   = (adx < ADX_CHOPPY) & (r2 < R2_CHOPPY)
    regime[trending] = "TRENDING"
    regime[choppy]   = "CHOPPY"
    return regime


def compute_ic(feature: pd.Series, target: pd.Series) -> dict:
    common = feature.dropna().index.intersection(target.dropna().index)
    if len(common) < 30:
        return {"ic": np.nan, "p": np.nan, "n": 0}
    rho, p = stats.spearmanr(feature.loc[common], target.loc[common])
    return {"ic": rho, "p": p, "n": len(common)}


SIGNAL_FEATURES = {
    "--- MOMENTUM signals ---": None,
    "rsi_14":                 "momentum",
    "macd_hist":              "momentum",
    "d_trend_score":          "momentum",
    "momentum_efficiency_10": "momentum",
    "--- MEAN-REVERSION signals ---": None,
    "buying_pressure":        "mean_rev",
    "tick_imbal_z":           "mean_rev",
    "bb_pct":                 "mean_rev",
    "pvd_10":                 "mean_rev",
    "--- CALENDAR signals ---": None,
    "is_fri":                 "calendar",
    "opex_monthly":           "calendar",
    "hour_of_day":            "calendar",
}
# Remove header entries
TEST_FEATURES = {k: v for k, v in SIGNAL_FEATURES.items() if v is not None}


def run():
    print("=" * 70)
    print("  Regime-Conditional Signal Research")
    print(f"  TRENDING (ADX>{ADX_TREND}, R²>{R2_TREND}) | "
          f"CHOPPY (ADX<{ADX_CHOPPY}, R²<{R2_CHOPPY})")
    print("=" * 70)

    # Collect regime-specific ICs per feature
    ic_all     = {f: [] for f in TEST_FEATURES}
    ic_trend   = {f: [] for f in TEST_FEATURES}
    ic_choppy  = {f: [] for f in TEST_FEATURES}
    regime_fracs = []

    for ticker in TICKERS:
        print(f"\n[rc] {ticker}...", flush=True)
        try:
            df = get_price_data(ticker, period=PERIOD, interval=INTERVAL)
            from data.indicators import add_all_indicators, add_calendar_features, add_macro_event_features
            df = add_all_indicators(df)
            df = add_macro_event_features(df)
            df = add_calendar_features(df)
            df = add_microstructure(df)

            target = build_target(df, n_bars=N_BARS)
            regime = label_regimes(df)

            n_trending = (regime == "TRENDING").sum()
            n_choppy   = (regime == "CHOPPY").sum()
            n_neutral  = (regime == "NEUTRAL").sum()
            total = len(regime)
            regime_fracs.append({
                "ticker": ticker,
                "pct_trending": n_trending / total * 100,
                "pct_choppy":   n_choppy   / total * 100,
                "pct_neutral":  n_neutral  / total * 100,
            })
            print(f"  Regime distribution: "
                  f"TRENDING={n_trending/total:.0%}  "
                  f"CHOPPY={n_choppy/total:.0%}  "
                  f"NEUTRAL={n_neutral/total:.0%}")

            for feat, ftype in TEST_FEATURES.items():
                if feat not in df.columns:
                    continue

                # Overall IC
                res_all = compute_ic(df[feat], target)
                ic_all[feat].append(res_all["ic"])

                # Regime-specific
                trend_idx  = df.index[regime == "TRENDING"]
                choppy_idx = df.index[regime == "CHOPPY"]

                res_trend  = compute_ic(df.loc[trend_idx,  feat] if len(trend_idx)  > 0 else pd.Series(dtype=float), target)
                res_choppy = compute_ic(df.loc[choppy_idx, feat] if len(choppy_idx) > 0 else pd.Series(dtype=float), target)

                ic_trend[feat].append(res_trend["ic"])
                ic_choppy[feat].append(res_choppy["ic"])

                sig_all  = "*" if not np.isnan(res_all["ic"])    and abs(res_all["ic"])    > 0.03 else " "
                sig_tr   = "*" if not np.isnan(res_trend["ic"])  and abs(res_trend["ic"])  > 0.03 else " "
                sig_ch   = "*" if not np.isnan(res_choppy["ic"]) and abs(res_choppy["ic"]) > 0.03 else " "
                print(f"  {feat:<30} "
                      f"all={res_all['ic']:+.4f}{sig_all}  "
                      f"trending={res_trend['ic']:+.4f}{sig_tr}  "
                      f"choppy={res_choppy['ic']:+.4f}{sig_ch}")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}")

    # Summary
    print("\n\n" + "=" * 70)
    print("  SUMMARY: Avg IC by regime across all tickers")
    print(f"  {'Feature':<30} {'Type':<10} {'IC_all':>8} {'IC_trend':>10} {'IC_choppy':>10} {'Regime_lift'}")
    print("  " + "-" * 80)

    rows = []
    for feat, ftype in TEST_FEATURES.items():
        all_ics    = [v for v in ic_all[feat]    if not np.isnan(v)]
        trend_ics  = [v for v in ic_trend[feat]  if not np.isnan(v)]
        choppy_ics = [v for v in ic_choppy[feat] if not np.isnan(v)]
        if not all_ics:
            continue
        avg_all    = np.nanmean(all_ics)
        avg_trend  = np.nanmean(trend_ics)  if trend_ics  else np.nan
        avg_choppy = np.nanmean(choppy_ics) if choppy_ics else np.nan
        # Regime lift: does the best regime IC beat overall IC?
        best_regime_ic = max(abs(avg_trend) if not np.isnan(avg_trend) else 0,
                             abs(avg_choppy) if not np.isnan(avg_choppy) else 0)
        lift = best_regime_ic - abs(avg_all)
        rows.append((feat, ftype, avg_all, avg_trend, avg_choppy, lift))

    rows.sort(key=lambda x: abs(x[5]), reverse=True)
    for feat, ftype, avg_all, avg_trend, avg_choppy, lift in rows:
        print(f"  {feat:<30} {ftype:<10} {avg_all:+8.4f} {avg_trend:+10.4f} {avg_choppy:+10.4f}  "
              f"{'+' if lift > 0 else ''}{lift:.4f}")

    print("\n\nKEY FINDING:")
    strong_regime = [(f, t, l) for f, t, ic_a, ic_tr, ic_ch, l in rows if l > 0.02]
    if strong_regime:
        print("  Features with regime-specific IC significantly higher than unconditional:")
        for feat, ftype, lift in strong_regime:
            print(f"    {feat:<30} type={ftype:<10} regime_lift=+{lift:.4f}")
        print("\n  RECOMMENDATION: Train regime-conditional models for these features.")
    else:
        print("  No feature shows strong regime-conditional IC improvement (>0.02 lift).")
        print("  Regime conditioning unlikely to add significant alpha.")

    print("\n\nRegime distribution across portfolio:")
    if regime_fracs:
        avg_trend  = np.mean([r["pct_trending"] for r in regime_fracs])
        avg_choppy = np.mean([r["pct_choppy"]   for r in regime_fracs])
        avg_neutral= np.mean([r["pct_neutral"]  for r in regime_fracs])
        print(f"  Avg TRENDING: {avg_trend:.1f}%  |  CHOPPY: {avg_choppy:.1f}%  |  NEUTRAL: {avg_neutral:.1f}%")


if __name__ == "__main__":
    run()
