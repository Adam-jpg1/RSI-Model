"""
Volume-Weighted Price Impact Research

Based on Almgren-Chriss (2001) market impact model: the price impact of a trade
is proportional to volume^0.5. Applied to hourly bars:

  price_impact = bar_return / sqrt(relative_volume)

Interpretation:
  HIGH price_impact (large return, low volume) = price moved without much trading
    -> thin market, potentially informationless price move -> mean-reversion expected
  LOW price_impact (small return, high volume) = lots of volume absorbed without price change
    -> institutional participation, distribution or accumulation in progress

Variants tested:
  vwpi_1h      = abs(close - open) / (sqrt(volume_ratio) × ATR)  normalized price impact
  vwpi_signed  = (close - open) / (sqrt(volume_ratio) × ATR)      signed version
  vwpi_z       = rolling z-score of vwpi over 20 bars
  amihud       = abs(return) / volume (Amihud 2002 illiquidity ratio)
  amihud_z     = z-score of amihud over 20 bars
  kyle_lambda  = price change / volume (classic Kyle 1985 lambda proxy)

Hypothesis: vwpi_signed IC > 0.03 → low-impact moves continue, high-impact moves reverse.
            amihud IC > 0.03 → illiquid markets → bigger reversals.

Also test: interaction with regime (is_choppy × vwpi_z)
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

THRESHOLD = 0.03  # min avg |IC| to recommend adding feature


def build_vwpi_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    open_ = df["Open"]
    vol   = df["Volume"] if "Volume" in df.columns else None

    if "ATR" in df.columns:
        atr = df["ATR"].replace(0, np.nan)
    else:
        atr = (df["High"] - df["Low"]).rolling(14).mean().replace(0, np.nan)

    # Average volume for normalizing
    if vol is not None:
        vol_mean = vol.rolling(20).mean().replace(0, np.nan)
        vol_ratio = vol / vol_mean
        sqrt_vol_ratio = np.sqrt(vol_ratio.replace(0, np.nan))

        bar_ret = (close - open_)

        # Almgren-Chriss style: return / sqrt(relative_volume) / ATR
        df["vwpi_signed"]  = bar_ret / (sqrt_vol_ratio * atr)
        df["vwpi_abs"]     = bar_ret.abs() / (sqrt_vol_ratio * atr)

        # Z-score versions
        def rolling_z(s: pd.Series, w: int = 20) -> pd.Series:
            m = s.rolling(w).mean()
            sd = s.rolling(w).std().replace(0, np.nan)
            return (s - m) / sd

        df["vwpi_z"]       = rolling_z(df["vwpi_signed"])

        # Amihud (2002) illiquidity ratio: |ret| / dollar_volume
        dollar_vol = vol * close
        daily_abs_ret = close.pct_change().abs()
        amihud_raw = daily_abs_ret / dollar_vol.replace(0, np.nan)
        df["amihud_z"]  = rolling_z(amihud_raw)

        # Kyle lambda: price change per unit volume
        price_change = close.diff().abs()
        kyle_raw = price_change / vol.replace(0, np.nan)
        df["kyle_z"]    = rolling_z(kyle_raw)

        # Participation rate: volume relative to 5-day average at this hour
        # Proxy for whether this is an unusual volume hour
        df["vol_surprise"] = vol_ratio - 1.0   # positive = above-avg volume

        # Volume-price confirmation: same direction, strong agreement
        df["vp_confirm"] = np.sign(close - open_) * (vol_ratio - 1.0)

    return df


def compute_ic(x: pd.Series, y: pd.Series) -> dict:
    common = x.dropna().index.intersection(y.dropna().index)
    if len(common) < 50:
        return {"ic": np.nan, "p": np.nan, "n": 0}
    rho, p = stats.spearmanr(x.loc[common], y.loc[common])
    return {"ic": rho, "p": p, "n": len(common)}


VWPI_FEATURES = [
    "vwpi_signed", "vwpi_abs", "vwpi_z",
    "amihud_z", "kyle_z",
    "vol_surprise", "vp_confirm",
]

BENCHMARK = ["ATR_pct", "is_fri", "buying_pressure"]


def run():
    print("=" * 70)
    print("  Volume-Weighted Price Impact Research")
    print(f"  Target: {N_BARS}-bar forward return  |  {PERIOD} {INTERVAL}")
    print("=" * 70)

    results = {f: [] for f in VWPI_FEATURES + BENCHMARK}

    for ticker in TICKERS:
        print(f"\n[pi] {ticker}...", flush=True)
        try:
            df = get_price_data(ticker, period=PERIOD, interval=INTERVAL)
            from data.indicators import add_all_indicators, add_calendar_features, add_microstructure_features
            df = add_all_indicators(df)
            df = add_calendar_features(df)
            df = add_microstructure_features(df)
            df = build_vwpi_features(df)

            target = build_target(df, n_bars=N_BARS)

            for feat in VWPI_FEATURES + BENCHMARK:
                if feat not in df.columns:
                    continue
                ic_res = compute_ic(df[feat], target)
                results[feat].append(ic_res["ic"])
                sig = "*" if ic_res["p"] < 0.05 and abs(ic_res["ic"]) > 0.02 else " "
                if abs(ic_res["ic"]) > 0.015 or not np.isnan(ic_res["ic"]):
                    print(f"  {feat:<25} IC={ic_res['ic']:+.4f}{sig}")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}")

    # Summary
    print("\n\n" + "=" * 70)
    print("  SUMMARY: Avg IC across all tickers")
    print(f"  {'Feature':<28} {'Avg_IC':>8} {'StdIC':>7} {'Sig%':>7} {'Category'}")
    print("  " + "-" * 65)

    rows = []
    for feat in VWPI_FEATURES + BENCHMARK:
        ics = [v for v in results[feat] if not np.isnan(v)]
        if not ics:
            continue
        avg_ic  = np.mean(ics)
        std_ic  = np.std(ics)
        sig_pct = sum(abs(v) > 0.02 for v in ics) / len(ics)
        cat = "NEW" if feat in VWPI_FEATURES else "benchmark"
        rows.append((feat, avg_ic, std_ic, sig_pct, cat))

    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, avg_ic, std_ic, sig_pct, cat in rows:
        print(f"  {feat:<28} {avg_ic:+.4f}  {std_ic:.4f}  {sig_pct:.0%}  {cat}")

    print(f"\nFEATURES with |avg_IC| > {THRESHOLD} (worth adding):")
    strong = [(f, ic) for f, ic, *_ in rows if abs(ic) > THRESHOLD and f in VWPI_FEATURES]
    if strong:
        for f, ic in strong:
            print(f"  {f:<28} avg_IC={ic:+.4f}")
    else:
        print("  (none exceeded threshold)")


if __name__ == "__main__":
    run()
