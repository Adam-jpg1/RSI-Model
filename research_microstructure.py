"""
Bar Microstructure Feature Research

Tests 9 new features derivable from hourly OHLCV that measure the INTERNAL
structure of each bar — not just where price ended up, but HOW it got there.

Hypothesis: Institutional order flow leaves footprints in bar structure
(wicks, body-to-range ratio, buying pressure estimate) that predict future
direction at the same frequency as good calendar/volatility signals.

Features tested:
  bar_efficiency      = |close-open| / (high-low)         ~ bar directional purity
  buying_pressure     = (close-low) / (high-low)          ~ Kyle lambda proxy
  wick_imbalance      = lower_wick - upper_wick            ~ net support vs resistance
  upper_wick_atr      = (high - max(o,c)) / ATR           ~ rejection magnitude in ATR units
  lower_wick_atr      = (min(o,c) - low) / ATR            ~ support magnitude in ATR units
  trend_quality_10    = R² of 10-bar linear fit on close  ~ trend smoothness
  momentum_efficiency = |net_10h_move| / sum(|1h_moves|)  ~ ratio net move / total path
  dollar_range_z      = z-score of volume*(high-low)       ~ abnormal combined activity
  tick_imbal_est      = (buying_pressure - 0.5) * volume   ~ signed estimated buy volume
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

TICKERS = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD  = "2y"
INTERVAL = "1h"
N_BARS_FWD = target_bars_for_interval(INTERVAL)  # e.g. 6


def build_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add bar microstructure features to df."""
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    open_ = df["Open"]

    rng = (high - low).replace(0, np.nan)

    # ---- Bar structure ----
    df["bar_efficiency"]  = (close - open_).abs() / rng
    df["buying_pressure"] = (close - low) / rng
    df["upper_wick"]      = (high - np.maximum(open_, close)) / rng
    df["lower_wick"]      = (np.minimum(open_, close) - low) / rng
    df["wick_imbalance"]  = df["lower_wick"] - df["upper_wick"]  # positive = bullish bias

    # Wicks normalised to ATR (shows magnitude, not just %)
    if "ATR" in df.columns:
        atr_safe = df["ATR"].replace(0, np.nan)
    else:
        from data.indicators import compute_atr
        atr_safe = compute_atr(df, 14).replace(0, np.nan)
    df["upper_wick_atr"] = (high - np.maximum(open_, close)) / atr_safe
    df["lower_wick_atr"] = (np.minimum(open_, close) - low)  / atr_safe

    # ---- Trend quality (smoothness of 10-bar trend) ----
    def rolling_r2(prices: pd.Series, window: int) -> pd.Series:
        """R² of linear fit on rolling window."""
        result = np.full(len(prices), np.nan)
        x = np.arange(window, dtype=float)
        arr = prices.values
        for i in range(window - 1, len(arr)):
            y = arr[i - window + 1 : i + 1]
            if np.any(np.isnan(y)):
                continue
            slope, intercept, r, p, se = stats.linregress(x, y)
            result[i] = r ** 2
        return pd.Series(result, index=prices.index)

    df["trend_quality_10"] = rolling_r2(close, 10)

    # ---- Momentum efficiency: how much of the journey is net move? ----
    abs_moves   = close.diff().abs()
    total_path  = abs_moves.rolling(10).sum().replace(0, np.nan)
    net_move    = close.diff(10).abs()
    df["momentum_efficiency_10"] = net_move / total_path  # 0=zigzag, 1=perfectly one-way

    # ---- Signed estimated buy volume (tick imbalance proxy) ----
    if "Volume" in df.columns:
        # dollar_range: volume × bar range, z-scored over 20 bars
        dollar_range = df["Volume"] * rng
        dr_mean = dollar_range.rolling(20).mean()
        dr_std  = dollar_range.rolling(20).std().replace(0, np.nan)
        df["dollar_range_z"] = (dollar_range - dr_mean) / dr_std

        # Estimated net buy volume (Kyle 1985 proxy)
        df["tick_imbal_est"] = (df["buying_pressure"] - 0.5) * df["Volume"]
        # Normalize to rolling std
        ti_std = df["tick_imbal_est"].rolling(20).std().replace(0, np.nan)
        df["tick_imbal_z"] = df["tick_imbal_est"] / ti_std

    return df


def compute_ic(feature: pd.Series, target: pd.Series) -> dict:
    """Spearman IC + p-value between feature and forward return."""
    common = feature.dropna().index.intersection(target.dropna().index)
    if len(common) < 50:
        return {"ic": np.nan, "p": np.nan, "n": 0}
    rho, p = stats.spearmanr(feature.loc[common], target.loc[common])
    return {"ic": rho, "p": p, "n": len(common)}


MICRO_FEATURES = [
    "bar_efficiency",
    "buying_pressure",
    "wick_imbalance",
    "upper_wick_atr",
    "lower_wick_atr",
    "trend_quality_10",
    "momentum_efficiency_10",
    "dollar_range_z",
    "tick_imbal_z",
]

EXISTING_BENCHMARK = [
    "is_fri",        # top consensus (need add_all_indicators + add_calendar_features)
    "ATR_pct",
    "d_trend_score",
    "open_gap_x",
    "pvd_10",
]


def run():
    print("=" * 70)
    print("  Bar Microstructure Feature Research")
    print(f"  Target: {N_BARS_FWD}-bar forward return  |  {PERIOD} {INTERVAL}")
    print("=" * 70)

    results = {}  # {feature: [ic per ticker]}

    for feat in MICRO_FEATURES + EXISTING_BENCHMARK:
        results[feat] = []

    for ticker in TICKERS:
        print(f"\n[ms] {ticker}...", flush=True)
        try:
            df = get_price_data(ticker, period=PERIOD, interval=INTERVAL)

            # Add full indicator set for benchmark features
            from data.indicators import add_all_indicators, add_calendar_features, add_macro_event_features
            df_full = add_all_indicators(df.copy())
            df_full = add_macro_event_features(df_full)
            df_full = add_calendar_features(df_full)
            df_full = build_microstructure_features(df_full)

            target = build_target(df, n_bars=N_BARS_FWD)

            for feat in MICRO_FEATURES + EXISTING_BENCHMARK:
                if feat not in df_full.columns:
                    continue
                ic_res = compute_ic(df_full[feat], target)
                results[feat].append(ic_res["ic"])
                sig = "*" if ic_res["p"] < 0.05 else " "
                print(f"  {feat:<30} IC={ic_res['ic']:+.4f}{sig}  (n={ic_res['n']})")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}")

    # Summary table
    print("\n\n" + "=" * 70)
    print("  SUMMARY: Avg IC across all tickers")
    print(f"  {'Feature':<30} {'Avg_IC':>8} {'StdIC':>7} {'Sig%':>7} {'Category'}")
    print("  " + "-" * 65)

    rows = []
    for feat in MICRO_FEATURES + EXISTING_BENCHMARK:
        ics = [v for v in results[feat] if not np.isnan(v)]
        if not ics:
            continue
        avg_ic  = np.mean(ics)
        std_ic  = np.std(ics)
        sig_pct = sum(abs(v) > 0.03 for v in ics) / len(ics)
        cat = "NEW" if feat in MICRO_FEATURES else "existing"
        rows.append((feat, avg_ic, std_ic, sig_pct, cat))

    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, avg_ic, std_ic, sig_pct, cat in rows:
        print(f"  {feat:<30} {avg_ic:+.4f}  {std_ic:.4f}  {sig_pct:.0%}  {cat}")

    print("\n\nNEW features with |avg_IC| > 0.03 (worth adding to pipeline):")
    strong = [(f, ic) for f, ic, *_ in rows if abs(ic) > 0.03 and f in MICRO_FEATURES]
    if strong:
        for f, ic in strong:
            print(f"  {f:<30} avg_IC={ic:+.4f}")
    else:
        print("  (none exceeded threshold)")

    print("\nAll existing benchmark ICs for reference:")
    for feat, avg_ic, std_ic, sig_pct, cat in rows:
        if cat == "existing":
            print(f"  {feat:<30} {avg_ic:+.4f}")


if __name__ == "__main__":
    run()
