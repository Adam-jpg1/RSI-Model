"""
Research: EPS Surprise as PEAD drift amplifier

Hypothesis: eps_surprise × pead_phase_drift is a stronger signal than
pead_drift_score (gap × drift_phase) because:
  - Gap size = market's immediate reaction (incomplete; may include non-EPS factors)
  - EPS surprise = fundamental beat/miss vs analyst consensus (normalized)
  - Combined: surprise * drift_phase captures BOTH when and how much to expect drift

Validity: EPS surprise is public after earnings announcement → valid for drift window
(days 7-20 post-event). No lookahead.

Expected IC target: >0.15 (comparable to or better than pead_drift_score)
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yfinance as yf

from data.fetcher import get_price_data, get_daily_data, get_earnings_dates
from data.indicators import (add_all_indicators, add_daily_trend_features,
                              add_pead_features, add_earnings_features)
from ml.features import build_target, target_bars_for_interval

TICKERS  = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD   = "2y"
INTERVAL = "1h"
N_BARS   = target_bars_for_interval(INTERVAL)  # 6-bar target


def get_earnings_surprise_map(ticker: str) -> dict:
    """Return {date: surprise_pct} from yfinance earnings_dates."""
    try:
        stock = yf.Ticker(ticker)
        ed = stock.earnings_dates
        if ed is None or ed.empty:
            return {}
        result = {}
        for ts, row in ed.iterrows():
            surp = row.get("Surprise(%)", float("nan"))
            if pd.notna(surp):
                d = pd.Timestamp(str(ts)[:10]).date()
                result[d] = float(surp)
        return result
    except Exception:
        return {}


def add_eps_surprise_features(df: pd.DataFrame, surprise_map: dict,
                               pead_days_col: str = "pead_days_since") -> pd.DataFrame:
    """
    Add eps_surprise and pead_drift_surprise columns.

    eps_surprise: EPS surprise % from the most recent earnings event
                  (capped at ±50 to avoid outlier distortion)
    pead_drift_surprise: eps_surprise * pead_phase_drift
                         (non-zero only in days 7-20 post-earnings)
    """
    if not surprise_map or pead_days_col not in df.columns:
        df["eps_surprise"] = 0.0
        df["pead_drift_surprise"] = 0.0
        return df

    import bisect
    from datetime import date as date_type

    sorted_dates = sorted(surprise_map.keys())
    n = len(df)
    eps_surp_arr = np.zeros(n)

    for i, ts in enumerate(df.index):
        try:
            d = pd.Timestamp(ts).date()
        except Exception:
            continue
        # Most recent earnings at or before this bar
        pos = bisect.bisect_right(sorted_dates, d) - 1
        if pos >= 0:
            eps_surp_arr[i] = np.clip(surprise_map[sorted_dates[pos]], -50.0, 50.0)

    df["eps_surprise"] = eps_surp_arr
    # Multiply by drift phase flag
    if "pead_phase_drift" in df.columns:
        df["pead_drift_surprise"] = df["eps_surprise"] * df["pead_phase_drift"]
    else:
        df["pead_drift_surprise"] = 0.0
    return df


def compute_ic(series: pd.Series, target: pd.Series) -> float:
    """Spearman IC between feature and forward return."""
    mask = series.notna() & target.notna() & (series != 0)
    if mask.sum() < 20:
        return float("nan")
    r, _ = spearmanr(series[mask], target[mask])
    return float(r)


def run():
    print("=" * 70)
    print("  EPS SURPRISE × PEAD DRIFT — IC Research")
    print("=" * 70)

    results = []
    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")
        try:
            df = get_price_data(ticker, period=PERIOD, interval=INTERVAL)
            df = add_all_indicators(df)
            try:
                daily_df = get_daily_data(ticker, period=PERIOD)
                daily_df = add_all_indicators(daily_df)
                df = add_daily_trend_features(df, daily_df)
                df = add_pead_features(df, daily_df)
            except Exception as e:
                print(f"  pead/daily error: {e}")
                continue
            edates = get_earnings_dates(ticker)
            if edates:
                df = add_earnings_features(df, edates)
            surp_map = get_earnings_surprise_map(ticker)
            print(f"  EPS surprise records: {len(surp_map)}")
            df = add_eps_surprise_features(df, surp_map)

            target = build_target(df, n_bars=N_BARS).dropna()
            df = df.loc[target.index]

            # IC for all candidate features
            features_to_test = {
                "pead_days_since":      df.get("pead_days_since",      pd.Series(dtype=float)),
                "pead_drift_score":     df.get("pead_drift_score",     pd.Series(dtype=float)),
                "pead_phase_drift":     df.get("pead_phase_drift",     pd.Series(dtype=float)),
                "eps_surprise":         df.get("eps_surprise",         pd.Series(dtype=float)),
                "pead_drift_surprise":  df.get("pead_drift_surprise",  pd.Series(dtype=float)),
                "days_since_earnings":  df.get("days_since_earnings",  pd.Series(dtype=float)),
            }

            row = {"ticker": ticker}
            for name, feat in features_to_test.items():
                if len(feat) == 0:
                    ic = float("nan")
                else:
                    feat_aligned = feat.reindex(target.index)
                    ic = compute_ic(feat_aligned, target)
                row[name] = ic
                print(f"  {name:30s} IC={ic:+.4f}" if pd.notna(ic) else f"  {name:30s} IC=   NaN")

            # IC within the drift window only (days 7-20)
            if "pead_phase_drift" in df.columns:
                drift_mask = df["pead_phase_drift"] > 0
                n_drift = drift_mask.sum()
                print(f"\n  Within drift window (n={n_drift}):")
                for name, feat in features_to_test.items():
                    if len(feat) == 0:
                        continue
                    feat_aligned = feat.reindex(target.index)
                    t_in_drift = target[drift_mask.reindex(target.index, fill_value=False)]
                    f_in_drift = feat_aligned[drift_mask.reindex(target.index, fill_value=False)]
                    ic_drift = compute_ic(f_in_drift, t_in_drift)
                    print(f"  [drift] {name:28s} IC={ic_drift:+.4f}" if pd.notna(ic_drift) else f"  [drift] {name:28s} IC=   NaN")

            results.append(row)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\n\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    if results:
        df_res = pd.DataFrame(results).set_index("ticker")
        df_res = df_res.round(4)
        print(df_res.to_string())
        print("\nMean IC across tickers:")
        print(df_res.mean().to_string())

    print("\n\nConclusion:")
    print("  If pead_drift_surprise IC >> pead_drift_score IC → implement globally")
    print("  If comparable or lower → pead_drift_score (gap proxy) is sufficient")


if __name__ == "__main__":
    run()
