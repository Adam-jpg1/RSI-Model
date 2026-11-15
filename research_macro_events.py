"""
Macro Calendar Event IC Analysis
Test whether FOMC and NFP event proximity have predictive power.

Hypothesis: stocks behave differently on Fed decision days (high vol, mean-reversion after)
and NFP days (risk-on/risk-off jump, then reversion).

If we find IC > 0.03, we add macro event flags as features.
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import date, timedelta
from scipy.stats import spearmanr

from data.fetcher import get_price_data
from data.indicators import add_all_indicators
from ml.features import target_bars_for_interval

TICKERS = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD = "2y"
FORWARD_BARS = target_bars_for_interval("1h")

# FOMC decision dates (2024-2026): ~8 per year, Wednesdays at 2 PM ET
# Source: Federal Reserve calendar
FOMC_DATES = [
    # 2025
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
    date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
    date(2025, 11, 7), date(2025, 12, 17),
    # 2026
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 11, 4),
    # 2024 (for older part of 2y window)
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),
    date(2024, 6, 12), date(2024, 7, 31), date(2024, 9, 18),
    date(2024, 11, 7), date(2024, 12, 18),
]

# NFP (Non-Farm Payrolls) release dates: first Friday of each month, 8:30 AM ET
# This creates pre-market moves and intraday volatility
NFP_DATES = [
    # 2024
    date(2024, 1, 5), date(2024, 2, 2), date(2024, 3, 8),
    date(2024, 4, 5), date(2024, 5, 3), date(2024, 6, 7),
    date(2024, 7, 5), date(2024, 8, 2), date(2024, 9, 6),
    date(2024, 10, 4), date(2024, 11, 1), date(2024, 12, 6),
    # 2025
    date(2025, 1, 10), date(2025, 2, 7), date(2025, 3, 7),
    date(2025, 4, 4), date(2025, 5, 2), date(2025, 6, 6),
    date(2025, 7, 3), date(2025, 8, 1), date(2025, 9, 5),
    date(2025, 10, 3), date(2025, 11, 7), date(2025, 12, 5),
    # 2026
    date(2026, 1, 9), date(2026, 2, 6), date(2026, 3, 6),
    date(2026, 4, 3), date(2026, 5, 1), date(2026, 6, 5),
    date(2026, 7, 10), date(2026, 8, 7), date(2026, 9, 4),
    date(2026, 10, 2), date(2026, 11, 6),
]

FOMC_SET = set(FOMC_DATES)
NFP_SET = set(NFP_DATES)


def build_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add macro calendar features to df."""
    df = df.copy()
    dates = pd.to_datetime(df.index).date

    # FOMC flags
    df["is_fomc_day"] = [1 if d in FOMC_SET else 0 for d in dates]
    df["is_fomc_eve"] = [1 if (d + timedelta(days=1)) in FOMC_SET else 0 for d in dates]
    df["is_fomc_next"] = [1 if (d - timedelta(days=1)) in FOMC_SET else 0 for d in dates]

    # NFP flags
    df["is_nfp_day"] = [1 if d in NFP_SET else 0 for d in dates]
    df["is_nfp_next"] = [1 if (d - timedelta(days=1)) in NFP_SET else 0 for d in dates]

    # Combined: "macro event week" (within 1 day of FOMC or NFP)
    df["macro_event_week"] = (
        (df["is_fomc_day"] + df["is_fomc_eve"] + df["is_fomc_next"] +
         df["is_nfp_day"] + df["is_nfp_next"]).clip(0, 1)
    )

    # FOMC day hour interaction: 2PM ET = 18:00 UTC (EDT) or 19:00 UTC (EST) — decision hour
    if hasattr(df.index, 'hour'):
        is_fomc = df["is_fomc_day"].values
        hour_utc = pd.to_datetime(df.index).hour
        df["fomc_post_decision"] = ((is_fomc == 1) & (hour_utc >= 18)).astype(int)
    else:
        df["fomc_post_decision"] = 0

    return df


def compute_event_ic(ticker: str) -> pd.DataFrame:
    df = get_price_data(ticker, period=PERIOD, interval="1h")
    df = add_all_indicators(df)
    df = build_macro_features(df)

    future_ret = df["Close"].shift(-FORWARD_BARS) / df["Close"] - 1.0
    macro_cols = ["is_fomc_day", "is_fomc_eve", "is_fomc_next",
                  "is_nfp_day", "is_nfp_next", "macro_event_week",
                  "fomc_post_decision"]

    rows = []
    for col in macro_cols:
        if col not in df.columns:
            continue
        common = future_ret.dropna().index.intersection(df[col].dropna().index)
        if len(common) < 50:
            continue
        ic, pval = spearmanr(df.loc[common, col], future_ret.loc[common], nan_policy="omit")
        if np.isnan(ic):
            continue

        # Also compute mean return on event days vs non-event days
        event_mask = df.loc[common, col] == 1
        if event_mask.sum() > 10:
            event_ret = future_ret.loc[common][event_mask].mean()
            non_event_ret = future_ret.loc[common][~event_mask].mean()
            lift = event_ret - non_event_ret
        else:
            event_ret = non_event_ret = lift = np.nan

        rows.append({
            "ticker": ticker,
            "feature": col,
            "IC": round(ic, 4),
            "pval": round(pval, 4),
            "significant": pval < 0.05,
            "n_event_bars": int(event_mask.sum()) if event_mask.sum() > 0 else 0,
            "event_return_pct": round(event_ret * 100, 4) if not np.isnan(event_ret) else np.nan,
            "non_event_return_pct": round(non_event_ret * 100, 4) if not np.isnan(non_event_ret) else np.nan,
            "lift_pct": round(lift * 100, 4) if not np.isnan(lift) else np.nan,
        })

    return pd.DataFrame(rows)


def run_all():
    print("=" * 70)
    print("  Macro Calendar Event IC Analysis (FOMC + NFP)")
    print("=" * 70)

    all_results = []
    for ticker in TICKERS:
        print(f"\n[macro-ic] {ticker}...")
        try:
            df = compute_event_ic(ticker)
            all_results.append(df)
        except Exception as e:
            print(f"  ERROR: {e}")

    if not all_results:
        print("No results.")
        return

    combined = pd.concat(all_results, ignore_index=True)

    # Group by feature, show cross-ticker summary
    print("\n" + "=" * 70)
    print("  MACRO EVENT IC BY FEATURE (cross-ticker averages)")
    print("=" * 70)
    print(f"{'Feature':<25} {'Avg_IC':>8} {'% Sig':>7} {'Avg_lift%':>10} {'n_bars':>7}")
    print("-" * 60)
    for feat in combined["feature"].unique():
        sub = combined[combined["feature"] == feat]
        avg_ic = sub["IC"].mean()
        pct_sig = sub["significant"].mean() * 100
        avg_lift = sub["lift_pct"].mean()
        n_bars = sub["n_event_bars"].mean()
        print(f"  {feat:<23} {avg_ic:>+7.4f}  {pct_sig:>6.0f}%  {avg_lift:>+9.4f}  {n_bars:>6.0f}")

    # Show significant results
    sig = combined[combined["significant"]].sort_values("IC", key=abs, ascending=False)
    if not sig.empty:
        print(f"\n{'Significant results (p<0.05)':}")
        print(f"{'Ticker':<8} {'Feature':<25} {'IC':>8} {'pval':>7} {'lift%':>8} {'n_event':>8}")
        print("-" * 65)
        for _, row in sig.iterrows():
            print(f"  {row['ticker']:<6} {row['feature']:<23} {row['IC']:>+7.4f}  "
                  f"{row['pval']:>6.4f}  {row['lift_pct']:>+7.4f}  {row['n_event_bars']:>7}")

    return combined


if __name__ == "__main__":
    run_all()
