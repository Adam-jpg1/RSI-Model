"""
Seasonality Research

SHAP analysis revealed week_of_year is the #8 most predictive feature
(mean|SHAP|=0.00746 across all 6 tickers). This confirms significant
seasonal patterns in tech stock returns.

This research maps:
  1. IC of week_of_year vs. 6h forward return by ticker
  2. Average return by calendar week — which weeks are systematically strong/weak?
  3. IC of month_of_year (already in FEATURE_COLS, let's find the pattern)
  4. Day-of-week IC (already computed: Fri=+0.097 average IC across tickers)
  5. Intraday time-of-day patterns (hour_of_day)
  6. Optimal seasonality interaction features

Application: If certain weeks show consistently strong IC (>0.05), we can:
  - Add week-specific momentum multipliers
  - Create is_strong_week / is_weak_week binary features
  - Weight Optuna trials differently by season

Key known seasonal patterns in tech:
  - January effect: tech stocks often rally in early January
  - Tax-loss selling in December
  - Earnings seasons: Jan/Apr/Jul/Oct (Q4/Q1/Q2/Q3 results)
  - Summer doldrums: low vol in July/August
  - September effect: historically weakest month for stocks
  - Santa Claus rally: last week of December
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from scipy import stats

from data.fetcher import get_price_data
from data.indicators import add_all_indicators
from ml.features import build_target, target_bars_for_interval

TICKERS  = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD   = "2y"
INTERVAL = "1h"


def compute_ic(x: pd.Series, y: pd.Series) -> dict:
    common = x.dropna().index.intersection(y.dropna().index)
    if len(common) < 30:
        return {"ic": np.nan, "p": np.nan, "n": 0}
    rho, p = stats.spearmanr(x.loc[common], y.loc[common])
    return {"ic": float(rho), "p": float(p), "n": len(common)}


def run():
    n_bars = target_bars_for_interval(INTERVAL)

    print("=" * 70)
    print("  Seasonality Research")
    print(f"  Tickers: {TICKERS}")
    print(f"  SHAP: week_of_year=#8, month_of_year=#38, day_of_week=#36")
    print("=" * 70)

    # Pre-fetch all data
    print("\nFetching data...")
    all_data = {}
    for sym in TICKERS:
        try:
            df = get_price_data(sym, period=PERIOD, interval=INTERVAL)
            df = add_all_indicators(df)
            tgt = build_target(df, n_bars=n_bars)
            mask = tgt.notna() & (np.arange(len(df)) < len(df) - n_bars)

            # Build time features
            idx = pd.to_datetime(df.index)
            df["hour_of_day"] = idx.hour
            df["day_of_week"] = idx.dayofweek
            df["month_of_year"] = idx.month
            df["week_of_year"] = idx.isocalendar().week.astype(int).values

            all_data[sym] = {"df": df, "target": tgt[mask], "mask": mask}
            print(f"  {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym}: ERROR {e}")

    # === Week of year IC ===
    print("\n\n" + "=" * 70)
    print("  WEEK OF YEAR vs. 6h forward return IC")
    print("  (Stocks trade 52 weeks/year — which weeks are systematically good?)")
    print("=" * 70)

    week_ics = {}
    for sym, td in all_data.items():
        df = td["df"]
        tgt = td["target"]
        mask = td["mask"]
        week_col = df["week_of_year"][mask]

        # IC per week
        week_ret = {}
        for w in range(1, 53):
            w_mask = week_col == w
            if w_mask.sum() < 10:
                continue
            avg_ret = tgt[w_mask].mean()
            week_ret[w] = avg_ret

        week_ics[sym] = week_ret

    # Cross-ticker consensus: average weekly return
    all_weeks = sorted(set(w for d in week_ics.values() for w in d.keys()))
    week_avg = {}
    for w in all_weeks:
        vals = [d[w] for d in week_ics.values() if w in d]
        if len(vals) >= 3:
            week_avg[w] = np.mean(vals)

    # Identify strongest and weakest weeks
    sorted_weeks = sorted(week_avg.items(), key=lambda x: x[1], reverse=True)
    strong_weeks = [w for w, v in sorted_weeks[:10] if abs(v) > 0.01]
    weak_weeks   = [w for w, v in sorted_weeks[-10:] if abs(v) > 0.01]

    print(f"\n  Strongest weeks (avg class=UP > FLAT/DOWN across tickers):")
    for w, v in sorted_weeks[:10]:
        bar = "+" * int(abs(v) * 300)
        print(f"    Week {w:>2}: {v:>+8.4f} {bar}")

    print(f"\n  Weakest weeks:")
    for w, v in sorted_weeks[-10:]:
        bar = "-" * int(abs(v) * 300)
        print(f"    Week {w:>2}: {v:>+8.4f} {bar}")

    # Overall IC of week_of_year feature
    print(f"\n  Cross-ticker IC of week_of_year:")
    for sym, td in all_data.items():
        df = td["df"]
        tgt = td["target"]
        mask = td["mask"]
        r = compute_ic(df["week_of_year"][mask], tgt)
        sig = "*" if r["p"] < 0.05 else " "
        print(f"    {sym}: IC={r['ic']:>+8.4f}  p={r['p']:.4f} {sig}")

    # === Month of year IC ===
    print("\n\n" + "=" * 70)
    print("  MONTH OF YEAR vs. 6h forward return")
    print("=" * 70)

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    month_avgs = {}
    for m in range(1, 13):
        vals = []
        for sym, td in all_data.items():
            df = td["df"]
            tgt = td["target"]
            mask = td["mask"]
            m_mask = df["month_of_year"][mask] == m
            if m_mask.sum() >= 20:
                vals.append(tgt[m_mask].mean())
        if vals:
            month_avgs[m] = np.mean(vals)

    print("\n  Average class (2=UP, 1=FLAT, 0=DOWN) by month:")
    for m in range(1, 13):
        if m in month_avgs:
            v = month_avgs[m]
            diff_from_flat = v - 1.0  # 1.0 = neutral/FLAT
            bar = "+" * int(abs(diff_from_flat) * 100) if diff_from_flat > 0 else \
                  "-" * int(abs(diff_from_flat) * 100)
            print(f"    {month_names[m]}: avg_class={v:.4f}  delta={diff_from_flat:>+.4f} {bar}")

    # === Day of week IC ===
    print("\n\n" + "=" * 70)
    print("  DAY OF WEEK: IC + average return")
    print("  (Known: Fri avg IC=+0.097, Wed IC=-0.076, Thu IC=-0.044, Tue IC=+0.056)")
    print("=" * 70)

    day_names = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri"}
    print(f"\n  {'Day':<6} {'Avg IC':>10} ", end="")
    for sym in TICKERS:
        print(f" {sym:>7}", end="")
    print()
    print("  " + "-" * 70)

    for d in range(5):
        ics = []
        for sym, td in all_data.items():
            df = td["df"]
            tgt = td["target"]
            mask = td["mask"]
            d_mask = df["day_of_week"][mask] == d
            if d_mask.sum() < 20:
                ics.append(np.nan)
                continue
            # IC of "is_this_day" vs return
            x = (df["day_of_week"][mask] == d).astype(float)
            r = compute_ic(x, tgt)
            ics.append(r["ic"])

        avg_ic = float(np.nanmean(ics))
        print(f"  {day_names[d]:<6} {avg_ic:>+10.4f}", end="")
        for ic in ics:
            print(f" {ic:>+7.3f}", end="")
        print()

    # === Hour of day IC ===
    print("\n\n" + "=" * 70)
    print("  HOUR OF DAY: IC + average return by hour (UTC hours)")
    print("  Market hours: 13:30-20:00 UTC (EDT) or 14:30-21:00 UTC (EST)")
    print("=" * 70)

    hour_avgs = {}
    for h in range(13, 21):
        vals = []
        for sym, td in all_data.items():
            df = td["df"]
            tgt = td["target"]
            mask = td["mask"]
            h_mask = df["hour_of_day"][mask] == h
            if h_mask.sum() >= 20:
                vals.append(tgt[h_mask].mean())
        if vals:
            hour_avgs[h] = np.mean(vals)

    print("\n  Average class by hour (UTC):")
    for h, v in sorted(hour_avgs.items()):
        diff = v - 1.0
        bar = "+" * int(abs(diff) * 200) if diff > 0 else "-" * int(abs(diff) * 200)
        et = h - 4  # EDT offset
        print(f"    {h}:00 UTC ({et}:00 ET): avg_class={v:.4f}  delta={diff:>+.4f} {bar}")

    # === Seasonality features to add ===
    print("\n\n" + "=" * 70)
    print("  RECOMMENDED SEASONAL FEATURES TO ADD")
    print("=" * 70)

    strong_w = sorted_weeks[:5]
    weak_w   = sorted_weeks[-5:]

    print(f"\n  Strongest weeks (avg across tickers): {[w for w, _ in strong_w]}")
    print(f"  Weakest weeks:                        {[w for w, _ in weak_w]}")

    # Binary features for strong/weak week clusters
    print(f"""
  New features:
    is_strong_week: 1 if week_of_year in {sorted([w for w, v in sorted_weeks[:5] if v > 0])}
    is_weak_week:   1 if week_of_year in {sorted([w for w, v in sorted_weeks[-5:] if v < 0])}
    is_q_earnings:  1 if month in [1, 4, 7, 10] (earnings season months)
    is_summer:      1 if month in [7, 8] (low-vol summer period)
    is_sept_oct:    1 if month in [9, 10] (historically weak months)
    is_open_hour:   1 if hour_of_day in [13, 14] (UTC, first 2 hours = most volatile)
    is_power_hour:  1 if hour_of_day == 19 (UTC, last hour = strong momentum)
    week_sin:       sin(2*pi*week_of_year/52) — continuous seasonal encoding
    week_cos:       cos(2*pi*week_of_year/52) — orthogonal seasonal component
""")

    print("  Circular encoding for XGBoost (learns non-monotone patterns better):")
    print("  week_sin = sin(2*pi*week/52)  — peaks at spring/fall equinox")
    print("  week_cos = cos(2*pi*week/52)  — peaks at mid-year")
    print("  month_sin = sin(2*pi*month/12)")
    print("  month_cos = cos(2*pi*month/12)")
    print()
    print("  Implement in: data/indicators.py add_calendar_features()")
    print("  Add to: ml/features.py FEATURE_COLS")


if __name__ == "__main__":
    run()
