"""
Calendar Anomaly + Volatility Regime IC Research

Tests:
  1. Turn-of-month drift (first/last N trading days of month)
  2. Options expiration effects (monthly OpEx 3rd Friday, weekly OpEx Fridays)
  3. VIX term structure (VIX9D / VIX ratio — near-term fear relative to medium-term)
  4. VIX percentile regime (current VIX vs trailing 30-bar percentile)
  5. Day-of-week effect
  6. Realized vol regime (trailing 10h HV vs 30h HV — trending vs ranging)

Hypothesis: These calendar/vol features are largely orthogonal to RSI/MACD
and may add independent alpha, especially for options-heavy names like our universe.
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from datetime import date, timedelta

from data.fetcher import get_price_data
from data.indicators import add_all_indicators
from ml.features import target_bars_for_interval

TICKERS = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD  = "2y"
FWD     = target_bars_for_interval("1h")   # 6 bars forward


# ─────────────────────────────────────────────────────────────
#  Calendar flag builders
# ─────────────────────────────────────────────────────────────

def add_turn_of_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn-of-month effect:
      tom_buy  = 1 if bar is in first 3 trading days of the month
      tom_sell = 1 if bar is in last 2 trading days of the month
    Uses per-month trading-day rank, DST-safe.
    """
    df = df.copy()
    dt_index     = pd.to_datetime(df.index)
    date_series  = pd.Series(dt_index.normalize(), index=df.index)
    month_series = pd.Series(dt_index.to_period("M"), index=df.index)

    # Build rank table over unique (month, date) pairs
    unique_days = (
        pd.DataFrame({"date": date_series, "month": month_series})
        .drop_duplicates()
        .sort_values(["month", "date"])
        .reset_index(drop=True)
    )
    unique_days["rank_fwd"] = unique_days.groupby("month").cumcount() + 1
    unique_days["rank_bwd"] = (
        unique_days.groupby("month")["rank_fwd"]
        .transform(lambda g: g.max() - g + 1)
    )

    fwd_map = dict(zip(zip(unique_days["month"], unique_days["date"]), unique_days["rank_fwd"]))
    bwd_map = dict(zip(zip(unique_days["month"], unique_days["date"]), unique_days["rank_bwd"]))

    keys = list(zip(month_series.values, date_series.values))
    df["tom_buy"]  = [(1 if fwd_map.get(k, 99) <= 3 else 0) for k in keys]
    df["tom_sell"] = [(1 if bwd_map.get(k, 99) <= 2 else 0) for k in keys]
    return df


def add_opex_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Options expiration features:
      opex_monthly  = 1 on the 3rd Friday of each month (monthly options expire)
      opex_weekly   = 1 on any Friday (weekly options expire)
      opex_thursday = 1 on Thursday before monthly OpEx (position squaring)
    """
    df = df.copy()
    dt_index = pd.to_datetime(df.index)
    dayofweek = dt_index.dayofweek   # Mon=0, Fri=4

    # 3rd Friday of month
    day = dt_index.day
    is_friday = (dayofweek == 4)
    is_3rd_friday = is_friday & (day >= 15) & (day <= 21)

    df["opex_monthly"]  = is_3rd_friday.astype(int)
    df["opex_weekly"]   = is_friday.astype(int)

    # Thursday before 3rd Friday: 3rd Thursday = day 14-20 on Thursday
    is_thursday = (dayofweek == 3)
    df["opex_thursday"] = (is_thursday & (day >= 14) & (day <= 20)).astype(int)

    return df


def add_vix_term_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    VIX term structure features (merged from daily VIX9D and VIX).
    VIX9D < VIX  => front-end contango (complacency about near-term)
    VIX9D > VIX  => near-term fear premium (backwardation)

    vix_term_ratio = VIX9D / VIX  (>1 = near-term fear, <1 = complacency)
    vix_pct_rank   = VIX vs trailing 30-bar percentile (0=low vol, 1=high vol)
    """
    df = df.copy()
    try:
        vix9d = get_price_data("^VIX9D", period=PERIOD, interval="1d")
        vix   = get_price_data("^VIX",   period=PERIOD, interval="1d")

        if vix9d is None or vix is None or len(vix9d) < 20:
            raise ValueError("VIX data unavailable")

        vix9d_close = vix9d["Close"].rename("vix9d")
        vix_close   = vix["Close"].rename("vix")

        # Merge to daily series
        daily = pd.concat([vix9d_close, vix_close], axis=1).dropna()
        daily["vix_term_ratio"] = daily["vix9d"] / daily["vix"].replace(0, np.nan)
        daily["vix_pct_rank"]   = (
            daily["vix"].rolling(30, min_periods=10)
            .apply(lambda x: (x[-1] > x[:-1]).mean(), raw=True)
        )

        # Align daily features to hourly df via date join
        dt_dates = pd.to_datetime(df.index).normalize()
        daily_idx = pd.to_datetime(daily.index).normalize()

        term_map  = dict(zip(daily_idx, daily["vix_term_ratio"]))
        rank_map  = dict(zip(daily_idx, daily["vix_pct_rank"]))
        vix_map   = dict(zip(daily_idx, daily["vix"]))

        df["vix_term_ratio"] = [term_map.get(d, np.nan) for d in dt_dates]
        df["vix_pct_rank"]   = [rank_map.get(d, np.nan) for d in dt_dates]
        df["vix_level"]      = [vix_map.get(d, np.nan)  for d in dt_dates]

        # Forward-fill for days with no VIX data (weekends/holidays were NaN after normalize)
        df["vix_term_ratio"] = df["vix_term_ratio"].ffill()
        df["vix_pct_rank"]   = df["vix_pct_rank"].ffill()
        df["vix_level"]      = df["vix_level"].ffill()

    except Exception as e:
        print(f"  [warn] VIX term structure unavailable: {e}")
        df["vix_term_ratio"] = np.nan
        df["vix_pct_rank"]   = np.nan
        df["vix_level"]      = np.nan

    return df


def add_vol_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realized volatility regime based on intraday returns.
      rv_short  = rolling 10-bar HV annualized
      rv_long   = rolling 30-bar HV annualized
      rv_ratio  = rv_short / rv_long (>1 = rising vol, <1 = contracting vol)
      is_low_vol = 1 if rv_short < 20th percentile of trailing 60-bar rv_short
    """
    df = df.copy()
    ret = df["Close"].pct_change()

    rv_10 = ret.rolling(10).std() * np.sqrt(252 * 7)   # annualize (7 bars/day)
    rv_30 = ret.rolling(30).std() * np.sqrt(252 * 7)

    df["rv_ratio"]  = (rv_10 / rv_30.replace(0, np.nan)).clip(0.1, 5.0)
    df["rv_regime"] = (
        rv_10.rolling(60, min_periods=20)
        .apply(lambda x: (x[-1] > x[:-1]).mean(), raw=True)
    )
    return df


def add_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week dummies: 0=Mon, 4=Fri."""
    df = df.copy()
    dow = pd.to_datetime(df.index).dayofweek
    df["dow"] = dow.values
    for d, name in zip(range(5), ["is_mon", "is_tue", "is_wed", "is_thu", "is_fri"]):
        df[name] = (dow == d).astype(int)
    return df


# ─────────────────────────────────────────────────────────────
#  IC computation
# ─────────────────────────────────────────────────────────────

FEATURES = [
    # Turn-of-month
    "tom_buy", "tom_sell",
    # OpEx
    "opex_monthly", "opex_weekly", "opex_thursday",
    # VIX term structure
    "vix_term_ratio", "vix_pct_rank", "vix_level",
    # Vol regime
    "rv_ratio", "rv_regime",
    # Day-of-week
    "is_mon", "is_tue", "is_wed", "is_thu", "is_fri",
]


def compute_ic_for_ticker(ticker: str) -> pd.DataFrame:
    df = get_price_data(ticker, period=PERIOD, interval="1h")
    df = add_all_indicators(df)
    df = add_turn_of_month(df)
    df = add_opex_features(df)
    df = add_vix_term_structure(df)
    df = add_vol_regime(df)
    df = add_day_of_week(df)

    future_ret = df["Close"].shift(-FWD) / df["Close"] - 1.0

    rows = []
    for col in FEATURES:
        if col not in df.columns:
            continue
        common = future_ret.dropna().index.intersection(df[col].dropna().index)
        if len(common) < 50:
            continue
        ic, pval = spearmanr(df.loc[common, col], future_ret.loc[common], nan_policy="omit")
        if np.isnan(ic):
            continue

        # For binary features: mean return on event vs non-event
        vals = df.loc[common, col]
        is_binary = vals.isin([0, 1]).all()
        if is_binary and vals.sum() > 10:
            mask = vals == 1
            lift = future_ret.loc[common][mask].mean() - future_ret.loc[common][~mask].mean()
            n_event = int(mask.sum())
        else:
            lift = np.nan
            n_event = len(common)

        rows.append({
            "ticker": ticker,
            "feature": col,
            "IC": round(ic, 5),
            "pval": round(pval, 5),
            "significant": pval < 0.05,
            "n_bars": len(common),
            "n_event": n_event,
            "lift_pct": round(lift * 100, 4) if not np.isnan(lift) else np.nan,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def run_all():
    print("=" * 72)
    print("  Calendar Anomaly + Volatility Regime IC Research")
    print("  Features: ToM, OpEx, VIX term structure, vol regime, DoW")
    print("=" * 72)

    all_dfs = []
    for ticker in TICKERS:
        print(f"\n[cal-vol-ic] {ticker}...", flush=True)
        try:
            result = compute_ic_for_ticker(ticker)
            all_dfs.append(result)
            sig = result[result["significant"]]
            if not sig.empty:
                for _, row in sig.sort_values("IC", key=abs, ascending=False).iterrows():
                    print(f"  ** {row['feature']:<20}  IC={row['IC']:+.4f}  p={row['pval']:.4f}"
                          f"  lift={row['lift_pct']:+.3f}%  n={row['n_event']}")
            else:
                print("  (no significant features)")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}")

    if not all_dfs:
        print("No results.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    print("\n\n" + "=" * 72)
    print("  SUMMARY: Cross-ticker IC averages (all features)")
    print("=" * 72)
    print(f"  {'Feature':<22} {'Avg_IC':>8} {'% Sig':>6} {'Avg_lift%':>11} {'Avg_n_event':>12}")
    print("  " + "-" * 65)

    for feat in FEATURES:
        sub = combined[combined["feature"] == feat]
        if sub.empty:
            continue
        avg_ic   = sub["IC"].mean()
        pct_sig  = sub["significant"].mean() * 100
        avg_lift = sub["lift_pct"].dropna().mean()
        avg_n    = sub["n_event"].mean()
        flag     = " <--" if pct_sig >= 50 and abs(avg_ic) >= 0.03 else ""
        print(f"  {feat:<22} {avg_ic:>+8.4f}  {pct_sig:>5.0f}%  {avg_lift:>+10.4f}  {avg_n:>11.0f}{flag}")

    print("\n\nSignificant results (p<0.05, |IC|>0.02):")
    sig_all = combined[(combined["significant"]) & (combined["IC"].abs() >= 0.02)]
    sig_all = sig_all.sort_values("IC", key=abs, ascending=False)
    print(f"  {'Ticker':<6} {'Feature':<22} {'IC':>8} {'p':>8} {'lift%':>10} {'n_event':>8}")
    print("  " + "-" * 65)
    for _, row in sig_all.iterrows():
        print(f"  {row['ticker']:<6} {row['feature']:<22} {row['IC']:>+8.4f}  "
              f"{row['pval']:>7.4f}  {row['lift_pct']:>+9.4f}  {row['n_event']:>7}")

    print(f"\nTotal significant features: {len(sig_all)}")
    print(f"Top candidates (|IC|>0.03 AND p<0.05 in >=3 tickers):")
    top = (sig_all.groupby("feature")
           .filter(lambda g: len(g) >= 3 and g["IC"].abs().mean() >= 0.03))
    if not top.empty:
        for feat in top["feature"].unique():
            s = top[top["feature"] == feat]
            print(f"  {feat}: avg_IC={s['IC'].mean():+.4f}  tickers={list(s['ticker'])}")
    else:
        print("  None meeting criteria")

    return combined


if __name__ == "__main__":
    run_all()
