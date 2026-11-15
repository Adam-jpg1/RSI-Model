"""
Cross-Sectional Momentum IC Research

Hypothesis: Within the 6-ticker portfolio, the RELATIVE STRENGTH RANK of a
stock vs its peers has predictive power for the next 6 hours.

This is a signal most per-ticker ML models miss because features are computed
in isolation. A stock at rank=1 (strongest recent performer in the group)
may continue outperforming (momentum) or revert (crowding/rotation).

We test:
  cs_rank_5d   — rank of trailing 5-day return among peers (0=weakest, 1=strongest)
  cs_rank_10d  — rank of trailing 10-day return
  cs_rank_20d  — rank of trailing 20-day return
  cs_rank_1d   — rank of trailing 1-day return (mean-reversion timescale)
  cs_rsi_rank  — rank of current RSI14 among peers (sector overbought/oversold)
  peer_spread  — ticker's trailing 5d return minus average peer return (absolute excess)

If IC is consistent (p<0.05 for 3+ tickers), we build a cross-sectional
feature pipeline that all tickers can use.
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data.fetcher import get_price_data
from data.indicators import add_all_indicators
from ml.features import target_bars_for_interval

TICKERS = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD  = "2y"
FWD     = target_bars_for_interval("1h")


def build_cross_sectional_features(all_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Given hourly price data for all tickers, compute cross-sectional rank features.
    Returns updated DataFrames keyed by ticker.
    """
    # Align all close prices to common index
    close_frames = {}
    rsi_frames   = {}

    for ticker, df in all_data.items():
        close_frames[ticker] = df["Close"].rename(ticker)
        if "RSI_14" in df.columns:
            rsi_frames[ticker] = df["RSI_14"].rename(ticker + "_rsi")

    # Build a common close panel (outer join, then ffill)
    close_panel = pd.concat(close_frames.values(), axis=1).sort_index().ffill()
    rsi_panel   = pd.concat(rsi_frames.values(), axis=1).sort_index().ffill() if rsi_frames else None

    # Compute trailing returns
    ret_1d  = close_panel.pct_change(7)    # ~1 trading day = 7 hourly bars
    ret_5d  = close_panel.pct_change(35)   # ~5 trading days
    ret_10d = close_panel.pct_change(70)   # ~10 trading days
    ret_20d = close_panel.pct_change(140)  # ~20 trading days

    n = len(TICKERS)

    def rank_row(row: pd.Series) -> pd.Series:
        """Rank a row: 0 = weakest, (n-1)/(n-1) = 1.0 = strongest. NaN → NaN."""
        ranked = row.rank(method="average", na_option="keep") - 1
        return ranked / max(n - 1, 1)

    # Cross-sectional rank at each bar
    cs_1d  = ret_1d.apply(rank_row, axis=1)
    cs_5d  = ret_5d.apply(rank_row, axis=1)
    cs_10d = ret_10d.apply(rank_row, axis=1)
    cs_20d = ret_20d.apply(rank_row, axis=1)
    cs_rsi = rsi_panel.apply(rank_row, axis=1) if rsi_panel is not None else None

    # Peer spread: ticker excess return vs equal-weight peers
    spread_5d = {}
    for ticker in TICKERS:
        peer_avg = ret_5d[[t for t in TICKERS if t != ticker]].mean(axis=1)
        spread_5d[ticker] = ret_5d[ticker] - peer_avg

    # Augment each ticker's DataFrame
    result = {}
    for ticker, df in all_data.items():
        df2 = df.copy()
        idx = df2.index

        def _align(panel_col):
            return panel_col.reindex(idx).ffill()

        df2["cs_rank_1d"]   = _align(cs_1d[ticker]).values
        df2["cs_rank_5d"]   = _align(cs_5d[ticker]).values
        df2["cs_rank_10d"]  = _align(cs_10d[ticker]).values
        df2["cs_rank_20d"]  = _align(cs_20d[ticker]).values
        if cs_rsi is not None and ticker in cs_rsi.columns:
            df2["cs_rsi_rank"]  = _align(cs_rsi[ticker]).values
        df2["peer_spread_5d"] = _align(spread_5d[ticker]).values

        result[ticker] = df2

    return result


def compute_ic(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    future_ret = df["Close"].shift(-FWD) / df["Close"] - 1.0
    features   = ["cs_rank_1d", "cs_rank_5d", "cs_rank_10d",
                  "cs_rank_20d", "cs_rsi_rank", "peer_spread_5d"]

    rows = []
    for col in features:
        if col not in df.columns:
            continue
        common = future_ret.dropna().index.intersection(df[col].dropna().index)
        if len(common) < 50:
            continue
        ic, pval = spearmanr(df.loc[common, col], future_ret.loc[common], nan_policy="omit")
        if np.isnan(ic):
            continue

        # Quintile analysis: does top quintile outperform bottom?
        vals = df.loc[common, col]
        fret = future_ret.loc[common]
        q20  = vals.quantile(0.20)
        q80  = vals.quantile(0.80)
        bot_ret = fret[vals <= q20].mean()
        top_ret = fret[vals >= q80].mean()
        quintile_spread = top_ret - bot_ret

        rows.append({
            "ticker":   ticker,
            "feature":  col,
            "IC":       round(ic, 5),
            "pval":     round(pval, 5),
            "sig":      pval < 0.05,
            "n_bars":   len(common),
            "top_q_ret_pct":  round(top_ret * 100, 4),
            "bot_q_ret_pct":  round(bot_ret * 100, 4),
            "q_spread_pct":   round(quintile_spread * 100, 4),
        })

    return pd.DataFrame(rows)


def run_all():
    print("=" * 70)
    print("  Cross-Sectional Momentum IC Research")
    print(f"  Universe: {TICKERS}")
    print(f"  Forward horizon: {FWD} bars ({FWD}h)")
    print("=" * 70)

    # Fetch all data
    print("\nFetching all ticker data...")
    all_data = {}
    for ticker in TICKERS:
        try:
            df = get_price_data(ticker, period=PERIOD, interval="1h")
            df = add_all_indicators(df)
            all_data[ticker] = df
            print(f"  {ticker}: {len(df)} bars")
        except Exception as e:
            print(f"  {ticker}: ERROR {e}")

    if len(all_data) < 2:
        print("Need at least 2 tickers.")
        return

    print("\nBuilding cross-sectional rank features...")
    all_data = build_cross_sectional_features(all_data)

    print("\nComputing IC for each ticker...")
    all_ic = []
    for ticker, df in all_data.items():
        result = compute_ic(ticker, df)
        all_ic.append(result)

        sig = result[result["sig"]]
        if not sig.empty:
            for _, row in sig.sort_values("IC", key=abs, ascending=False).iterrows():
                print(f"  ** {ticker} {row['feature']:<18}  IC={row['IC']:+.4f}  p={row['pval']:.4f}"
                      f"  q_spread={row['q_spread_pct']:+.3f}%")
        else:
            # Print best anyway
            if not result.empty:
                best = result.loc[result["IC"].abs().idxmax()]
                print(f"     {ticker} best: {best['feature']:<18}  IC={best['IC']:+.4f}  p={best['pval']:.4f}")

    combined = pd.concat(all_ic, ignore_index=True)

    print("\n\n" + "=" * 70)
    print("  SUMMARY: Cross-ticker IC averages")
    print("=" * 70)
    print(f"  {'Feature':<20} {'Avg_IC':>8} {'% Sig':>7} {'Avg_q_spread%':>14}")
    print("  " + "-" * 55)

    features = ["cs_rank_1d", "cs_rank_5d", "cs_rank_10d", "cs_rank_20d",
                "cs_rsi_rank", "peer_spread_5d"]
    for feat in features:
        sub = combined[combined["feature"] == feat]
        if sub.empty:
            continue
        avg_ic  = sub["IC"].mean()
        pct_sig = sub["sig"].mean() * 100
        avg_qs  = sub["q_spread_pct"].mean()
        flag    = " <-- STRONG" if pct_sig >= 67 and abs(avg_ic) >= 0.03 else (
                  " <--" if pct_sig >= 50 and abs(avg_ic) >= 0.02 else "")
        print(f"  {feat:<20} {avg_ic:>+8.4f}  {pct_sig:>6.0f}%  {avg_qs:>+13.4f}{flag}")

    # Direction analysis: momentum or mean-reversion?
    print("\n\nDirection: momentum (IC>0) = high rank -> strong future return")
    print("          mean-reversion (IC<0) = high rank -> weak future return")
    for feat in ["cs_rank_5d", "cs_rank_10d"]:
        sub = combined[combined["feature"] == feat]
        if sub.empty: continue
        pos = (sub["IC"] > 0).sum()
        neg = (sub["IC"] < 0).sum()
        avg = sub["IC"].mean()
        print(f"  {feat}: {pos} momentum / {neg} mean-reversion  avg_IC={avg:+.4f}")

    # Final verdict
    print("\n\nVERDICT:")
    top_feats = combined[(combined["sig"]) & (combined["IC"].abs() >= 0.025)]
    if not top_feats.empty:
        for feat in top_feats["feature"].unique():
            s = top_feats[top_feats["feature"] == feat]
            avg_ic = s["IC"].mean()
            tickers = list(s["ticker"])
            direction = "momentum" if avg_ic > 0 else "mean-reversion"
            print(f"  {feat}: avg_IC={avg_ic:+.4f}  tickers={tickers}  [{direction}]")
    else:
        print("  No features met |IC|>=0.025 and p<0.05 threshold.")

    return combined


if __name__ == "__main__":
    run_all()
