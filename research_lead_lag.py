"""
Lead-Lag Pairs Research

Hypothesis: In tech/semiconductor sectors, large-cap price moves propagate to
correlated stocks with 1-6 hour delay (institutional sector rotation, ETF
rebalancing, risk-on/risk-off flows).

If NVDA has a strong move, AMD/MRVL should follow within 1-3 hours.
If SPY rallies strongly, tech stocks follow.
If TSLA moves (beta=1.8), SNOW/APP sentiment adjusts.

Edge: Existing signals (rsi_14, macd_hist) only use the ticker's OWN price history.
      Lead-lag signals add a "look-ahead" into the near future by using correlated
      tickers' CURRENT state as a predictor of THIS ticker's NEXT state.

Methodology:
  1. For each ticker pair (A, B): compute IC of A's current 6h return vs B's next 6h return
  2. Test lags 1, 2, 3, 6 hours
  3. Any pair with |IC| > 0.03 at any lag = worth adding as a feature

Pairs to test (by sector proximity):
  Semiconductor: NVDA -> AMD, NVDA -> MRVL, AMD -> MRVL
  AI platform:   NVDA -> APP, AMD -> APP
  Growth tech:   APP -> TSLA (risk-on)
  Market beta:   SPY -> each ticker
  Broad tech:    XLK -> each ticker

Feature design (if signals found):
  nvda_ret6_lag1  = NVDA's return over last 6h (lagged 1 bar = 1h delay)
  nvda_ret6_lag2  = NVDA's return over last 12h vs 6h ago
  spy_ret6_lag1   = SPY's last 6h return
  sector_lead_z   = standardized version of leading ticker's return
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from scipy import stats
from itertools import product

from data.fetcher import get_price_data
from ml.features import build_target, target_bars_for_interval

TICKERS  = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
LEADERS  = ["NVDA", "SPY", "XLK", "QQQ"]  # potential leaders to test
PERIOD   = "2y"
INTERVAL = "1h"
N_BARS   = target_bars_for_interval(INTERVAL)  # 6h forward

LAGS = [1, 2, 3, 6]  # hours to lag the leader's return


def compute_ic(x: pd.Series, y: pd.Series) -> dict:
    common = x.dropna().index.intersection(y.dropna().index)
    if len(common) < 50:
        return {"ic": np.nan, "p": np.nan, "n": 0}
    rho, p = stats.spearmanr(x.loc[common], y.loc[common])
    return {"ic": rho, "p": p, "n": len(common)}


def run():
    print("=" * 70)
    print("  Lead-Lag Pairs Research")
    print(f"  Leaders: {LEADERS} -> Followers: {TICKERS}")
    print(f"  Test: IC of leader's N-bar return vs follower's next-{N_BARS}-bar return")
    print("=" * 70)

    # Pre-fetch leader data
    print("\nFetching leader data...")
    leader_data = {}
    for sym in LEADERS:
        try:
            df = get_price_data(sym, period=PERIOD, interval=INTERVAL)
            leader_data[sym] = df["Close"]
            print(f"  {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym}: ERROR {e}")

    # Pre-fetch all ticker data (also test cross-ticker leads within portfolio)
    print("\nFetching portfolio ticker data...")
    ticker_data = {}
    for sym in TICKERS:
        try:
            df = get_price_data(sym, period=PERIOD, interval=INTERVAL)
            ticker_data[sym] = df["Close"]
            print(f"  {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym}: ERROR {e}")

    all_leaders = {**leader_data, **ticker_data}  # test cross-ticker leads too

    results = []

    print("\n\nTest results (leader -> follower, lag):")
    print(f"  {'Leader':<8} {'Follower':<8} {'Lag':>5} {'IC':>8} {'p-val':>8} {'n':>6} sig")
    print("  " + "-" * 55)

    for leader_sym, follower_sym in product(list(all_leaders.keys()), TICKERS):
        if leader_sym == follower_sym:
            continue

        leader_close   = all_leaders.get(leader_sym)
        follower_close = ticker_data.get(follower_sym)

        if leader_close is None or follower_close is None:
            continue

        # Build follower target: next N_BARS return
        follower_target = build_target(
            pd.DataFrame({"Close": follower_close}),
            n_bars=N_BARS
        )

        for lag in LAGS:
            # Leader signal: return over last N_BARS bars (6h), lagged by `lag` hours
            leader_signal = leader_close.pct_change(N_BARS).shift(lag)

            ic_res = compute_ic(leader_signal, follower_target)
            sig = "*" if ic_res["p"] < 0.05 and abs(ic_res["ic"]) > 0.02 else " "
            strong = "**" if ic_res["p"] < 0.01 and abs(ic_res["ic"]) > 0.04 else sig

            if abs(ic_res["ic"]) > 0.02 or not np.isnan(ic_res["ic"]):
                results.append({
                    "leader":   leader_sym,
                    "follower": follower_sym,
                    "lag":      lag,
                    "ic":       ic_res["ic"],
                    "p":        ic_res["p"],
                    "n":        ic_res["n"],
                    "sig":      strong,
                })

            if abs(ic_res["ic"]) > 0.025:
                print(f"  {leader_sym:<8} {follower_sym:<8} {lag:>5}h {ic_res['ic']:>+8.4f} "
                      f"{ic_res['p']:>8.4f} {ic_res['n']:>6} {strong}")

    # Summary: best leads per follower
    print("\n\n" + "=" * 70)
    print("  BEST LEAD-LAG SIGNALS per follower ticker")
    print("=" * 70)

    significant = [r for r in results if abs(r["ic"]) > 0.03 and r["p"] < 0.05]

    for follower in TICKERS:
        follower_results = [r for r in significant if r["follower"] == follower]
        follower_results.sort(key=lambda x: abs(x["ic"]), reverse=True)
        if follower_results:
            print(f"\n  {follower}:")
            for r in follower_results[:5]:
                print(f"    {r['leader']:<8} lag={r['lag']}h  IC={r['ic']:+.4f}  p={r['p']:.4f} {r['sig']}")

    # Feature recommendation
    print("\n\n" + "=" * 70)
    print("  FEATURES TO ADD (leader x lag pairs with |IC| > 0.04)")
    print("=" * 70)
    top_leads = sorted(significant, key=lambda x: abs(x["ic"]), reverse=True)
    added_features = set()
    for r in top_leads:
        if abs(r["ic"]) > 0.04:
            feat_name = f"{r['leader'].lower().replace('^','')}_ret6_lag{r['lag']}"
            if feat_name not in added_features:
                print(f"  add_feature: {feat_name:<35} -> {r['follower']}  IC={r['ic']:+.4f}")
                added_features.add(feat_name)

    if not added_features:
        print("  No pairs exceeded IC > 0.04 threshold.")

    # Strongest consistent signals across multiple lags
    print("\n\nSTRONGEST cross-asset leads (consistent IC across 2+ lags):")
    pair_groups = {}
    for r in results:
        key = (r["leader"], r["follower"])
        if key not in pair_groups:
            pair_groups[key] = []
        pair_groups[key].append(r)

    consistent = []
    for (lead, follow), lag_results in pair_groups.items():
        strong_lags = [r for r in lag_results if abs(r["ic"]) > 0.03 and r["p"] < 0.05]
        if len(strong_lags) >= 2:
            avg_ic = np.mean([abs(r["ic"]) for r in strong_lags])
            consistent.append((lead, follow, avg_ic, strong_lags))

    consistent.sort(key=lambda x: x[2], reverse=True)
    for lead, follow, avg_ic, strong_lags in consistent[:10]:
        print(f"  {lead:<8} -> {follow:<8}  avg|IC|={avg_ic:.4f}  "
              f"n_strong_lags={len(strong_lags)}")


if __name__ == "__main__":
    run()
