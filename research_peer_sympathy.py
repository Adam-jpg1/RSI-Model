"""
Peer Earnings Sympathy Research

Hypothesis: When a peer tech stock reports a strong earnings beat,
related stocks drift in the same direction — especially in the 1-2 weeks
before THEY report their own earnings. This is the "sympathy play."

This is DIFFERENT from PEAD (which tracks a ticker's OWN gap).
This tracks: how do PEER earnings gaps predict THIS ticker's return?

Expected mechanism:
  - AMD beats → investors bid up NVDA/MRVL in anticipation of sector strength
  - GOOGL misses → SNAP/META sell off on sector contagion fears
  - Effect peaks: 1-7 days after peer event, fades after 2 weeks
  - Effect strongest when target ticker is ALSO approaching its own earnings

New features proposed:
  peer_sym_score   — exponential decay-weighted sum of recent peer earnings gaps
  peer_sym_bull    — 1 if net sympathy is positive (bullish peer earnings context)
  peer_sym_x_daze  — peer_sym_score × (1 - days_to_own_earnings/30) — amplified near own report

This research tests IC of these features against 6h forward return.
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from scipy import stats

from data.fetcher import get_price_data, get_earnings_dates
from data.indicators import add_all_indicators
from ml.features import build_target, target_bars_for_interval

TICKERS  = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
# Extended peers: also include NVDA, META, INTC for broader sector context
PEERS_UNIVERSE = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA", "NVDA", "META"]
PERIOD   = "2y"
INTERVAL = "1h"
DECAY_DAYS = 14.0   # half-life for recency weighting


def compute_ic(x: pd.Series, y: pd.Series) -> dict:
    common = x.dropna().index.intersection(y.dropna().index)
    if len(common) < 30:
        return {"ic": np.nan, "p": np.nan, "n": 0}
    rho, p = stats.spearmanr(x.loc[common], y.loc[common])
    return {"ic": float(rho), "p": float(p), "n": len(common)}


def get_earnings_gaps(ticker: str, hourly_df: pd.DataFrame) -> list[dict]:
    """
    Returns list of {date, gap} for detected earnings events.
    Uses actual yfinance earnings dates + open/close gap to measure surprise.
    Falls back to 3-sigma overnight gap detection.
    """
    edates = get_earnings_dates(ticker)
    if not edates:
        return []

    daily_close = hourly_df["Close"].resample("1D").last().dropna()
    gap_events = []
    for edate in edates:
        edate_pd = pd.Timestamp(edate).tz_localize("America/New_York") if hasattr(pd.Timestamp(edate), 'tz') and pd.Timestamp(edate).tz is None else pd.Timestamp(edate)
        # Find closest trading day
        try:
            edate_pd = pd.Timestamp(edate)
            # Find daily close before and after earnings
            before = daily_close[daily_close.index.date < edate_pd.date()]
            after  = daily_close[daily_close.index.date >= edate_pd.date()]
            if len(before) < 1 or len(after) < 1:
                continue
            close_before = float(before.iloc[-1])
            close_after  = float(after.iloc[0])
            gap = (close_after / close_before) - 1.0
            if abs(gap) > 0.001:  # ignore sub-0.1% gaps (not real earnings gaps)
                gap_events.append({"date": edate_pd.date(), "gap": gap})
        except Exception:
            continue

    return gap_events


def compute_peer_sympathy_features(
    target_ticker: str,
    target_df: pd.DataFrame,
    peer_gaps: dict,  # {ticker: [{date, gap}, ...]}
    target_edates: list,
) -> pd.DataFrame:
    """
    For each hourly bar in target_df, compute:
      peer_sym_score : exp-decay weighted sum of peer gaps in last 30 days
      peer_sym_bull  : 1 if score > 0
      days_to_own    : days until next own earnings event (or 999 if unknown)
      peer_sym_x_daze: peer_sym_score amplified near own earnings
    """
    idx = pd.to_datetime(target_df.index)
    n = len(idx)

    peer_sym_score = np.zeros(n, dtype=np.float32)
    days_to_own    = np.full(n, 999.0, dtype=np.float32)

    # Pre-compute target earnings date list (sorted)
    target_dates = sorted([pd.Timestamp(d).date() for d in target_edates]) if target_edates else []

    # All peer events: flat list of (date, gap) pairs
    all_peer_events = []
    for t, events in peer_gaps.items():
        if t == target_ticker:
            continue
        for ev in events:
            all_peer_events.append((ev["date"], ev["gap"]))
    all_peer_events.sort(key=lambda x: x[0])

    for i, ts in enumerate(idx):
        bar_date = ts.date()

        # Peer sympathy: sum recent peer gaps with exponential decay
        score = 0.0
        for ev_date, ev_gap in all_peer_events:
            days_ago = (bar_date - ev_date).days
            if days_ago < 0:
                continue  # future event, skip
            if days_ago > 60:
                continue  # too old, skip
            weight = np.exp(-days_ago / DECAY_DAYS)
            score += ev_gap * weight
        peer_sym_score[i] = score

        # Days to own earnings
        if target_dates:
            future = [d for d in target_dates if d > bar_date]
            if future:
                days_to_own[i] = (future[0] - bar_date).days

    result = pd.DataFrame(index=target_df.index)
    result["peer_sym_score"] = peer_sym_score
    result["peer_sym_bull"]  = (peer_sym_score > 0).astype(np.int8)
    result["days_to_own"]    = days_to_own
    # Interaction: sympathy signal amplified when close to own earnings
    # Clamp days_to_own at 30 for scaling (beyond 30 days = no interaction)
    proximity = (1.0 - np.minimum(days_to_own, 30.0) / 30.0)  # 1 = earnings tomorrow
    result["peer_sym_x_prox"] = peer_sym_score * proximity

    return result


def run():
    n_bars = target_bars_for_interval(INTERVAL)

    print("=" * 70)
    print("  PEER EARNINGS SYMPATHY RESEARCH")
    print("  Hypothesis: peer earnings beats predict target drift before OWN report")
    print(f"  Tickers: {TICKERS}")
    print(f"  Peers universe: {PEERS_UNIVERSE}")
    print(f"  Decay half-life: {DECAY_DAYS} days")
    print("=" * 70)

    # Fetch all hourly data
    print("\nFetching hourly data...")
    all_hourly = {}
    for sym in PEERS_UNIVERSE:
        try:
            df = get_price_data(sym, period=PERIOD, interval=INTERVAL)
            df = add_all_indicators(df)
            all_hourly[sym] = df
            print(f"  {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym}: ERROR {e}")

    # Compute earnings gaps for all peers
    print("\nComputing earnings gaps for all peers...")
    all_peer_gaps = {}
    all_edates    = {}
    for sym in PEERS_UNIVERSE:
        df = all_hourly.get(sym)
        if df is None:
            continue
        edates = get_earnings_dates(sym)
        all_edates[sym] = edates or []
        gaps = get_earnings_gaps(sym, df)
        all_peer_gaps[sym] = gaps
        gap_vals = [g["gap"] for g in gaps]
        if gap_vals:
            print(f"  {sym}: {len(gaps)} events  avg_gap={np.mean(gap_vals):+.2%}  "
                  f"max={max(gap_vals):+.2%}  min={min(gap_vals):+.2%}")
        else:
            print(f"  {sym}: 0 events detected")

    # For each target ticker, compute IC of sympathy features
    print("\n\n" + "=" * 70)
    print("  IC ANALYSIS: PEER SYMPATHY vs 6h forward return")
    print("=" * 70)

    results = {}
    for target in TICKERS:
        df = all_hourly.get(target)
        if df is None:
            continue

        tgt = build_target(df, n_bars=n_bars)
        mask = tgt.notna() & (np.arange(len(df)) < len(df) - n_bars)

        sym_feats = compute_peer_sympathy_features(
            target, df, all_peer_gaps, all_edates.get(target, [])
        )

        tgt_m = tgt[mask]

        print(f"\n  {target}:")
        ticker_results = {}
        for feat in ["peer_sym_score", "peer_sym_bull", "peer_sym_x_prox"]:
            feat_m = sym_feats.loc[mask, feat] if feat in sym_feats.columns else pd.Series(np.nan, index=tgt_m.index)
            r = compute_ic(feat_m, tgt_m)
            sig = "*" if r["p"] < 0.05 else " "
            print(f"    {feat:<22}: IC={r['ic']:>+8.4f}  p={r['p']:.4f} {sig}  n={r['n']}")
            ticker_results[feat] = r
        results[target] = ticker_results

    # Cross-ticker summary
    print("\n\n" + "=" * 70)
    print("  CROSS-TICKER AVERAGE IC")
    print("=" * 70)
    for feat in ["peer_sym_score", "peer_sym_bull", "peer_sym_x_prox"]:
        ics = [results[t][feat]["ic"] for t in TICKERS if t in results and
               results[t][feat]["ic"] is not np.nan and not np.isnan(results[t][feat]["ic"])]
        if ics:
            avg_ic = np.mean(ics)
            sig_count = sum(1 for t in TICKERS if t in results and
                           results[t][feat].get("p", 1.0) < 0.05)
            print(f"  {feat:<22}: avg IC={avg_ic:>+8.4f}  "
                  f"significant in {sig_count}/{len(TICKERS)} tickers")

    # Time-decay analysis: does IC decay with days_since_peer_event?
    print("\n\n" + "=" * 70)
    print("  FRESHNESS ANALYSIS: IC vs days since most recent peer event")
    print("  (Do fresh peer events <7d have stronger IC than older events 7-14d?)")
    print("=" * 70)

    for target in TICKERS:
        df = all_hourly.get(target)
        if df is None:
            continue

        tgt = build_target(df, n_bars=n_bars)
        mask = tgt.notna() & (np.arange(len(df)) < len(df) - n_bars)
        tgt_m = tgt[mask]

        # For each bar: what's the age of most recent peer event?
        idx = pd.to_datetime(df.index)
        all_peer_dates = sorted(set(
            ev["date"]
            for sym, events in all_peer_gaps.items()
            if sym != target
            for ev in events
        ))

        if not all_peer_dates:
            continue

        days_since_peer = np.full(len(df), 999.0)
        for i, ts in enumerate(idx):
            bar_date = ts.date()
            past = [d for d in all_peer_dates if d <= bar_date]
            if past:
                days_since_peer[i] = (bar_date - past[-1]).days

        days_s = pd.Series(days_since_peer, index=df.index)

        # IC for fresh events (0-7 days) vs medium (7-14 days) vs stale (14-30 days)
        sym_feats = compute_peer_sympathy_features(
            target, df, all_peer_gaps, all_edates.get(target, [])
        )

        print(f"\n  {target}:")
        for label, cond in [("0-7d",  (days_s <= 7)),
                             ("7-14d", (days_s > 7) & (days_s <= 14)),
                             ("14-30d",(days_s > 14) & (days_s <= 30))]:
            subset_mask = mask & cond
            if subset_mask.sum() < 30:
                continue
            r = compute_ic(sym_feats.loc[subset_mask, "peer_sym_score"], tgt[subset_mask])
            sig = "*" if r["p"] < 0.05 else " "
            print(f"    since_peer={label:<6}: IC={r['ic']:>+8.4f}  p={r['p']:.4f} {sig}  n={r['n']}")

    # Magnitude analysis: does SIZE of peer gap matter?
    print("\n\n" + "=" * 70)
    print("  MAGNITUDE ANALYSIS: Small vs large peer earnings gaps")
    print("  (Do big beats/misses create stronger sympathy than small ones?)")
    print("=" * 70)

    for target in ["AMD", "MRVL"]:  # Two most correlated pairs
        df = all_hourly.get(target)
        if df is None:
            continue

        tgt = build_target(df, n_bars=n_bars)
        mask = tgt.notna() & (np.arange(len(df)) < len(df) - n_bars)
        tgt_m = tgt[mask]

        sym_feats = compute_peer_sympathy_features(
            target, df, all_peer_gaps, all_edates.get(target, [])
        )

        # Split by absolute score magnitude
        abs_score = sym_feats["peer_sym_score"].abs()
        q33 = abs_score[mask].quantile(0.33)
        q66 = abs_score[mask].quantile(0.66)

        print(f"\n  {target}:")
        for label, cond in [("Low |score|",  (abs_score <= q33)),
                             ("Med |score|",  (abs_score > q33) & (abs_score <= q66)),
                             ("High |score|", (abs_score > q66))]:
            sub = mask & cond
            if sub.sum() < 30:
                continue
            r = compute_ic(sym_feats.loc[sub, "peer_sym_score"], tgt[sub])
            print(f"    {label:<14}: IC={r['ic']:>+8.4f}  p={r['p']:.4f}  n={r['n']}")

    # Recommendation
    print("\n\n" + "=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)

    avg_ics = {}
    for feat in ["peer_sym_score", "peer_sym_bull", "peer_sym_x_prox"]:
        ics = [results[t][feat]["ic"] for t in TICKERS if t in results and
               not np.isnan(results[t][feat].get("ic", np.nan))]
        avg_ics[feat] = np.mean(ics) if ics else 0.0

    best_feat = max(avg_ics, key=lambda k: abs(avg_ics[k]))
    best_ic   = avg_ics[best_feat]
    print(f"\n  Best feature: {best_feat}  avg IC={best_ic:>+.4f}")

    if abs(best_ic) > 0.03:
        print("  VERDICT: Meaningful signal — add peer_sympathy features to indicators.py")
        print("  Implement in: data/indicators.py add_peer_sympathy_features()")
        print("  Add to: ml/features.py FEATURE_COLS")
        print("  Strongest in: check per-ticker breakdown above")
    elif abs(best_ic) > 0.01:
        print("  VERDICT: Marginal signal — worth including but not transformative")
    else:
        print("  VERDICT: No meaningful signal — peer earnings sympathy not predictive")
        print("  Possible reason: price-return cross-ticker features already capture this")


if __name__ == "__main__":
    run()
