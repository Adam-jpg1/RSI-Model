#!/usr/bin/env python3
"""
Summarizes all experiment results from research_results.json and ranks
configurations against the daily_confirm baseline.
"""
import json, sys
from datetime import datetime

RESULTS_FILE = "research_results.json"
BASELINE_KEY = "experiment_B2_daily_confirm"


def avg_cagr(data):
    """Extract avg CAGR from various result shapes."""
    if isinstance(data, list):
        valid = [r for r in data if isinstance(r, dict) and not r.get("error") and r.get("total_trades", 0) >= 5]
        if not valid:
            return None
        return round(sum(r["cagr_pct"] for r in valid) / len(valid), 2)
    elif isinstance(data, dict):
        # Could be a combo dict: {label: {avg_cagr: X}}
        # or summary dict: {summary: {avg_cagr: X}, rows: [...]}
        if "summary" in data and isinstance(data["summary"], dict):
            return data["summary"].get("avg_cagr")
        # Combo experiment: find best sub-config
        candidates = []
        for k, v in data.items():
            if isinstance(v, dict) and "avg_cagr" in v:
                candidates.append((k, v["avg_cagr"]))
        if candidates:
            # Return (best_label, best_cagr)
            return candidates
    return None


def main():
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    # Get baseline
    baseline_data = results.get(BASELINE_KEY, {}).get("data")
    if baseline_data:
        baseline_cagr = avg_cagr(baseline_data)
        if isinstance(baseline_cagr, list):
            baseline_cagr = max(x[1] for x in baseline_cagr)
    else:
        baseline_cagr = None

    print(f"\n{'='*80}")
    print(f"  EXPERIMENT RESULTS SUMMARY  ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"  Baseline (daily_confirm): {baseline_cagr:+.2f}% avg CAGR" if baseline_cagr else "  No baseline found")
    print(f"{'='*80}")

    rows = []
    for key, entry in results.items():
        if key == BASELINE_KEY:
            continue
        if not isinstance(entry, dict) or "data" not in entry:
            continue
        data = entry["data"]
        ts = entry.get("timestamp", "")[:16]
        result = avg_cagr(data)
        if result is None:
            continue
        if isinstance(result, list):
            # Combo experiment — show best config
            best_label, best_cagr = max(result, key=lambda x: x[1])
            improvement = best_cagr - (baseline_cagr or 0)
            rows.append((key, f"best={best_label}", best_cagr, improvement, ts))
        else:
            improvement = result - (baseline_cagr or 0)
            rows.append((key, "", result, improvement, ts))

    # Sort by CAGR descending
    rows.sort(key=lambda x: -x[2])

    print(f"\n  {'EXPERIMENT':<40} {'BEST_CONFIG':<20} {'CAGR':>7} {'vs_BASE':>8}  {'TIME'}")
    print(f"  {'-'*40} {'-'*20} {'-'*7} {'-'*8}  {'-'*16}")
    for key, label, cagr, improvement, ts in rows:
        marker = " <-- BEST" if cagr == rows[0][2] else ""
        sign = "+" if improvement >= 0 else ""
        print(f"  {key:<40} {label:<20} {cagr:>+6.2f}% {sign}{improvement:>+6.2f}%  {ts}{marker}")

    print(f"\n  Baseline:  {BASELINE_KEY} = {baseline_cagr:+.2f}%")
    if rows:
        best = rows[0]
        print(f"  Best exp:  {best[0]} ({best[1]}) = {best[2]:+.2f}%  (delta: {best[3]:+.2f}%)")
    print()


if __name__ == "__main__":
    main()
