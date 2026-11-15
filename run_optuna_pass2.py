"""
Second-pass Optuna optimization: uses new confidence tuning + better initialization
from first-pass params files. Run after run_optuna_all.py completes.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

TICKERS = ["AMD", "ANET", "APP", "AVGO", "GE", "T", "MU", "LRCX", "MSFT"]
N_TRIALS = 80  # more trials for second pass: better initial distribution from pass 1
N_FOLDS = 4

from ml.optimizer import optimize_ticker

results = {}
for ticker in TICKERS:
    print(f"\n=== Pass2: Optimizing {ticker} ({TICKERS.index(ticker)+1}/{len(TICKERS)}) ===")
    try:
        r = optimize_ticker(
            ticker,
            n_trials=N_TRIALS,
            n_folds=N_FOLDS,
            interval="1h",
            daily_confirm=True,
        )
        results[ticker] = r
        score = r["best_score"]
        adx = r["adx_filter_min"]
        hold = r["max_hold_bars"]
        conf = r.get("min_confidence", 0.52)
        print(f"[done] {ticker}: score={score:.4f}  adx={adx}  hold={hold}  conf={conf:.2f}")
    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        results[ticker] = {"best_score": -999.0, "error": str(e)}

print("\n\n=== PASS 2 OPTIMIZATION SUMMARY ===")
for ticker, r in results.items():
    score = r.get("best_score", -999)
    conf = r.get("min_confidence", 0.52)
    print(f"  {ticker:6s}: score={score:.4f}  conf={conf:.2f}")
