"""Re-optimize APP, AMD, GOOGL with n_ensemble=3 on current data."""
import sys
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')

from ml.optimizer import optimize_ticker
from edge_research import run_wf, print_table

TICKERS = ["APP", "AMD", "GOOGL"]
N_TRIALS = 60
N_ENS = 3

rows = []
for ticker in TICKERS:
    print(f"\n{'='*60}")
    print(f"  Optimizing {ticker} ({N_TRIALS} trials, n_ensemble={N_ENS})")
    print(f"{'='*60}")
    result = optimize_ticker(ticker, n_trials=N_TRIALS, n_ensemble=N_ENS)
    print(f"  {ticker} done: score={result.get('best_score', 0):.4f}")

    print(f"\n  Validating {ticker}...")
    r = run_wf(ticker, n_folds=4, n_ensemble=N_ENS, allow_short=False)
    rows.append(r)

print_table(rows, "Re-optimized tickers (n_ensemble=3)")
