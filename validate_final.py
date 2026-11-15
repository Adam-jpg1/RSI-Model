"""Final 6-ticker portfolio validation."""
import sys
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')

from edge_research import run_wf, print_table

# n_ensemble must match what was used during Optuna optimization:
#   APP/AMD/GOOGL → optimized with n_ensemble=1 (single-model Optuna)
#   SNOW/MRVL/TSLA → re-optimized with n_ensemble=3
portfolio = [
    ("APP",   1, False),
    ("AMD",   1, False),
    ("GOOGL", 1, False),
    ("SNOW",  3, False),
    ("MRVL",  3, False),
    ("TSLA",  3, False),
]

rows = []
for ticker, n_ens, short in portfolio:
    print(f"[{ticker}]...")
    r = run_wf(ticker, n_folds=4, n_ensemble=n_ens, allow_short=short)
    rows.append(r)

print_table(rows, "FINAL PORTFOLIO — 6 tickers (APP AMD GOOGL SNOW MRVL TSLA)")
