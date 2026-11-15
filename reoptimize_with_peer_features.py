"""
Re-optimize APP and MRVL with new cross-asset peer RSI features.
These features have IC 0.05-0.09 but require Optuna re-tuning to integrate properly.
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from ml.optimizer import optimize_ticker
from edge_research import run_wf, print_table

# Tickers with strongest new cross-asset signals:
# APP: nvda_rsi14_lag6 IC=-0.086 (best signal)
# MRVL: vix_rsi14_lag1 IC=+0.062, xlk_rsi14_lag1 IC=-0.056
# Run n_ensemble=1 for APP (consistent with current setup) and n_ensemble=3 for MRVL
REOPT_CONFIG = [
    ("APP",  1),
    ("MRVL", 3),
]

print("=" * 65)
print("  Re-optimizing with cross-asset peer RSI features")
print("  (nvda_rsi14 lags, xlk_rsi14 lags, vix_rsi14 lags)")
print("=" * 65)

for ticker, n_ens in REOPT_CONFIG:
    print(f"\n--- Optimizing {ticker} (n_ensemble={n_ens}, 60 trials) ---")
    result = optimize_ticker(
        ticker,
        n_trials=60,
        n_folds=4,
        period="2y",
        interval="1h",
        daily_confirm=False,
        allow_short=False,
        n_ensemble=n_ens,
    )
    print(f"  Best score: {result.get('best_score', '?'):.4f}")
    print(f"  ADX: {result.get('adx_filter_min', '?'):.1f}")
    print(f"  conf: {result.get('min_confidence', '?'):.2f}")

print("\n--- Post-optimization walk-forward validation ---")
rows = []
for ticker, n_ens in REOPT_CONFIG:
    r = run_wf(ticker, n_folds=4, n_ensemble=n_ens, allow_short=False)
    rows.append(r)

print_table(rows, "APP+MRVL after re-optimization with peer RSI features")
print()
print("Reference (without peer features):")
print("  APP  : CAGR=+7.0%  trades=146  P(+)=99%")
print("  MRVL : CAGR=+3.3%  trades=74   P(+)=97%")
