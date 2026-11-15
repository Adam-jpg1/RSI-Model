"""
Wave 4 Portfolio Re-Optimization

New features added since Wave 3:
  - Bar microstructure: buying_pressure, bar_efficiency, wick_imbalance,
    momentum_efficiency_10, tick_imbal_z
    (buying_pressure IC=-0.024 overall, -0.049 in CHOPPY; tick_imbal_z IC=-0.025, -0.077 in CHOPPY)

  - Regime detection + interaction features:
    is_trending, is_choppy, regime_score
    pvd_choppy    = pvd_10 × is_choppy   (pvd_10 IC jumps from +0.017 to +0.117 in CHOPPY)
    tick_choppy   = tick_imbal_z × is_choppy
    opex_choppy   = opex_monthly × is_choppy  (IC jumps from +0.035 to +0.086 in CHOPPY)
    pvd_trending  = pvd_10 × is_trending

  Key insight: market spends 71.7% of time in NEUTRAL regime. Training one model
  for all regimes averages out the strong choppy-regime signals. Interaction features
  give XGBoost the information needed to apply each signal conditionally per regime.

Previous results:
  Wave 1 (baseline):              AVG CAGR = +4.7%
  Wave 2 (peer_RSI + open + macro): AVG CAGR = +2.9%  (regression from regime overfitting)
  Wave 3 (calendar + cross-sect + fold_consistency): target >+4.7%
  Wave 4 target: beat Wave 3 baseline with regime-conditional signals

Changes from Wave 3 optimizer:
  - Same fold_consistency ** 2 penalty
  - Same max_depth <= 6
  - 100 trials per ticker
  - All new feature functions called in optimizer pipeline
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

from data.fetcher import get_price_data
from ml.optimizer import optimize_ticker
from edge_research import run_wf, print_table

CONFIG_LIST = [
    # (ticker, n_ensemble)
    ("APP",   1),
    ("AMD",   1),
    ("GOOGL", 1),
    ("SNOW",  3),
    ("MRVL",  3),
    ("TSLA",  3),
]
ALL_TICKERS = [t for t, _ in CONFIG_LIST]
PERIOD   = "2y"
INTERVAL = "1h"

print("=" * 70)
print("  WAVE 4 PORTFOLIO RE-OPTIMIZATION")
print("  New: bar microstructure + regime detection + regime×signal interactions")
print("  pvd_choppy IC=+0.117 | tick_choppy IC=-0.077 | opex_choppy IC=+0.086")
print("=" * 70)

# Pre-fetch all tickers' Close series for cross-sectional features
print("\nPre-fetching all ticker data for cross-sectional rank features...")
all_close: dict = {}
for sym in ALL_TICKERS:
    try:
        df = get_price_data(sym, period=PERIOD, interval=INTERVAL)
        all_close[sym] = df["Close"].rename(sym)
        print(f"  {sym}: {len(df)} bars")
    except Exception as e:
        print(f"  {sym}: ERROR {e}")
        all_close[sym] = None

for ticker, n_ens in CONFIG_LIST:
    peer_close = {t: s for t, s in all_close.items()
                  if t != ticker and s is not None}
    print(f"\n--- {ticker} (n_ensemble={n_ens}, 100 trials) ---")
    result = optimize_ticker(
        ticker,
        n_trials=100,
        n_folds=4,
        period=PERIOD,
        interval=INTERVAL,
        daily_confirm=False,
        allow_short=False,
        n_ensemble=n_ens,
        peer_close_dict=peer_close,
    )
    score = result.get("best_score", "?")
    adx   = result.get("adx_filter_min", "?")
    conf  = result.get("min_confidence", "?")
    try:
        print(f"  DONE  score={score:.4f}  adx={adx:.1f}  conf={conf:.2f}")
    except Exception:
        print(f"  DONE  score={score}  adx={adx}  conf={conf}")

print("\n\n--- Final walk-forward validation ---\n")
rows = []
for ticker, n_ens in CONFIG_LIST:
    r = run_wf(ticker, n_folds=4, n_ensemble=n_ens, allow_short=False)
    rows.append(r)
    print(f"  {ticker}: CAGR={r['cagr']:+.1f}%  trades={r['total_trades']}  P(+)={r['p_profitable']:.0%}")

print_table(rows, "WAVE 4 — microstructure + regime interactions")
print()
print("Wave 3 target (calendar + cross-sectional + fold_consistency):")
print("  (results pending Wave 3 completion)")
print()
print("Wave 1 baseline (old features):")
print("  APP   CAGR=+7.0%  trades=146  P(+)=99%")
print("  AMD   CAGR=+7.2%  trades=289  P(+)=99%")
print("  GOOGL CAGR=+3.1%  trades=181  P(+)=99%")
print("  SNOW  CAGR=+5.0%  trades=55   P(+)=99%")
print("  MRVL  CAGR=+3.3%  trades=74   P(+)=97%")
print("  TSLA  CAGR=+2.3%  trades=31   P(+)=94%")
print("  AVG   CAGR=+4.7%")
