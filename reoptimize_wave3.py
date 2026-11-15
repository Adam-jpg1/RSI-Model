"""
Wave 3 Portfolio Re-Optimization

New features added since wave 2:
  - Calendar / day-of-week: is_fri (+0.097 avg IC), is_wed (-0.076), is_thu (-0.044),
    opex_monthly (+0.056), is_tue (+0.056)
  - Cross-sectional rank: cs_rank_5d (-0.021), peer_spread_5d (-0.020)
    (mean-reversion for 5/6 tickers; momentum for APP)

Optimizer improvements:
  - fold_consistency ** 2 penalty: models that only trade in 1-2 folds are discarded
  - max_depth capped at 6 (was 8) to prevent overfitting on 100+ features
  - 100 trials per ticker (was 60) for better Optuna convergence

Previous results:
  Wave 1 (baseline):              AVG CAGR = +4.7%
  Wave 2 (peer_RSI + open + macro): AVG CAGR = +2.9%  ← regression from regime overfitting
  Wave 3 target: beat +4.7% with consistent cross-fold signals

Pre-fetch strategy: fetch all 6 tickers' Close series once, pass as peer_close_dict
to each optimize_ticker() call for cross-sectional rank features.
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
print("  WAVE 3 PORTFOLIO RE-OPTIMIZATION")
print("  New: calendar (is_fri/wed/thu, opex_monthly) + cross-sectional rank")
print("  Fix: fold_consistency penalty + max_depth<=6")
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

print_table(rows, "WAVE 3 — calendar + cross-sectional + consistency fix")
print()
print("Wave 2 (peer_RSI + open + macro):")
print("  APP   CAGR=+2.5%  trades=25   P(+)=99%  (1-fold regression)")
print("  AMD   CAGR=+3.1%  trades=90   P(+)=90%")
print("  GOOGL CAGR=+2.2%  trades=83   P(+)=95%")
print("  SNOW  CAGR=+5.1%  trades=55   P(+)=91%")
print("  MRVL  CAGR=+3.1%  trades=19   P(+)=97%  (1-fold)")
print("  TSLA  CAGR=+1.4%  trades=13   P(+)=92%")
print("  AVG   CAGR=+2.9%")
print()
print("Wave 1 baseline (old features):")
print("  APP   CAGR=+7.0%  trades=146  P(+)=99%")
print("  AMD   CAGR=+7.2%  trades=289  P(+)=99%")
print("  GOOGL CAGR=+3.1%  trades=181  P(+)=99%")
print("  SNOW  CAGR=+5.0%  trades=55   P(+)=99%")
print("  MRVL  CAGR=+3.3%  trades=74   P(+)=97%")
print("  TSLA  CAGR=+2.3%  trades=31   P(+)=94%")
print("  AVG   CAGR=+4.7%")
