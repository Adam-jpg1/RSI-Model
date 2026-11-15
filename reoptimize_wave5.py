"""
Wave 5 Portfolio Re-Optimization

Critical fix added since Wave 4:
  - Earnings proximity features (days_to_earnings, days_since_earnings, near_earnings)
    now included in optimizer pipeline for the FIRST TIME.

  PEAD deep-dive IC findings:
    days_to_earnings:
      APP:   5-10 days before earnings: IC=-0.287  (p<0.001)
      AMD:   0-5  days before earnings: IC=-0.234  (p<0.001)
      GOOGL: 5-10 days before earnings: IC=-0.371  (p<0.001)
      MRVL:  0-5  days before earnings: IC=+0.268  (p<0.001)
      TSLA:  5-10 days before earnings: IC=+0.218  (p<0.001)

  These IC values (0.17-0.37) are an order of magnitude larger than most
  features. They were in FEATURE_COLS but NEVER computed during optimization.

  Additionaly: post-earnings drift is strongest 10-20 days AFTER earnings:
    AMD:   10-15d post: IC=+0.215  |  15-20d post: IC=+0.234
    GOOGL: 15-20d post: IC=+0.218
    APP:   15-20d post: IC=+0.190

Previous results:
  Wave 1 (baseline):              AVG CAGR = +4.7%
  Wave 2 (peer_RSI + open + macro): AVG CAGR = +2.9%  (regression)
  Wave 3 (calendar + cross-sect + fold_consistency): target >+4.7%
  Wave 4 (microstructure + regime + lead-lag): results pending
  Wave 5 target: beat all previous waves — earnings features add high-IC signal

Changes from Wave 4:
  - Same fold_consistency ** 2 penalty
  - Same max_depth <= 6
  - 100 trials per ticker
  - optimizer.py now calls add_earnings_features() — this is the key addition
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
print("  WAVE 5 PORTFOLIO RE-OPTIMIZATION")
print("  Critical: earnings proximity features added to optimizer for 1st time")
print("  Pre-earnings countdown IC: GOOGL -0.37, APP -0.29, AMD -0.23")
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

print_table(rows, "WAVE 5 — earnings proximity features added")
print()
print("Wave 1 baseline (old features):")
print("  APP   CAGR=+7.0%  trades=146  P(+)=99%")
print("  AMD   CAGR=+7.2%  trades=289  P(+)=99%")
print("  GOOGL CAGR=+3.1%  trades=181  P(+)=99%")
print("  SNOW  CAGR=+5.0%  trades=55   P(+)=99%")
print("  MRVL  CAGR=+3.3%  trades=74   P(+)=97%")
print("  TSLA  CAGR=+2.3%  trades=31   P(+)=94%")
print("  AVG   CAGR=+4.7%")
