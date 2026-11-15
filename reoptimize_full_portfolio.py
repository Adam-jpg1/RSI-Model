"""
Full portfolio re-optimization with all new features:
  - Cross-asset peer RSI lags (nvda_rsi14, xlk_rsi14, vix_rsi14)
  - Opening-hour interaction features (open_gap_x, open_macd_slope_x)
  - Macro calendar events (is_fomc_day, fomc_post_decision, is_fomc_next, is_nfp_day)

Run order: APP, AMD, GOOGL, SNOW, MRVL, TSLA
n_ensemble per ticker matches the validated configuration.
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

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

print("=" * 65)
print("  FULL PORTFOLIO RE-OPTIMIZATION")
print("  Features: peer_RSI_lags + open_interactions + macro_events")
print("=" * 65)

for ticker, n_ens in CONFIG_LIST:
    print(f"\n--- {ticker} (n_ensemble={n_ens}, 60 trials) ---")
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
    score = result.get("best_score", "?")
    adx = result.get("adx_filter_min", "?")
    conf = result.get("min_confidence", "?")
    print(f"  DONE  score={score:.4f}  adx={adx:.1f}  conf={conf:.2f}")

print("\n\n--- Final walk-forward validation with new params ---\n")
rows = []
for ticker, n_ens in CONFIG_LIST:
    r = run_wf(ticker, n_folds=4, n_ensemble=n_ens, allow_short=False)
    rows.append(r)
    print(f"  {ticker}: CAGR={r['cagr']:+.1f}%  trades={r['total_trades']}  P(+)={r['p_profitable']:.0%}")

print_table(rows, "FULL PORTFOLIO — all new features (peer_RSI + open_x + macro_events)")
print()
print("Previous best (without new features):")
print("  APP   CAGR=+7.0%  trades=146  P(+)=99%")
print("  AMD   CAGR=+7.2%  trades=289  P(+)=99%")
print("  GOOGL CAGR=+3.1%  trades=181  P(+)=99%")
print("  SNOW  CAGR=+5.0%  trades=55   P(+)=99%")
print("  MRVL  CAGR=+3.3%  trades=74   P(+)=97%")
print("  TSLA  CAGR=+2.3%  trades=31   P(+)=94%")
print("  AVG   CAGR=+4.7%")
