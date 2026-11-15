"""Quick test: compare 8% vs 10% max_position_pct on key tickers."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from backtest.engine import backtest
from ml.model import RSIModel
from config import CONFIG

TICKERS = ["AMD", "QCOM", "CRM", "TSLA", "META", "CAT", "GS", "GE", "AMZN", "XOM", "NVDA", "JPM"]

print(f"\n{'='*72}")
print(f"  Position size comparison  (TP={CONFIG.bot.take_profit_pct*100:.0f}%  SL={CONFIG.bot.stop_loss_pct*100:.0f}%)")
print(f"  {'TICKER':<7} {'8% return':>10} {'10% return':>10}  {'8% DD':>7} {'10% DD':>7}  {'DELTA':>7}")
print(f"{'='*72}")

totals = {8: [], 10: []}
for ticker in TICKERS:
    m = RSIModel()
    m.load(ticker=ticker)

    CONFIG.bot.max_position_pct = 0.08
    r8 = backtest(ticker, m, period="2y", interval="1h", sentiment_score=0.0, skip_pct=0.5, allow_short=True)

    CONFIG.bot.max_position_pct = 0.10
    r10 = backtest(ticker, m, period="2y", interval="1h", sentiment_score=0.0, skip_pct=0.5, allow_short=True)

    delta = r10["total_return_pct"] - r8["total_return_pct"]
    totals[8].append(r8["total_return_pct"])
    totals[10].append(r10["total_return_pct"])
    print(f"  {ticker:<7} {r8['total_return_pct']:>+9.2f}%  {r10['total_return_pct']:>+9.2f}%  "
          f"{r8['metrics']['max_drawdown_pct']:>6.1f}%  {r10['metrics']['max_drawdown_pct']:>6.1f}%  {delta:>+6.2f}%")

avg8 = sum(totals[8]) / len(totals[8])
avg10 = sum(totals[10]) / len(totals[10])
print(f"{'='*72}")
print(f"  {'AVERAGE':<7} {avg8:>+9.2f}%  {avg10:>+9.2f}%")
CONFIG.bot.max_position_pct = 0.08
print(f"\n  Config is 8% (unchanged). To use 10%, set max_position_pct=0.10 in config.py")
