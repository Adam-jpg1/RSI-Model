"""Quick test: compare 4% vs 5% take_profit on AMD, QCOM, CRM."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from backtest.engine import backtest
from ml.model import RSIModel
from config import CONFIG

TICKERS = ["AMD", "QCOM", "CRM", "TSLA", "META", "NVDA"]

print(f"\n{'='*58}")
print(f"  Take-profit comparison  (stop={CONFIG.bot.stop_loss_pct*100:.1f}%)")
print(f"  {'TICKER':<7} {'4% TP':>8} {'5% TP':>8}  {'DELTA':>8}")
print(f"{'='*58}")

for ticker in TICKERS:
    m = RSIModel()
    m.load(ticker=ticker)
    CONFIG.bot.take_profit_pct = 0.04
    r4 = backtest(ticker, m, period="2y", interval="1h", sentiment_score=0.0, skip_pct=0.5, allow_short=True)
    CONFIG.bot.take_profit_pct = 0.05
    r5 = backtest(ticker, m, period="2y", interval="1h", sentiment_score=0.0, skip_pct=0.5, allow_short=True)
    delta = r5["total_return_pct"] - r4["total_return_pct"]
    print(f"  {ticker:<7} {r4['total_return_pct']:>+7.2f}%  {r5['total_return_pct']:>+7.2f}%  {delta:>+7.2f}%")

print(f"{'='*58}")
CONFIG.bot.take_profit_pct = 0.05
print(f"\n5% take_profit is now active in config.")
