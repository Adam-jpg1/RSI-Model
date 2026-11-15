"""Quick backtest on the last 3 newly trained tickers."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from backtest.engine import backtest
from ml.model import RSIModel
from config import CONFIG

CONFIG.bot.max_position_pct = 0.10
NEW_TICKERS = ["COIN", "ADBE", "ANET"]

print(f"\n  {'TICKER':<7} {'RETURN':>8} {'WIN%':>6} {'TRADES':>7} {'SHARPE':>7} {'DD%':>6}")
print(f"  {'-'*48}")
for t in NEW_TICKERS:
    try:
        m = RSIModel()
        m.load(ticker=t)
        r = backtest(t, m, period="2y", interval="1h", sentiment_score=0.0, skip_pct=0.5, allow_short=True)
        m2 = r["metrics"]
        print(f"  {t:<7} {r['total_return_pct']:>+7.1f}%  {m2['win_rate_pct']:>5.1f}%  {m2['total_trades']:>6}  "
              f"{m2['sharpe_ratio']:>6.2f}  {m2['max_drawdown_pct']:>5.1f}%")
    except Exception as e:
        print(f"  {t}: ERROR {e}")
