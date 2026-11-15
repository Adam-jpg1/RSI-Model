"""
Analyze exit reason distribution and avg PnL by exit type for a sample of tickers.
Run: python analyze_exits.py
Helps decide if take_profit/stop_loss thresholds should be adjusted.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backtest.engine import backtest
from ml.model import RSIModel
from config import CONFIG


SAMPLE_TICKERS = ["AMD", "QCOM", "CRM", "TSLA", "META", "NVDA", "GS", "AMZN"]


def run():
    print(f"\nExit Reason Analysis  (stop={CONFIG.bot.stop_loss_pct*100:.1f}%  take={CONFIG.bot.take_profit_pct*100:.1f}%)\n")
    print(f"  {'TICKER':<7} {'TOTAL':>6} {'TAKE%':>7} {'STOP%':>7} {'SIG%':>7} {'EOD%':>6}  {'TAKE_AVG':>9} {'STOP_AVG':>9} {'SIG_AVG':>9}")
    print(f"  {'-'*78}")

    for ticker in SAMPLE_TICKERS:
        try:
            m = RSIModel()
            m.load(ticker=ticker)
            result = backtest(ticker, m, period="2y", interval="1h",
                              sentiment_score=0.0, skip_pct=0.5,
                              allow_short=CONFIG.bot.allow_short)

            trades = result["trades"]
            closed = trades[trades["type"].isin(["SELL", "COVER"])]
            if len(closed) == 0:
                continue

            by_reason = closed.groupby("exit_reason")
            total = len(closed)

            def pct_and_avg(reason):
                g = by_reason.get_group(reason) if reason in by_reason.groups else None
                if g is None or len(g) == 0:
                    return 0.0, 0.0
                return len(g) / total * 100, g["pnl_pct"].mean()

            take_pct, take_avg = pct_and_avg("take")
            stop_pct, stop_avg = pct_and_avg("stop")
            sig_pct, sig_avg   = pct_and_avg("signal")
            eod_pct, _          = pct_and_avg("eod")

            print(f"  {ticker:<7} {total:>6} {take_pct:>6.1f}%  {stop_pct:>6.1f}%  "
                  f"{sig_pct:>6.1f}%  {eod_pct:>5.1f}%  "
                  f"{take_avg:>+8.2f}%  {stop_avg:>+8.2f}%  {sig_avg:>+8.2f}%")
        except Exception as e:
            print(f"  {ticker}: ERROR {e}")

    print(f"\n  TAKE = hit take-profit,  STOP = hit stop-loss,  SIG = model signal flip,  EOD = forced close")
    print(f"\n  Insight: if SIG exits dominate and SIG_AVG >> 0, consider raising take_profit.")
    print(f"  If STOP% is high, consider widening stop_loss.")


if __name__ == "__main__":
    run()
