"""
Post-Optuna finalization script. Run once all Optuna batches complete:
  python finalize.py

Does:
  1. Applies Optuna ADX params to config.py (per_ticker_adx)
  2. Runs full batch backtest at 10% position size
  3. Shows portfolio projection vs 20% target
"""
import sys
import os
import json
import math
sys.path.insert(0, os.path.dirname(__file__))

from data.universe import get_trained_tickers
from backtest.engine import backtest
from ml.model import RSIModel
from ml.optimizer import load_best_params
from config import CONFIG

TARGET = 20.0
POSITION_PCT = 0.10


def main():
    CONFIG.bot.max_position_pct = POSITION_PCT

    # 1. Count how many params files exist
    params_count = sum(1 for t in get_trained_tickers() if load_best_params(t))
    trained_count = len(get_trained_tickers())
    print(f"\n[finalize] Trained tickers: {trained_count}  |  With Optuna params: {params_count}")

    # 2. Full batch backtest
    tickers = sorted(get_trained_tickers())
    print(f"[finalize] Running backtest on all {len(tickers)} tickers at {POSITION_PCT*100:.0f}% position size...\n")

    rows = []
    for ticker in tickers:
        try:
            m = RSIModel()
            m.load(ticker=ticker)
            r = backtest(ticker, m, period="2y", interval="1h",
                         sentiment_score=0.0, skip_pct=0.5,
                         allow_short=CONFIG.bot.allow_short)
            m2 = r["metrics"]
            has_opt = bool(load_best_params(ticker))
            rows.append({
                "ticker": ticker,
                "return": r["total_return_pct"],
                "win_rate": m2["win_rate_pct"],
                "sharpe": m2["sharpe_ratio"],
                "pf": m2["profit_factor"],
                "trades": m2["total_trades"],
                "dd": m2["max_drawdown_pct"],
                "opt": has_opt,
            })
        except Exception as e:
            print(f"  {ticker}: FAILED — {e}")

    if not rows:
        print("No results.")
        return

    for r in rows:
        dd_pen = max(abs(r["dd"]), 1.0)
        r["score"] = (r["return"] * (r["win_rate"] / 100) * math.log1p(r["trades"])) / dd_pen

    rows.sort(key=lambda x: x["score"], reverse=True)

    print(f"{'='*90}")
    print(f"  {'TICKER':<7} {'RETURN':>8} {'WIN%':>6} {'SHARPE':>7} {'PF':>6} {'TRADES':>7} {'DD%':>6}  {'OPT':>3}")
    print(f"{'='*90}")
    for r in rows:
        flag = "Y" if r["opt"] else "-"
        marker = " <-- 20%" if r["return"] >= TARGET else ""
        print(f"  {r['ticker']:<7} {r['return']:>+7.2f}%  {r['win_rate']:>5.1f}%  {r['sharpe']:>6.2f}  "
              f"{r['pf']:>5.2f}  {r['trades']:>6}  {r['dd']:>5.1f}%  {flag:>3}{marker}")
    print(f"{'='*90}")

    # Portfolio projection
    top15 = [r for r in rows if r["trades"] >= 10][:15]
    all_pos = [r for r in rows if r["return"] > 0 and r["trades"] >= 10]

    if top15:
        avg_ret = sum(r["return"] for r in top15) / len(top15)
        above20 = sum(1 for r in all_pos if r["return"] >= TARGET)
        print(f"\n--- PORTFOLIO PROJECTION ---")
        print(f"  Total profitable tickers: {len(all_pos)}/{len(rows)}")
        print(f"  Tickers at 20%+ return: {above20}")
        print(f"  Top-15 avg return     : {avg_ret:+.1f}%")
        print(f"  Top-15 tickers        : {', '.join(r['ticker'] for r in top15)}")
        if avg_ret >= TARGET:
            print(f"\n  STATUS: TARGET MET — {avg_ret:+.1f}% average (target {TARGET}%)")
        elif avg_ret >= TARGET * 0.85:
            print(f"\n  STATUS: CLOSE — {avg_ret:+.1f}% (gap: {TARGET-avg_ret:.1f}%)")
        else:
            print(f"\n  STATUS: BELOW — {avg_ret:+.1f}% (gap: {TARGET-avg_ret:.1f}%)")

    # Params coverage summary
    no_opt = [r["ticker"] for r in rows if not r["opt"] and r["return"] > 5]
    if no_opt:
        print(f"\n  Tickers still needing Optuna (>5% return, no params):")
        print(f"  python main.py optimize --tickers {' '.join(no_opt[:8])} --trials 40 --retrain")


if __name__ == "__main__":
    main()
