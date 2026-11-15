"""
Batch backtest all trained tickers and rank by a composite score.
Run: python batch_eval.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data.universe import get_trained_tickers
from backtest.engine import backtest
from ml.model import RSIModel
from ml.optimizer import load_best_params
from config import CONFIG


def run_batch(skip_pct: float = 0.5, allow_short: bool = True):
    tickers = sorted(get_trained_tickers())
    print(f"\nBatch-evaluating {len(tickers)} tickers  (skip_pct={skip_pct}, shorts={'on' if allow_short else 'off'})\n")

    rows = []
    for ticker in tickers:
        try:
            model = RSIModel()
            model.load(ticker=ticker)
            result = backtest(
                ticker, model,
                period="2y",
                interval="1h",
                sentiment_score=0.0,
                skip_pct=skip_pct,
                allow_short=allow_short,
            )
            m = result["metrics"]
            optuna = load_best_params(ticker)
            rows.append({
                "ticker": ticker,
                "return": result["total_return_pct"],
                "win_rate": m["win_rate_pct"],
                "sharpe": m["sharpe_ratio"],
                "pf": m["profit_factor"],
                "trades": m["total_trades"],
                "dd": m["max_drawdown_pct"],
                "optuna": "Y" if optuna else "-",
            })
            status = f"  {ticker:<6} {result['total_return_pct']:>+7.2f}%  win={m['win_rate_pct']:.1f}%  trades={m['total_trades']}"
            print(status)
        except Exception as e:
            print(f"  {ticker:<6} ERROR: {e}")

    if not rows:
        print("No results.")
        return

    # Composite score: return * win_rate * log(trades+1) / max_drawdown_penalty
    import math
    for r in rows:
        dd_pen = max(abs(r["dd"]), 1.0)
        trade_bonus = math.log1p(r["trades"])
        # Reward positive return, high win rate, more trades, penalise drawdown
        r["score"] = (r["return"] * (r["win_rate"] / 100) * trade_bonus) / dd_pen

    rows.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'='*82}")
    print(f"  {'TICKER':<7} {'RETURN':>8} {'WIN%':>6} {'SHARPE':>7} {'PF':>6} {'TRADES':>7} {'DD%':>6} {'OPT':>4}  {'SCORE':>8}")
    print(f"{'='*82}")
    for r in rows:
        star = " *" if r["score"] > 0 and r["trades"] >= 10 else ""
        print(f"  {r['ticker']:<7} {r['return']:>+7.2f}%  {r['win_rate']:>5.1f}%  {r['sharpe']:>6.2f}  "
              f"{r['pf']:>5.2f}  {r['trades']:>6}  {r['dd']:>5.1f}%  {r['optuna']:>3}  {r['score']:>7.2f}{star}")
    print(f"{'='*82}")

    # Recommend top candidates for Optuna
    top_opt = [r["ticker"] for r in rows if r["score"] > 0 and r["trades"] >= 10 and r["optuna"] == "-"][:8]
    already_opt = [r["ticker"] for r in rows if r["score"] > 0 and r["trades"] >= 10 and r["optuna"] == "Y"]

    print(f"\n--- Optimize these next (positive score, 10+ trades, no Optuna params yet) ---")
    if top_opt:
        print(f"  python main.py optimize --tickers {' '.join(top_opt)} --trials 60 --retrain")
    else:
        print("  All top candidates already have Optuna params.")

    if already_opt:
        print(f"\n--- Already Optuna-tuned (verify with backtest) ---")
        print(f"  {' '.join(already_opt)}")

    # Annualised estimate (backtest covers ~1yr at skip_pct=0.5 on 2yr data)
    positive = [r for r in rows if r["return"] > 0 and r["trades"] >= 10]
    if positive:
        avg_return = sum(r["return"] for r in positive) / len(positive)
        print(f"\n--- Portfolio estimate ---")
        print(f"  {len(positive)} profitable tickers (10+ trades)  avg ~{avg_return:+.1f}% per ticker")
        print(f"  Diversified portfolio of top 10 -> estimated {avg_return * 0.7:+.1f}%-{avg_return:+.1f}% annual")


if __name__ == "__main__":
    run_batch()
