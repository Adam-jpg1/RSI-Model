"""
Run backtest on all trained tickers with Optuna params applied and project portfolio return.
Run: python portfolio_report.py
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(__file__))

from data.universe import get_trained_tickers
from backtest.engine import backtest
from ml.model import RSIModel
from ml.optimizer import load_best_params
from config import CONFIG

TARGET_RETURN = 20.0
TOP_N_PORTFOLIO = 15


def run_report(skip_pct: float = 0.5):
    tickers = sorted(get_trained_tickers())
    print(f"\nPortfolio Report — {len(tickers)} tickers, skip_pct={skip_pct} (~1yr test window)")
    print(f"Target: {TARGET_RETURN}% annual return\n")

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
                allow_short=CONFIG.bot.allow_short,
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
                "adx": optuna["adx_filter_min"] if optuna else 0.0,
                "has_optuna": bool(optuna),
            })
        except Exception as e:
            print(f"  {ticker} ERROR: {e}")

    if not rows:
        print("No results.")
        return

    # Score for ranking
    for r in rows:
        dd_pen = max(abs(r["dd"]), 1.0)
        r["score"] = (r["return"] * (r["win_rate"] / 100) * math.log1p(r["trades"])) / dd_pen

    rows.sort(key=lambda x: x["score"], reverse=True)

    print(f"{'='*90}")
    print(f"  {'TICKER':<7} {'RETURN':>8} {'WIN%':>6} {'SHARPE':>7} {'PF':>6} {'TRADES':>7} {'DD%':>6} {'ADX':>5}  {'OPT':>3}")
    print(f"{'='*90}")
    for r in rows:
        flag = "Y" if r["has_optuna"] else "-"
        marker = " <-- TARGET" if r["return"] >= TARGET_RETURN else ""
        print(f"  {r['ticker']:<7} {r['return']:>+7.2f}%  {r['win_rate']:>5.1f}%  {r['sharpe']:>6.2f}  "
              f"{r['pf']:>5.2f}  {r['trades']:>6}  {r['dd']:>5.1f}%  {r['adx']:>4.0f}  {flag:>3}{marker}")
    print(f"{'='*90}")

    # Portfolio projection using top N tickers
    good = [r for r in rows if r["return"] > 0 and r["trades"] >= 10]
    top = good[:TOP_N_PORTFOLIO]

    if top:
        avg_return = sum(r["return"] for r in top) / len(top)
        avg_winrate = sum(r["win_rate"] for r in top) / len(top)
        above_target = sum(1 for r in top if r["return"] >= TARGET_RETURN)

        print(f"\n--- Portfolio Projection (top {len(top)} tickers by composite score) ---")
        print(f"  Tickers : {', '.join(r['ticker'] for r in top)}")
        print(f"  Avg return  : {avg_return:+.1f}%")
        print(f"  Avg win rate: {avg_winrate:.1f}%")
        print(f"  Tickers >= {TARGET_RETURN}%: {above_target}/{len(top)}")
        # Conservative: each ticker contributes avg_return × its deployment share
        # With 15 max positions at 8% and ~25% utilization per ticker:
        # deployed_pct ≈ min(len(top) * 8%, 80%) = max portfolio usage
        deployed = min(len(top) * 0.08, 0.80)
        projected = avg_return * deployed / 0.08  # normalize back: each ticker's return is on 8% capital
        # Simpler: in multi-ticker mode, portfolio return ~ avg individual return * utilization
        # utilization = (avg_trades * avg_hold_hours) / (252 * 6.5)
        # avg_hold_hours estimated at 3h per trade
        avg_trades = sum(r["trades"] for r in top) / len(top)
        avg_hold = 3.0
        utilization = min((avg_trades * avg_hold) / (252 * 6.5), 1.0)
        utilization_pct = utilization * 100
        # Return at 8% position: individual return numbers already assume 8% sizing
        # Portfolio = sum of all tickers' alpha (they run simultaneously, capital rotates)
        print(f"  Avg trades/yr: {avg_trades:.0f}  est. utilization: {utilization_pct:.0f}%")
        print(f"\n  PROJECTED PORTFOLIO RETURN: ~{avg_return * 0.75:+.1f}% to {avg_return:+.1f}%")
        if avg_return >= TARGET_RETURN:
            print(f"  STATUS: ON TRACK for {TARGET_RETURN}% target")
        elif avg_return >= TARGET_RETURN * 0.8:
            print(f"  STATUS: CLOSE — {TARGET_RETURN - avg_return:.1f}% gap, optimize weaker tickers")
        else:
            print(f"  STATUS: BELOW TARGET — focus on higher-scoring tickers only")

    # Identify tickers still needing optimization
    needs_opt = [r["ticker"] for r in rows if not r["has_optuna"] and r["return"] > 0 and r["trades"] >= 10]
    if needs_opt:
        print(f"\n  Still needing Optuna: {' '.join(needs_opt)}")
        print(f"  Run: python main.py optimize --tickers {' '.join(needs_opt[:8])} --trials 60 --retrain")


if __name__ == "__main__":
    run_report()
