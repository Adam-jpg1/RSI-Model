"""
RSI(2) paper-test: last-2-year out-of-sample evaluation.

Reads best_params from rsi2_results.json (produced by rsi2_strategy.py),
then runs a fresh backtest on the most recent 2 years of daily data.
No re-fitting; parameters are taken straight from the IS optimisation.

Usage:
  python rsi2_paper_test.py                     # last 2 calendar years
  python rsi2_paper_test.py --start 2023-01-01  # custom start date
"""

import os, sys, json, math, argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

# Reuse all strategy logic from rsi2_strategy.py
sys.path.insert(0, os.path.dirname(__file__))

try:
    from rsi2_news import NewsJudgment
    _HAS_NEWS = True
except ImportError:
    _HAS_NEWS = False
import rsi2_strategy as _strat
from rsi2_strategy import (
    TICKERS, COST_PER_SIDE, ANNUAL_BARS, RESULTS_JSON,
    fetch_prices, fetch_cash_rate, fetch_vix, backtest_one, combine_returns,
    compute_metrics, bah_metrics, _split_params,
)

TODAY      = datetime.today().strftime("%Y-%m-%d")
TWO_YR_AGO = (datetime.today() - relativedelta(years=2)).strftime("%Y-%m-%d")


# ── helpers ───────────────────────────────────────────────────────────────────
def load_best_params(json_path: str) -> dict:
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"{json_path} not found — run rsi2_strategy.py first."
        )
    with open(json_path) as fh:
        data = json.load(fh)
    return data["best_params"]


def pct(x):
    return f"{x:>+8.2f}%"

def flt(x, d=3):
    return f"{x:>+8.{d}f}"


def print_comparison(strategy: dict, bah: dict, label: str):
    rows = [
        ("Metric",       f"RSI(2) {label}", "B&H SPY"),
        ("CAGR",         pct(strategy["cagr_pct"]),         pct(bah["cagr_pct"])),
        ("Sharpe",       flt(strategy["sharpe"]),            flt(bah["sharpe"])),
        ("Deflated SR",  f"{strategy['deflated_sharpe']:>8.4f}",
                         f"{bah['deflated_sharpe']:>8.4f}"),
        ("Max Drawdown", pct(strategy["max_drawdown_pct"]),  pct(bah["max_drawdown_pct"])),
        ("Win Rate",     f"{strategy['win_rate_pct']:>7.1f}%", "    n/a"),
        ("Exposure",     f"{strategy['exposure_pct']:>7.1f}%", "  100.0%"),
        ("# Trades",     f"{strategy['n_trades']:>8}",       "        1"),
        ("Avg days/trade",f"{strategy['avg_trade_days']:>7.1f}", "      n/a"),
        ("Total return", pct(strategy["total_return_pct"]),  pct(bah["total_return_pct"])),
    ]
    w = [18, 18, 12]
    print()
    for r in rows:
        print("  " + "".join(str(c).ljust(w[min(i, 2)]) for i, c in enumerate(r)))


def print_trade_log(trades: pd.DataFrame):
    if len(trades) == 0:
        print("  No completed trades in this period.")
        return
    print(f"\n  {'Entry':<12} {'Exit':<12} {'Days':>5} {'Ret%':>8}  Result")
    print("  " + "-" * 46)
    for _, row in trades.sort_values("entry_date").iterrows():
        flag = "WIN " if row["win"] else "LOSS"
        print(f"  {str(row['entry_date'])[:10]:<12} "
              f"{str(row['exit_date'])[:10]:<12} "
              f"{int(row['duration_days']):>5} "
              f"{row['trade_ret_pct']:>+8.3f}%  {flag}")
    wins   = trades[trades["win"]]
    losses = trades[~trades["win"]]
    print(f"\n  Wins : {len(wins):>3}  avg {wins['trade_ret_pct'].mean():>+.3f}%  "
          f"best  {wins['trade_ret_pct'].max():>+.3f}%")
    print(f"  Losses: {len(losses):>2}  avg {losses['trade_ret_pct'].mean():>+.3f}%  "
          f"worst {losses['trade_ret_pct'].min():>+.3f}%")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RSI(2) 2-year paper test")
    parser.add_argument("--start", default=TWO_YR_AGO,
                        help=f"Start date (default: {TWO_YR_AGO})")
    parser.add_argument("--end",   default=TODAY,
                        help=f"End date   (default: {TODAY})")
    parser.add_argument("--json",  default=RESULTS_JSON,
                        help="Path to rsi2_results.json")
    args = parser.parse_args()

    print("=" * 68)
    print("  RSI(2) PAPER TEST  —  LAST 2 YEARS")
    print("=" * 68)
    print(f"  Period : {args.start}  to  {args.end}")
    print(f"  Params : {args.json}")

    # Load best IS params
    best_params = load_best_params(args.json)
    best_bt, best_sm = _split_params(best_params)
    print(f"\n  Best IS params (fixed — no re-fitting on paper-test period):")
    for k, v in best_params.items():
        print(f"    {k:<15} = {v}")

    # Need some extra history before paper-test start for warmup
    # (SMA-200 needs 200 trading days = ~10 months)
    warmup_start = (datetime.strptime(args.start, "%Y-%m-%d")
                    - relativedelta(months=12)).strftime("%Y-%m-%d")

    print(f"\n  Downloading prices (incl. 12-month warmup from {warmup_start}) ...",
          end=" ", flush=True)
    prices = fetch_prices(TICKERS, warmup_start, args.end)
    if not prices:
        print("ERROR: no price data.")
        return
    print(f"OK ({len(prices)} tickers)")

    # Load T-bill rate into the module-level variable used by backtest_one()
    print(f"  Fetching T-bill rate ...", end=" ", flush=True)
    _strat._CASH_RATE = fetch_cash_rate(warmup_start, args.end)
    if len(_strat._CASH_RATE) > 0:
        avg = float(_strat._CASH_RATE.mean()) * ANNUAL_BARS * 100
        print(f"OK (avg {avg:.2f}% annualised)")
    else:
        print("unavailable")
    print(f"  Fetching VIX ...", end=" ", flush=True)
    _strat._VIX = fetch_vix(warmup_start, args.end)
    if len(_strat._VIX) > 0:
        print(f"OK (avg VIX: {float(_strat._VIX.mean()):.1f})")
    else:
        print("unavailable")

    # Clip to paper-test window
    test_prices = {t: s.loc[args.start:args.end] for t, s in prices.items()}
    # But run backtest on full (warmup + test) so indicators are warm on day 1 of test
    full_prices = prices

    # Per-ticker backtest on full data, then slice returns to paper-test window
    results = {}
    for ticker, close in full_prices.items():
        res = backtest_one(close, **best_bt)
        # Restrict all series to paper-test window
        results[ticker] = {
            k: (v.loc[args.start:args.end] if isinstance(v, pd.Series) else v)
            for k, v in res.items()
        }
        # Re-build equity from sliced returns
        results[ticker]["equity"] = (
            (1.0 + results[ticker]["returns"].fillna(0.0)).cumprod() * 100_000.0
        )
        # Re-extract trades that fall within paper-test window
        trades = res["trades"]
        if len(trades):
            results[ticker]["trades"] = trades[
                (trades["entry_date"] >= args.start) |
                (trades["exit_date"]  >= args.start)
            ].reset_index(drop=True)
        else:
            results[ticker]["trades"] = trades

    # B&H benchmark for same window
    spy_close = test_prices.get("SPY", test_prices[list(test_prices.keys())[0]])
    bah = bah_metrics(spy_close, label="B&H SPY", n_trials=1)

    # Combined portfolio
    comb_ret, comb_eq, comb_trades = combine_returns(
        results, sizing_mode=best_sm, rsi_entry=best_bt["rsi_entry"]
    )
    n_combos = 1  # already selected; no multiple-testing penalty on paper test
    combined = compute_metrics(comb_ret, comb_eq, comb_trades,
                               n_trials=n_combos, label="Combined")

    print("\n" + "=" * 68)
    print("  COMBINED PORTFOLIO (SPY + QQQ + IWM)")
    print("=" * 68)
    print_comparison(combined, bah, label="(combined)")

    print("\n" + "=" * 68)
    print("  PER-TICKER BREAKDOWN")
    print("=" * 68)
    print(f"\n  {'Ticker':<8} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'Win%':>7} "
          f"{'Exp%':>7} {'N':>4}")
    print("  " + "-" * 55)
    for ticker, res in results.items():
        m = compute_metrics(res["returns"], res["equity"], res["trades"],
                            n_trials=1, label=ticker)
        print(f"  {ticker:<8} {pct(m['cagr_pct'])} {flt(m['sharpe'])} "
              f"{pct(m['max_drawdown_pct'])} {m['win_rate_pct']:>6.1f}% "
              f"{m['exposure_pct']:>6.1f}% {m['n_trades']:>4}")

    print("\n" + "=" * 68)
    print("  COMBINED TRADE LOG (paper-test window)")
    print("=" * 68)
    print_trade_log(comb_trades)

    # Year-by-year breakdown inside the 2-year window
    years = sorted({str(d)[:4] for d in comb_ret.index})
    if len(years) > 1:
        print("\n" + "=" * 68)
        print("  YEAR-BY-YEAR BREAKDOWN")
        print("=" * 68)
        print(f"\n  {'Year':<6} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'Win%':>7} {'N':>4}")
        print("  " + "-" * 44)
        for yr in years:
            yr_ret = comb_ret[comb_ret.index.year == int(yr)]
            yr_eq  = (1.0 + yr_ret.fillna(0.0)).cumprod() * 100_000.0
            yr_tr  = comb_trades[comb_trades["entry_date"].astype(str).str[:4] == yr] \
                     if len(comb_trades) else comb_trades
            m = compute_metrics(yr_ret, yr_eq, yr_tr, n_trials=1)
            print(f"  {yr:<6} {pct(m['cagr_pct'])} {flt(m['sharpe'])} "
                  f"{pct(m['max_drawdown_pct'])} {m['win_rate_pct']:>6.1f}% "
                  f"{m['n_trades']:>4}")

    print(f"\n  Sizing mode : {best_sm}")
    print(f"  DSR note    : DSR on paper test = {combined['deflated_sharpe']:.4f} "
          f"(no IS-search penalty; pure OOS result)")

    # ── live news sentiment overlay ──────────────────────────────────────────
    if _HAS_NEWS:
        print("\n" + "=" * 68)
        print("  CURRENT NEWS SENTIMENT (live — today's signal modifier)")
        print("=" * 68)
        nj = NewsJudgment(use_finbert=True)
        nj.report(TICKERS)

        print("  Trading action summary:")
        all_clear = True
        for ticker in TICKERS:
            modifier, label, detail = nj.get_modifier(ticker)
            if label != "NORMAL":
                all_clear = False
            print(f"    {ticker:<6} {label:<7}  modifier={modifier:.2f}  {detail}")
        if all_clear:
            print("    -> All tickers NORMAL: RSI signals active at full size.")
        print()
    else:
        print()



if __name__ == "__main__":
    main()
