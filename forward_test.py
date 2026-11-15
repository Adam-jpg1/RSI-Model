"""
True out-of-sample forward test.

Splits each ticker's 2yr history in half: trains on year 1, tests on year 2.
This answers "what would have happened if we deployed the model a year ago?"

Unlike batch_eval.py (which tests models already trained on all data),
this trains a fresh model per ticker on only the first half, so the test
period is genuinely unseen data.

Usage:
  python forward_test.py                        # all trained tickers
  python forward_test.py --tickers AAPL AMD MU  # specific tickers
  python forward_test.py --top 15               # top N by return
"""
import sys
import os
import argparse
import math
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from data.fetcher import get_price_data, get_daily_data
from data.indicators import add_all_indicators, add_daily_trend_features
from ml.features import get_feature_matrix, build_target, target_bars_for_interval
from ml.model import RSIModel
from backtest.engine import _run_simulation
from data.universe import get_trained_tickers
from config import CONFIG

INTERVAL = "1h"
FETCH_PERIOD = "2y"   # always fetch 2yr so test window is a full year
MIN_TRAIN_BARS = 200
MIN_TEST_BARS = 50


def forward_test_ticker(ticker: str) -> dict:
    """
    Train on the first half of 2yr data, test on the second half.
    The train half approximates the config.data.train_period (9m) deployment setup.
    """
    df = get_price_data(ticker, period=FETCH_PERIOD, interval=INTERVAL)
    df = add_all_indicators(df)
    try:
        daily_df = get_daily_data(ticker, period=FETCH_PERIOD)
        df = add_daily_trend_features(df, daily_df)
    except Exception:
        pass

    features = get_feature_matrix(df, sentiment_score=0.0).dropna()
    df_aligned = df.loc[features.index]

    n = len(features)
    split = n // 2

    if split < MIN_TRAIN_BARS:
        raise ValueError(f"Not enough training bars: {split} (need {MIN_TRAIN_BARS})")
    if (n - split) < MIN_TEST_BARS:
        raise ValueError(f"Not enough test bars: {n - split} (need {MIN_TEST_BARS})")

    # Train on first half only
    n_bars = target_bars_for_interval(INTERVAL)
    X_train = features.iloc[:split]
    y_train = build_target(df_aligned.iloc[:split], n_bars=n_bars)
    mask = X_train.notna().all(axis=1) & y_train.notna()
    X_train, y_train = X_train[mask], y_train[mask]

    if len(X_train) < MIN_TRAIN_BARS:
        raise ValueError(f"Too few clean training rows: {len(X_train)}")

    model = RSIModel()
    model.build()
    model.fit(X_train, y_train)

    # Test on second half — the model has never seen this data
    X_test = features.iloc[split:]
    df_test = df_aligned.iloc[split:]

    per_ticker_adx = CONFIG.bot.per_ticker_adx.get(ticker)
    adx_min = per_ticker_adx if per_ticker_adx is not None else CONFIG.bot.adx_filter_min

    result = _run_simulation(
        df_test, X_test, model,
        initial_capital=100_000.0,
        ticker=ticker,
        adx_filter_min=adx_min,
        allow_short=CONFIG.bot.allow_short,
    )

    test_start = df_test.index[0].strftime("%Y-%m-%d") if len(df_test) else "?"
    test_end = df_test.index[-1].strftime("%Y-%m-%d") if len(df_test) else "?"
    result["test_window"] = f"{test_start} to {test_end}"
    result["test_bars"] = len(X_test)
    return result


def main():
    parser = argparse.ArgumentParser(description="True out-of-sample forward test")
    parser.add_argument("--tickers", nargs="+", help="Tickers to test (default: all trained)")
    parser.add_argument("--top", type=int, default=0, help="Show only top N results")
    args = parser.parse_args()

    tickers = args.tickers or sorted(get_trained_tickers())

    print(f"\nForward test: train on year 1, test on year 2")
    print(f"Models trained fresh on first-half data only — test period is unseen.")
    print(f"Testing {len(tickers)} tickers...\n")

    rows = []
    failed = []

    for ticker in tickers:
        try:
            r = forward_test_ticker(ticker)
            m = r["metrics"]
            rows.append({
                "ticker": ticker,
                "return": r["total_return_pct"],
                "win_rate": m["win_rate_pct"],
                "sharpe": m["sharpe_ratio"],
                "pf": m["profit_factor"],
                "trades": m["total_trades"],
                "dd": m["max_drawdown_pct"],
                "window": r["test_window"],
            })
            print(f"  {ticker:<6} {r['total_return_pct']:>+7.2f}%  "
                  f"win={m['win_rate_pct']:.1f}%  "
                  f"trades={m['total_trades']}  "
                  f"[{r['test_window']}]")
        except Exception as e:
            failed.append(ticker)
            print(f"  {ticker}: SKIP — {e}")

    if not rows:
        print("No results.")
        return

    rows.sort(key=lambda x: x["return"], reverse=True)

    if args.top:
        rows = rows[:args.top]

    profitable = [r for r in rows if r["return"] > 0 and r["trades"] >= 10]
    avg_all = sum(r["return"] for r in rows) / len(rows)
    avg_prof = sum(r["return"] for r in profitable) / len(profitable) if profitable else 0.0
    above_20 = sum(1 for r in rows if r["return"] >= 20)
    above_10 = sum(1 for r in rows if r["return"] >= 10)

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {'TICKER':<7} {'RETURN':>8} {'WIN%':>6} {'SHARPE':>7} {'PF':>6} {'TRADES':>7} {'DD%':>6}")
    print(sep)
    for r in rows:
        marker = " <--" if r["return"] >= 20 else ""
        print(f"  {r['ticker']:<7} {r['return']:>+7.2f}%  {r['win_rate']:>5.1f}%  "
              f"{r['sharpe']:>6.2f}  {r['pf']:>5.2f}  {r['trades']:>6}  "
              f"{r['dd']:>5.1f}%{marker}")
    print(sep)

    print(f"\n  Results ({len(rows)} tickers):")
    print(f"  Average return (all)        : {avg_all:+.1f}%")
    print(f"  Average return (profitable) : {avg_prof:+.1f}%")
    print(f"  Profitable tickers (10+ tr) : {len(profitable)}/{len(rows)}")
    print(f"  Tickers at 20%+ return      : {above_20}")
    print(f"  Tickers at 10%+ return      : {above_10}")

    if avg_all >= 20:
        print(f"\n  STATUS: STRONG — {avg_all:+.1f}% out-of-sample average")
    elif avg_all >= 12:
        print(f"\n  STATUS: SOLID — {avg_all:+.1f}% out-of-sample average")
    elif avg_all >= 5:
        print(f"\n  STATUS: MODEST — {avg_all:+.1f}% out-of-sample average")
    else:
        print(f"\n  STATUS: WEAK — {avg_all:+.1f}% out-of-sample (investigate before deploying)")

    if failed:
        print(f"\n  Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
