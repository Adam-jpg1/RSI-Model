import argparse
import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from config import CONFIG

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)


# ------------------------------------------------------------------ #
# Command handlers                                                    #
# ------------------------------------------------------------------ #

def cmd_train(args):
    from ml.trainer import train_on_ticker, train_multi_ticker
    tickers = args.tickers or CONFIG.tickers
    if len(tickers) == 1:
        train_on_ticker(tickers[0], period=args.period, interval=args.interval)
    else:
        train_multi_ticker(tickers)


def cmd_backtest(args):
    from ml.model import RSIModel
    from backtest.engine import backtest, print_backtest_report
    from data.fetcher import get_stock_news
    from sentiment.analyzer import analyze_headlines

    tickers = args.tickers or CONFIG.tickers
    for ticker in tickers:
        model = RSIModel()
        model.load(ticker=ticker)
        try:
            headlines = get_stock_news(ticker)
            sentiment = analyze_headlines(headlines)
        except Exception:
            sentiment = {"score": 0.0}
        result = backtest(
            ticker, model,
            period=args.period,
            interval=args.interval,
            sentiment_score=sentiment["score"],
            skip_pct=args.skip_pct,
            allow_short=args.short,
        )
        print_backtest_report(result)
        if args.plot:
            _plot_equity(result, ticker)


def cmd_simulate(args):
    from backtest.simulate import (
        walk_forward_backtest, bootstrap_confidence,
        print_simulation_report, run_adx_comparison,
    )
    tickers = args.tickers or CONFIG.tickers
    interval = getattr(args, "interval", "1h")

    if args.adx_compare:
        for ticker in tickers:
            run_adx_comparison(ticker)
        return

    for ticker in tickers:
        wf = walk_forward_backtest(ticker, n_folds=args.folds, adx_filter_min=args.adx_min,
                                   interval=interval)
        boot = bootstrap_confidence(wf["combined_pnl"])
        print_simulation_report(ticker, wf, boot)


def cmd_trade(args):
    from ml.model import RSIModel
    from bot.trader import LiveTrader

    if not CONFIG.bot.alpaca_api_key:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        sys.exit(1)

    tickers = args.tickers or CONFIG.tickers
    models = {}
    for t in tickers:
        m = RSIModel()
        try:
            m.load(ticker=t)
            models[t] = m
        except FileNotFoundError:
            print(f"[warn] No model for {t} — train first: python main.py train --tickers {t}")

    if not models:
        print("No trained models found. Run: python main.py train")
        sys.exit(1)

    trader = LiveTrader(models, tickers=list(models.keys()))
    trader.run()


def cmd_run(args):
    """Autonomous mode: discovers and trades across the full trained universe."""
    if not CONFIG.bot.alpaca_api_key:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        sys.exit(1)

    from bot.trader import LiveTrader
    trader = LiveTrader.load_all_trained()

    if not trader.models:
        print("No trained models found. Run: python main.py train --tickers AAPL MSFT NVDA ...")
        sys.exit(1)

    if args.dashboard:
        import threading
        from bot.alpaca_client import AlpacaClient
        from dashboard.terminal import Dashboard
        from screening.scanner import scan_universe

        client = AlpacaClient()
        dash = Dashboard(client=client, scanner_fn=scan_universe, scan_interval_seconds=300)
        t = threading.Thread(target=dash.run_live, kwargs={"refresh_seconds": 30}, daemon=True)
        t.start()

    trader.run_autonomous()


def cmd_universe(args):
    from data.universe import get_universe, get_trained_tickers

    universe = get_universe(force_refresh=args.refresh)
    trained = set(get_trained_tickers())
    trained_in_universe = [t for t in universe if t in trained]
    untrained = [t for t in universe if t not in trained]

    print(f"\nUniverse size       : {len(universe)}")
    print(f"Trained models      : {len(trained_in_universe)}")
    print(f"Untrained tickers   : {len(untrained)}")

    if args.list_trained:
        print("\nTrained:")
        for t in sorted(trained_in_universe):
            print(f"  {t}")

    if args.list_untrained:
        print(f"\nUntrained (first 50):")
        for t in untrained[:50]:
            print(f"  {t}")


def cmd_screen(args):
    from screening.scanner import scan_universe, get_untrained_tickers
    from data.universe import get_universe

    opps = scan_universe(
        tickers=args.tickers or None,
        max_workers=args.workers,
        top_n=args.top_n,
    )

    print(f"\n{'='*72}")
    print(f"  {'TICKER':<8} {'SIGNAL':<7} {'CONF':>6}  {'ADX':>6}  {'RSI':>6}  {'VOLR':>6}  {'PRICE':>10}")
    print(f"{'='*72}")
    for opp in opps:
        sig = opp["signal"]
        marker = "▲" if sig == "UP" else ("▼" if sig == "DOWN" else "–")
        print(f"  {opp['ticker']:<8} {marker}{sig:<6} {opp['confidence']:>6.3f}  "
              f"{opp['adx']:>6.1f}  {opp['rsi']:>6.1f}  "
              f"{opp.get('vol_ratio', 1.0):>6.2f}  "
              f"${opp['price']:>9.2f}")
    print(f"{'='*72}")

    if args.train_top:
        from ml.trainer import train_on_ticker
        universe = get_universe()
        untrained = get_untrained_tickers(universe)
        candidates = untrained[:args.train_top]
        print(f"\n[screen] Training models for top {len(candidates)} untrained candidates: {candidates}")
        for ticker in candidates:
            try:
                train_on_ticker(ticker)
            except Exception as e:
                print(f"  {ticker} failed: {e}")


def cmd_optimize(args):
    from ml.optimizer import optimize_ticker

    tickers = args.tickers or CONFIG.tickers
    interval = getattr(args, "interval", "1h")
    daily_confirm = getattr(args, "daily_confirm", False)
    allow_short = getattr(args, "allow_short", None)
    for ticker in tickers:
        try:
            optimize_ticker(
                ticker,
                n_trials=args.trials,
                n_folds=args.folds,
                period=args.period,
                interval=interval,
                daily_confirm=daily_confirm,
                allow_short=allow_short,
            )
        except Exception as e:
            print(f"[optuna] {ticker} failed: {e}")

    if args.retrain:
        from ml.trainer import train_on_ticker
        print("\n[optimize] Retraining with best params...")
        for ticker in tickers:
            try:
                train_on_ticker(ticker, interval=interval)
            except Exception as e:
                print(f"  {ticker} retrain failed: {e}")


def cmd_sentiment(args):
    from data.fetcher import get_stock_news, get_company_info
    from sentiment.analyzer import analyze_headlines

    tickers = args.tickers or CONFIG.tickers
    for ticker in tickers:
        info = get_company_info(ticker)
        headlines = get_stock_news(ticker)
        result = analyze_headlines(headlines)

        print(f"\n{'='*55}")
        print(f"  {ticker} — {info['name']}")
        print(f"  Price: ${info.get('current_price', 'N/A')}  |  Sector: {info.get('sector', 'N/A')}")
        print(f"  Sentiment: {result['label'].upper()} ({result['score']:+.3f})  |  Articles: {result['n_articles']}")
        print(f"{'='*55}")
        for a in result["articles"][:5]:
            icon = "+" if a["sentiment"] == "positive" else ("-" if a["sentiment"] == "negative" else "~")
            print(f"  [{icon}] {a['title'][:70]}")
            print(f"       conf={a['confidence']:.2f}  age={a['age_hours']:.1f}h  ({a['provider']})")


def cmd_dashboard(args):
    from bot.alpaca_client import AlpacaClient
    from dashboard.terminal import Dashboard
    from screening.scanner import scan_universe

    if not CONFIG.bot.alpaca_api_key:
        client = None
        print("[dashboard] No Alpaca keys — showing model status only")
    else:
        client = AlpacaClient()

    dash = Dashboard(client=client, scanner_fn=scan_universe, scan_interval_seconds=300)
    dash.run_live(refresh_seconds=args.refresh)


# ------------------------------------------------------------------ #
# Utilities                                                           #
# ------------------------------------------------------------------ #

def _plot_equity(result, ticker: str):
    try:
        import matplotlib.pyplot as plt
        eq = result["equity_curve"]["equity"]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(eq.index, eq.values, label="Equity")
        ax.set_title(f"{ticker} Backtest Equity Curve")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"backtest_{ticker}.png", dpi=150)
        print(f"  Equity curve saved to backtest_{ticker}.png")
        plt.close()
    except ImportError:
        print("  (pip install matplotlib to enable equity plots)")


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="RSI-Model: autonomous ML trading system")
    sub = parser.add_subparsers(dest="command")

    # train
    p = sub.add_parser("train", help="Train XGBoost model (auto-uses Optuna params if present)")
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--period", default=None, help="Data period (default: config.data.train_period = 9m)")
    p.add_argument("--interval", default="1h")

    # backtest
    p = sub.add_parser("backtest", help="Backtest a trained model")
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--period", default="2y")
    p.add_argument("--interval", default="1h")
    p.add_argument("--skip-pct", type=float, default=0.85, dest="skip_pct")
    p.add_argument("--short", action="store_true", help="Enable short selling in backtest")
    p.add_argument("--plot", action="store_true")

    # simulate
    p = sub.add_parser("simulate", help="Walk-forward validation + bootstrap confidence")
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--adx-min", type=float, default=None, dest="adx_min",
                   help="ADX threshold (default: auto-load from Optuna params, or 0 to disable)")
    p.add_argument("--adx-compare", action="store_true", dest="adx_compare")
    p.add_argument("--interval", default="1h", help="Bar interval (1h, 15m …)")

    # trade (manual ticker list)
    p = sub.add_parser("trade", help="Start Alpaca paper/live bot (manual ticker list)")
    p.add_argument("--tickers", nargs="+", default=None)

    # run (autonomous mode)
    p = sub.add_parser("run", help="Autonomous trading across full trained universe")
    p.add_argument("--dashboard", action="store_true", help="Show live terminal dashboard alongside")

    # universe
    p = sub.add_parser("universe", help="Manage the S&P500 + NASDAQ-100 ticker universe")
    p.add_argument("--refresh", action="store_true", help="Force re-fetch from Wikipedia")
    p.add_argument("--list-trained", action="store_true", dest="list_trained")
    p.add_argument("--list-untrained", action="store_true", dest="list_untrained")

    # screen
    p = sub.add_parser("screen", help="Scan universe for ML signals")
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--top-n", type=int, default=20, dest="top_n")
    p.add_argument("--workers", type=int, default=CONFIG.screening.max_scan_workers)
    p.add_argument("--train-top", type=int, default=0, dest="train_top",
                   help="Train models for top N untrained candidates after scanning")

    # optimize
    p = sub.add_parser("optimize", help="Optuna hyperparameter search per ticker")
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--period", default="2y")
    p.add_argument("--interval", default="1h", help="Bar interval (1h, 15m, 5m …)")
    p.add_argument("--retrain", action="store_true", help="Retrain immediately after optimization")
    p.add_argument("--daily-confirm", action="store_true", dest="daily_confirm",
                   help="Optimize for daily_confirm gate (saves params_{ticker}_dc.json)")
    p.add_argument("--long-only", action="store_true", dest="allow_short",
                   help="Optimize long-only (no shorts)")

    # sentiment
    p = sub.add_parser("sentiment", help="Show news sentiment for tickers")
    p.add_argument("--tickers", nargs="+", default=None)

    # dashboard
    p = sub.add_parser("dashboard", help="Live Rich terminal dashboard")
    p.add_argument("--refresh", type=int, default=30, help="Refresh interval in seconds")

    args = parser.parse_args()

    dispatch = {
        "train": cmd_train,
        "backtest": cmd_backtest,
        "simulate": cmd_simulate,
        "trade": cmd_trade,
        "run": cmd_run,
        "universe": cmd_universe,
        "screen": cmd_screen,
        "optimize": cmd_optimize,
        "sentiment": cmd_sentiment,
        "dashboard": cmd_dashboard,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
