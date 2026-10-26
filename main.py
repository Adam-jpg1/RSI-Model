import argparse
import os
import sys

from config import CONFIG


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

    model = RSIModel()
    model.load()

    tickers = args.tickers or CONFIG.tickers
    for ticker in tickers:
        headlines = get_stock_news(ticker)
        sentiment = analyze_headlines(headlines)
        result = backtest(
            ticker, model,
            period=args.period,
            interval=args.interval,
            sentiment_score=sentiment["score"],
        )
        print_backtest_report(result)

        if args.plot:
            _plot_equity(result, ticker)


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
            m.load()
            models[t] = m
        except FileNotFoundError:
            print(f"[warn] No model for {t} — train first with: python main.py train --tickers {t}")

    if not models:
        print("No trained models found. Run: python main.py train")
        sys.exit(1)

    trader = LiveTrader(models, tickers=list(models.keys()))
    trader.run()


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


def main():
    parser = argparse.ArgumentParser(description="RSI-Model: ML trading system")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Train XGBoost model on historical data")
    train_p.add_argument("--tickers", nargs="+", default=None)
    train_p.add_argument("--period", default="180d")
    train_p.add_argument("--interval", default="5m")

    bt_p = sub.add_parser("backtest", help="Run backtest on trained model")
    bt_p.add_argument("--tickers", nargs="+", default=None)
    bt_p.add_argument("--period", default="60d")
    bt_p.add_argument("--interval", default="5m")
    bt_p.add_argument("--plot", action="store_true")

    trade_p = sub.add_parser("trade", help="Start live Alpaca paper/live bot")
    trade_p.add_argument("--tickers", nargs="+", default=None)

    sent_p = sub.add_parser("sentiment", help="Show news sentiment for tickers")
    sent_p.add_argument("--tickers", nargs="+", default=None)

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "trade":
        cmd_trade(args)
    elif args.command == "sentiment":
        cmd_sentiment(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
