import pandas as pd
import numpy as np
from typing import Dict, Optional
from data.fetcher import get_price_data, get_stock_news
from data.indicators import add_all_indicators
from sentiment.analyzer import analyze_headlines
from ml.features import get_feature_matrix
from ml.model import RSIModel
from config import CONFIG


def kelly_fraction(win_prob: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 0.0
    odds = avg_win / avg_loss
    f = (odds * win_prob - (1 - win_prob)) / odds
    return float(np.clip(f * CONFIG.bot.kelly_fraction, 0.0, CONFIG.bot.max_position_pct))


def backtest(
    ticker: str,
    model: RSIModel,
    period: str = "60d",
    interval: str = "5m",
    initial_capital: float = 100_000.0,
    sentiment_score: float = 0.0,
) -> Dict:
    df = get_price_data(ticker, period=period, interval=interval)
    df = add_all_indicators(df)
    features = get_feature_matrix(df, sentiment_score)

    capital = initial_capital
    position = 0.0       # shares held
    entry_price = 0.0
    trades = []
    equity_curve = []

    feature_rows = features.dropna()

    for i, (idx, row) in enumerate(feature_rows.iterrows()):
        close = df.loc[idx, "Close"]
        equity_curve.append({"time": idx, "equity": capital + position * close})

        if not model.is_trained:
            continue

        prediction = model.predict_single(row)
        signal = prediction["signal"]
        confidence = prediction["confidence"]

        if confidence < CONFIG.bot.min_confidence:
            continue

        if position > 0:
            pnl_pct = (close - entry_price) / entry_price
            stop = -CONFIG.bot.stop_loss_pct
            take = CONFIG.bot.take_profit_pct
            if pnl_pct <= stop or pnl_pct >= take or signal in ("DOWN", "FLAT"):
                pnl = position * (close - entry_price)
                capital += position * close
                trades.append({
                    "type": "SELL", "price": close, "shares": position,
                    "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct * 100, 2),
                    "exit_reason": "stop" if pnl_pct <= stop else ("take" if pnl_pct >= take else "signal"),
                    "time": idx,
                })
                position = 0.0
                entry_price = 0.0

        if position == 0 and signal == "UP" and confidence >= CONFIG.bot.min_confidence:
            size_pct = min(confidence * CONFIG.bot.max_position_pct, CONFIG.bot.max_position_pct)
            alloc = capital * size_pct
            shares = alloc / close
            if shares * close <= capital:
                position = shares
                entry_price = close
                capital -= shares * close
                trades.append({
                    "type": "BUY", "price": close, "shares": round(shares, 4),
                    "confidence": round(confidence, 3), "time": idx,
                })

    equity_df = pd.DataFrame(equity_curve).set_index("time")
    final_equity = equity_df["equity"].iloc[-1] if len(equity_df) else initial_capital
    total_return = (final_equity / initial_capital - 1.0) * 100

    trades_df = pd.DataFrame(trades)
    sell_trades = trades_df[trades_df["type"] == "SELL"] if len(trades_df) else pd.DataFrame()

    metrics = _compute_metrics(equity_df["equity"], initial_capital, sell_trades, total_return)
    return {
        "ticker": ticker,
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return, 2),
        "metrics": metrics,
        "trades": trades_df,
        "equity_curve": equity_df,
    }


def _compute_metrics(equity: pd.Series, initial: float, sell_trades: pd.DataFrame, total_return: float) -> Dict:
    returns = equity.pct_change().dropna()
    periods_per_year = 252 * 78  # ~78 5-min bars per trading day
    sharpe = 0.0
    if returns.std() > 0:
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = float(drawdown.min()) * 100

    win_rate, avg_win, avg_loss = 0.0, 0.0, 0.0
    if len(sell_trades):
        wins = sell_trades[sell_trades["pnl"] > 0]
        losses = sell_trades[sell_trades["pnl"] <= 0]
        win_rate = len(wins) / len(sell_trades) * 100
        avg_win = float(wins["pnl_pct"].mean()) if len(wins) else 0.0
        avg_loss = float(losses["pnl_pct"].mean()) if len(losses) else 0.0

    profit_factor = 0.0
    if len(sell_trades) and avg_loss != 0:
        gross_profit = sell_trades[sell_trades["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(sell_trades[sell_trades["pnl"] <= 0]["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": len(sell_trades),
        "win_rate_pct": round(win_rate, 2),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "profit_factor": round(profit_factor, 3),
    }


def print_backtest_report(result: Dict):
    m = result["metrics"]
    print(f"\n{'='*50}")
    print(f"  BACKTEST: {result['ticker']}")
    print(f"{'='*50}")
    print(f"  Total return  : {result['total_return_pct']:+.2f}%")
    print(f"  Final equity  : ${result['final_equity']:,.2f}")
    print(f"  Sharpe ratio  : {m['sharpe_ratio']:.3f}")
    print(f"  Max drawdown  : {m['max_drawdown_pct']:.2f}%")
    print(f"  Total trades  : {m['total_trades']}")
    print(f"  Win rate      : {m['win_rate_pct']:.1f}%")
    print(f"  Avg win       : {m['avg_win_pct']:+.3f}%")
    print(f"  Avg loss      : {m['avg_loss_pct']:+.3f}%")
    print(f"  Profit factor : {m['profit_factor']:.3f}")
    print(f"{'='*50}")
