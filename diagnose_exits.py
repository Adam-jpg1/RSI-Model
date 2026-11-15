"""
Diagnose exit reason breakdown to find where we're losing edge.
Shows stop/take/signal exit split, and avg PnL per exit type.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from config import CONFIG
from backtest.simulate import walk_forward_backtest

TICKERS = ["AAPL", "ABBV", "AMD", "ANET", "APP", "COIN", "MSFT", "NVDA", "T", "TSLA"]

def diagnose(ticker):
    result = walk_forward_backtest(ticker, n_folds=4)
    all_trades = []
    for fold in result["fold_results"]:
        t = fold["trades"]
        if len(t):
            closed = t[t["type"].isin(["SELL", "COVER"])]
            if len(closed):
                all_trades.append(closed)
    if not all_trades:
        return None
    df = pd.concat(all_trades)
    total = len(df)
    if total == 0:
        return None

    by_reason = df.groupby("exit_reason").agg(
        count=("pnl_pct", "count"),
        avg_pnl=("pnl_pct", "mean"),
        win_rate=("pnl_pct", lambda x: (x > 0).mean() * 100),
    ).round(3)

    wins = df[df["pnl_pct"] > 0]
    losses = df[df["pnl_pct"] <= 0]
    overall_wr = len(wins) / total * 100
    avg_win = wins["pnl_pct"].mean() if len(wins) else 0
    avg_loss = losses["pnl_pct"].mean() if len(losses) else 0
    expectancy = (overall_wr/100 * avg_win) + ((1 - overall_wr/100) * avg_loss)

    print(f"\n{'='*58}")
    print(f"  {ticker}  ({total} trades total)")
    print(f"  Overall: win={overall_wr:.1f}%  avg_win={avg_win:+.2f}%  avg_loss={avg_loss:+.2f}%")
    print(f"  Expectancy per trade: {expectancy:+.3f}%")
    print(f"{'='*58}")
    print(f"  {'EXIT':>8}  {'COUNT':>6}  {'%':>5}  {'AVG_PNL':>9}  {'WIN%':>6}")
    for reason, row in by_reason.iterrows():
        pct_of_trades = row["count"] / total * 100
        print(f"  {reason:>8}  {int(row['count']):>6}  {pct_of_trades:>4.0f}%  "
              f"{row['avg_pnl']:>+8.3f}%  {row['win_rate']:>5.1f}%")
    return by_reason

print("Diagnosing exit reasons across tickers...\n")
summaries = []
for t in TICKERS:
    try:
        diagnose(t)
    except Exception as e:
        print(f"  {t}: ERROR - {e}")
