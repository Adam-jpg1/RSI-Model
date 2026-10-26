# RSI-Model

ML-powered trading system: XGBoost signal classifier + Alpaca live bot + FinBERT news sentiment.

## Setup

```bash
pip install -r requirements.txt
```

For live/paper trading, set environment variables:
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```
Get free paper trading keys at [alpaca.markets](https://alpaca.markets).

## Usage

```bash
# 1. Train the model (fetches 180d of 5m bars + news sentiment)
python main.py train --tickers AAPL MSFT NVDA

# 2. Backtest on held-out data
python main.py backtest --tickers AAPL --plot

# 3. Check news sentiment
python main.py sentiment --tickers AAPL NVDA

# 4. Start live/paper bot (requires Alpaca keys)
python main.py trade --tickers AAPL MSFT
```

## Architecture

```
data/
  fetcher.py       — yfinance price data + news
  indicators.py    — 30+ indicators: RSI, MACD, BB, ATR, Stoch, CCI, OBV, VWAP
sentiment/
  analyzer.py      — FinBERT news scorer with recency decay weighting
ml/
  features.py      — feature matrix + time-series CV split
  model.py         — XGBoost classifier (DOWN / FLAT / UP)
  trainer.py       — training pipeline with cross-validation
bot/
  alpaca_client.py — Alpaca paper/live order execution
  strategy.py      — signal + Kelly-based position sizing
  trader.py        — live polling loop with stop-loss / take-profit
backtest/
  engine.py        — P&L engine: Sharpe, max drawdown, win rate, profit factor
config.py          — all parameters in one place
main.py            — CLI entry point
```

## Features fed to XGBoost (~31 inputs)
- RSI at 3 periods (7, 14, 21) + slope
- MACD line, signal, histogram + slope
- Bollinger Bands %B + bandwidth
- ATR % of price
- Stochastic K & D
- CCI, Williams %R
- OBV / OBV-SMA ratio
- Price vs VWAP
- Volume ratio (vs 20-bar avg)
- Price momentum (1, 3, 5, 10 bars)
- Rolling volatility (10, 20 bars)
- EMA & SMA crossovers
- **FinBERT news sentiment score** (time-decay weighted)
