# RSI-Model

Autonomous quantitative trading bot. Trains an XGBoost signal classifier per ticker, optimises hyperparameters with Optuna, and runs a continuous live/paper trading loop via the Alpaca API.

Built for high-momentum tech equities (APP, AMD, GOOGL, SNOW, MRVL, TSLA) but generalises to any liquid ticker.

---

## Requirements

- Python 3.10+
- Alpaca account (free paper trading at [alpaca.markets](https://alpaca.markets))
- Internet access (yfinance for price data)
- PyTorch — optional, for FinBERT news sentiment (degrades gracefully if absent)

---

## Installation

```bash
git clone https://github.com/Adam-jpg1/RSI-Model-old.git
cd RSI-Model-old

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Alpaca keys:

```
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Quick start

```bash
# Pre-flight check — verifies models, data, Alpaca connection
python preflight.py

# Start paper trading on the validated portfolio
python main.py trade --tickers APP AMD GOOGL SNOW MRVL TSLA

# Autonomous mode — scans full universe, discovers opportunities automatically
python main.py run --dashboard
```

---

## Commands

| Command | Description |
|---------|-------------|
| `python main.py train --tickers APP AMD` | Train models using Optuna-tuned hyperparameters |
| `python main.py optimize --tickers APP --trials 100` | Bayesian hyperparameter search, saves to `models/params_*.json` |
| `python main.py trade --tickers APP AMD GOOGL SNOW MRVL TSLA` | Manual mode — trade a fixed ticker list |
| `python main.py run` | Autonomous mode — scan universe, discover + trade opportunities |
| `python main.py run --dashboard` | Autonomous mode with live terminal dashboard |
| `python main.py screen --top-n 20` | Scan universe, rank tickers by signal strength |
| `python main.py backtest --tickers APP` | Held-out backtest on last 15% of data |
| `python main.py simulate --tickers APP --folds 4` | Walk-forward validation with bootstrap confidence intervals |
| `python main.py sentiment --tickers APP AMD` | Show FinBERT sentiment for recent headlines |
| `python preflight.py` | Pre-flight system check before going live |

---

## Architecture

```
RSI-Model/
├── config.py               All parameters: data, ML, bot, universe, screening
├── main.py                 CLI entry point
│
├── data/
│   ├── fetcher.py          yfinance price/news/earnings data with 24h caching
│   ├── indicators.py       50+ indicators + PEAD/earnings/regime/microstructure features
│   └── universe.py         S&P 500 + NASDAQ-100 universe; 24h JSON cache
│
├── ml/
│   ├── features.py         174-feature matrix + percentile label assignment
│   ├── model.py            XGBoost 3-class (UP/FLAT/DOWN) + RobustScaler + ensemble
│   ├── trainer.py          5-fold CV → final model; auto-loads Optuna params
│   └── optimizer.py        Optuna Bayesian search (100 trials); saves params_*.json
│
├── backtest/
│   ├── engine.py           Long/short simulation; Calmar, Sharpe, drawdown, win rate
│   └── simulate.py         Walk-forward + bootstrap confidence intervals
│
├── bot/
│   ├── alpaca_client.py    Alpaca SDK wrapper (paper + live)
│   ├── strategy.py         Signal generation + time-of-day confidence adjustment
│   └── trader.py           LiveTrader: poll loop, position management, auto-retraining
│
├── portfolio/
│   └── manager.py          Risk limits: exposure, concentration, Kelly sizing
│
├── screening/
│   └── scanner.py          Parallel universe scan; ranks by signal strength
│
├── dashboard/
│   └── terminal.py         Rich live terminal: account / positions / signals
│
├── sentiment/
│   └── analyzer.py         FinBERT pipeline; recency-decay weighting
│
└── models/                 Trained XGBoost models + Optuna params (per ticker)
```

---

## Feature set (174 features)

| Group | Description |
|-------|-------------|
| Technical | RSI (7/14/21), MACD, Bollinger Bands, ATR, Stochastic, CCI, Williams %R, OBV, ADX |
| Price | Returns (1/3/5/10 bars), volatility (10/20), VWAP, candle structure |
| Multi-timeframe | Daily RSI, MACD, ADX, SMA50/200, trend score; weekly context |
| Cross-asset | SPY, NVDA, XLK, VIX hourly; TLT/HYG/GLD/UUP daily macro regime |
| Earnings | Days to/since earnings, pre-earnings countdown (IC 0.17–0.37); PEAD drift phases (days 7–20, IC 0.19–0.23) |
| Seasonality | Circular week/month encoding, OPEX (3rd Friday), FOMC/NFP calendar |
| Microstructure | Buying pressure, bar efficiency, wick imbalance, tick imbalance Z-score |
| Regime | Trending/choppy classification; regime × signal interaction features |
| Cross-sectional | 6-ticker relative strength ranks; lead-lag returns across portfolio |
| Lags | t-1/t-2/t-3 of key indicators for sequential context |

---

## Validated results

Walk-forward, 4 folds, 2 years of hourly data, long-only:

| Ticker | CAGR | Max DD | Win Rate | Trades | P(profitable) |
|--------|------|--------|----------|--------|---------------|
| APP    | +7.0% | -3.5% | 56.9%   | 146    | 99% |
| AMD    | +7.2% | -2.6% | 51.8%   | 289    | 99% |
| GOOGL  | +3.1% | -1.2% | 55.2%   | 181    | 99% |
| SNOW   | +5.0% | -1.6% | 52.5%   | 55     | 99% |
| MRVL   | +3.3% | -1.7% | 47.5%   | 74     | 97% |
| TSLA   | +2.3% | -1.8% | 52.2%   | 31     | 94% |
| **AVG** | **+4.7%** | | | **776 total** | |

---

## Risk disclosures

- **Paper trading by default.** Set `paper_trading = False` in `config.py` only with a funded Alpaca account.
- **Past performance is not indicative of future results.** Models are trained on 2024–2026 market conditions.
- **No guarantee of profitability.** This is a research tool, not financial advice.
- **Short selling** is enabled by default but sized at 80% of normal. Shorts carry theoretically unlimited loss.
