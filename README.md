# RSI-Model

Autonomous quantitative trading research platform built on XGBoost, FinBERT news sentiment, and the Alpaca paper-trading API.

Covers the S&P 500 and NASDAQ-100. Trains an independent model per ticker, optimises hyperparameters with Optuna, and runs a continuous paper-trading loop that discovers opportunities, manages positions, and retrains models automatically.

---

## Table of contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Alpaca paper-trading setup](#alpaca-paper-trading-setup)
4. [Quick start](#quick-start)
5. [Commands](#commands)
6. [Architecture](#architecture)
7. [Feature set](#feature-set)
8. [Validated results](#validated-results)
9. [Risk disclosures](#risk-disclosures)

---

## Requirements

- Python 3.10+
- Internet access (yfinance, Wikipedia for universe, Alpaca API)
- Alpaca paper-trading account (free) for `trade` / `run` commands
- PyTorch for FinBERT sentiment (optional — degrades gracefully if absent)

---

## Installation

```bash
git clone <repo-url>
cd RSI-Model

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Alpaca paper-trading setup

1. Create a free account at [alpaca.markets](https://alpaca.markets)
2. Navigate to **Paper Trading → API Keys** and generate a key pair
3. Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

4. Load the variables before running trading commands:

```bash
# Windows (PowerShell)
$env:ALPACA_API_KEY  = "PKxxx..."
$env:ALPACA_SECRET_KEY = "xxx..."

# macOS / Linux
export ALPACA_API_KEY="PKxxx..."
export ALPACA_SECRET_KEY="xxx..."
```

> **Never commit `.env` to git.** It is listed in `.gitignore`. Only `.env.example` (which contains no real values) is tracked.

---

## Quick start

```bash
# 1. Train models for the three pre-validated tickers
python main.py train --tickers AAPL MSFT NVDA

# 2. Tune hyperparameters with Optuna, then retrain immediately
python main.py optimize --tickers AAPL MSFT NVDA --trials 60 --retrain

# 3. Walk-forward validation + bootstrap confidence intervals
python main.py simulate --tickers AAPL --folds 4 --adx-min 20

# 4. Scan the full universe for signals (requires trained models)
python main.py screen --top-n 20

# 5. Start autonomous paper trading
python main.py run --dashboard
```

---

## Commands

### `train`
Train an XGBoost model for one or more tickers. Automatically loads Optuna-tuned hyperparameters from `models/params_{TICKER}.json` if present.

```bash
python main.py train --tickers AAPL MSFT NVDA
python main.py train --tickers SPY --period 2y --interval 1h
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tickers` | AAPL MSFT NVDA | Space-separated ticker list |
| `--period` | `2y` | yfinance history period |
| `--interval` | `1h` | Bar interval |

---

### `optimize`
Bayesian hyperparameter search using Optuna. Tunes 9 XGBoost parameters and the per-ticker ADX threshold simultaneously. Results saved to `models/params_{TICKER}.json` and auto-loaded on the next `train` run.

```bash
python main.py optimize --tickers AAPL --trials 60
python main.py optimize --tickers AAPL MSFT NVDA --trials 100 --retrain
```

| Flag | Default | Description |
|------|---------|-------------|
| `--trials` | 60 | Optuna trial budget |
| `--folds` | 3 | Walk-forward folds used in objective |
| `--period` | `2y` | Training data window |
| `--retrain` | off | Retrain immediately after optimization |

---

### `simulate`
Walk-forward out-of-sample validation with bootstrap 95% confidence intervals.

```bash
python main.py simulate --tickers AAPL --folds 4 --adx-min 20
python main.py simulate --tickers AAPL --adx-compare   # compare multiple ADX thresholds
```

---

### `backtest`
Simple held-out backtest on the last N% of data (default: last 15%).

```bash
python main.py backtest --tickers AAPL --plot
python main.py backtest --tickers NVDA --short          # enable short selling
python main.py backtest --tickers AAPL --skip-pct 0.80
```

---

### `screen`
Scan all tickers that have trained models in the universe and rank them by signal strength.

```bash
python main.py screen --top-n 20
python main.py screen --train-top 10   # also train models for 10 untrained candidates
python main.py screen --tickers AAPL MSFT NVDA   # scan specific tickers only
```

---

### `universe`
Inspect the S&P 500 + NASDAQ-100 ticker universe (fetched from Wikipedia, cached 24h).

```bash
python main.py universe
python main.py universe --refresh            # force re-fetch
python main.py universe --list-trained       # show which tickers have models
python main.py universe --list-untrained     # show tickers without models
```

---

### `run`
Fully autonomous mode. Loads every trained model on disk, scans the universe for opportunities on each poll cycle, opens and manages long and short positions, and retrains stale models overnight.

```bash
python main.py run
python main.py run --dashboard    # run with live Rich terminal dashboard
```

Requires `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` to be set.

---

### `trade`
Manual mode — same as `run` but restricted to an explicit ticker list.

```bash
python main.py trade --tickers AAPL MSFT NVDA
```

---

### `dashboard`
Standalone live terminal dashboard. Can be run without a trading session to monitor models and scan for signals.

```bash
python main.py dashboard
python main.py dashboard --refresh 15    # refresh every 15 seconds
```

---

### `sentiment`
Show FinBERT sentiment scores for recent news articles.

```bash
python main.py sentiment --tickers AAPL NVDA TSLA
```

---

## Architecture

```
RSI-Model/
│
├── config.py                  All parameters: data, ML, bot, universe, screening
├── main.py                    CLI entry point (argparse)
│
├── data/
│   ├── fetcher.py             yfinance price + news; period auto-cap for Yahoo limits
│   ├── indicators.py          35+ indicators: RSI/MACD/BB/ATR/Stoch/CCI/OBV/VWAP/ADX
│   └── universe.py            S&P 500 + NASDAQ-100 fetcher; 24h JSON cache
│
├── ml/
│   ├── features.py            63-feature matrix + percentile labelling + time-series split
│   ├── model.py               XGBoost + RobustScaler; per-ticker save/load
│   ├── trainer.py             5-fold CV → final model; auto-loads Optuna params
│   └── optimizer.py           Optuna: 60-trial Bayesian search over XGBoost + ADX threshold
│
├── screening/
│   └── scanner.py             Parallel universe scan (ThreadPoolExecutor); ranks opportunities
│
├── portfolio/
│   └── manager.py             Risk limits: max positions, exposure, per-trade sizing
│
├── backtest/
│   ├── engine.py              Long + short simulation; Sharpe, drawdown, win rate, PF
│   └── simulate.py            Walk-forward + bootstrap confidence intervals
│
├── bot/
│   ├── alpaca_client.py       Alpaca SDK wrapper (paper + live)
│   ├── strategy.py            Signal generation + position sizing (shorts at 80%)
│   └── trader.py              LiveTrader: manual mode + autonomous mode + retraining
│
├── dashboard/
│   └── terminal.py            Rich terminal dashboard: account / positions / signals / models
│
├── sentiment/
│   └── analyzer.py            FinBERT pipeline; recency-decay weighting; graceful fallback
│
└── models/                    Saved artifacts (git-ignored; regenerate with train/optimize)
    ├── xgb_model_{TICKER}.json
    ├── scaler_{TICKER}.pkl
    ├── params_{TICKER}.json   Optuna best hyperparameters
    └── universe_cache.json    Cached ticker universe
```

### Signal pipeline

```
Price data (yfinance)
    │
    ▼
35+ indicators (RSI, MACD, BB, ATR, ADX, ...)
    │
    ▼
63-feature matrix
  = base indicators
  + lags t-1 / t-2 / t-3 for 7 key indicators
  + hour_of_day + day_of_week
  + FinBERT sentiment score
    │
    ▼
XGBoost classifier  →  DOWN / FLAT / UP  +  confidence
    │
    ▼
ADX regime filter (per-ticker threshold)
    │
    ▼
Portfolio risk check (exposure, concentration, position count)
    │
    ▼
Alpaca market order  →  stop-loss 2% / take-profit 4%
```

---

## Feature set

| Group | Features |
|-------|----------|
| Momentum | RSI (7, 14, 21), RSI slope, ROC (5, 10, 20) |
| Trend | MACD, MACD signal, MACD hist + slope, EMA 9/21 cross, SMA 9/21 cross, price vs SMA20/SMA50, ADX, DI+, DI− |
| Volatility | Bollinger %B + bandwidth + slope, ATR % of price, rolling vol (10, 20 bars) |
| Volume | OBV ratio, volume ratio, volume acceleration |
| Price | Returns (1, 3, 5, 10 bars), price vs VWAP + slope |
| Oscillators | Stochastic K + D + slope, CCI, Williams %R |
| Time | Hour of day, day of week |
| Sentiment | FinBERT score (recency-weighted average over 24h of headlines) |
| Lags | t-1 / t-2 / t-3 of RSI_14, RSI_7, MACD_hist, price_vs_vwap, volume_ratio, BB_pct, Stoch_K |

Labels are assigned by percentile (33rd / 67th) rather than fixed thresholds, guaranteeing ~33/33/33 class balance regardless of regime.

---

## Validated results

Walk-forward validation, 4 folds, 2 years of hourly data, 10,000 bootstrap samples:

| Ticker | ADX filter | Trades | Mean return | 95% CI | P(profitable) |
|--------|-----------|--------|-------------|--------|--------------|
| AAPL | > 20 | 71 | +0.502% | [+0.08%, +0.93%] | 98.9% |
| MSFT | > 30 | 67 | +0.611% | [−0.01%, +1.28%] | 97.4% |
| NVDA | none | 121 | +0.448% | [+0.04%, +0.86%] | 98.4% |

**Annualised return on deployed capital: ~2.8–4.6% per ticker.**
**Max drawdown: < 1.5% across all tickers.**

The strongest features across all tickers are `price_vs_vwap` lags (t-1, t-2, t-3), `hour_of_day`, and `day_of_week`.

---

## Risk disclosures

- **Paper trading only.** The default configuration uses Alpaca's paper-trading endpoint. Live trading requires a funded brokerage account and a deliberate code change.
- **No slippage model.** Approximately 0.05–0.1% per trade would reduce stated returns.
- **Backfill bias.** Models are trained and tested on 2024–2026 market conditions. Performance in different regimes (high inflation, bear market, low-vol) may degrade.
- **Short selling risk.** Short positions carry theoretically unlimited loss. The bot sizes shorts at 80% of normal and enforces the same 2% stop loss.
- **MSFT confidence interval grazes zero.** That ticker is the weakest of the three validated symbols; treat its signals with extra caution.
- **This is a research tool, not financial advice.**
