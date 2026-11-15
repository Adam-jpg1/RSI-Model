import os
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class DataConfig:
    default_period: str = "2y"
    train_period: str = "1y"        # training window — 1yr balances regime freshness vs data volume
    default_interval: str = "1h"
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    sma_periods: List[int] = field(default_factory=lambda: [9, 20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 55])
    bb_period: int = 20
    bb_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    stoch_k: int = 14
    stoch_d: int = 3
    cci_period: int = 20
    williams_period: int = 14
    obv_sma: int = 20


@dataclass
class MLConfig:
    target_bars: int = 6           # bars ahead to predict; auto-scaled per interval by trainer
    target_threshold: float = 0.006
    label_pct_low: float = 0.33
    label_pct_high: float = 0.67
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    n_splits: int = 5
    # target_bars auto-scaling: predict the same horizon regardless of bar size
    # 1h → 6 bars (6h), 15m → 24 bars (6h), 5m → 72 bars (6h), 1d → 1 bar
    target_bars_by_interval: Dict[str, int] = field(default_factory=lambda: {
        "1d": 1, "1h": 6, "60m": 6, "30m": 12, "15m": 24, "5m": 72, "2m": 180, "1m": 360,
    })
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "min_child_weight": 6,
        "gamma": 0.1,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": 42,
    })
    model_path: str = "models/xgb_model.json"
    scaler_path: str = "models/scaler.pkl"
    use_first_touch_target: bool = False  # if True, use first-touch stop/take target instead of percentile


@dataclass
class SentimentConfig:
    model_name: str = "ProsusAI/finbert"
    max_length: int = 512
    news_lookback_hours: int = 24
    sentiment_decay: float = 0.9


@dataclass
class UniverseConfig:
    include_sp500: bool = True
    include_nasdaq100: bool = True
    cache_ttl_hours: int = 24
    cache_path: str = "models/universe_cache.json"
    # S&P 500 and NASDAQ-100 are already screened for size/liquidity — no additional filters needed


@dataclass
class ScreeningConfig:
    min_adx: float = 15.0            # signals below this ADX threshold are excluded from rankings
    min_confidence: float = 0.40
    scan_period: str = "20d"      # needs ~130 hourly bars for SMA50 + all indicators
    scan_interval: str = "1h"
    top_n_opportunities: int = 20
    max_scan_workers: int = 6        # parallel yfinance fetches


@dataclass
class BotConfig:
    paper_trading: bool = True
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    poll_interval_seconds: int = 60
    adx_filter_min: float = 12.0            # global ADX threshold (overridden per-ticker by per_ticker_adx)
    per_ticker_adx: Dict[str, float] = field(default_factory=lambda: {
        "AAPL": 11.0,
        "ABBV": 24.8,
        "ADBE": 0.9,
        "AMAT": 8.5,
        "AMD": 39.5,
        "AMZN": 39.5,
        "ANET": 30.2,
        "APP": 35.1,
        "AVGO": 12.2,
        "BAC": 35.9,
        "CAT": 39.5,
        "COIN": 25.1,
        "COST": 3.8,
        "CRM": 27.3,
        "CRWD": 38.6,
        "CVX": 28.3,
        "DIS": 33.7,
        "GE": 17.6,
        "GOOGL": 39.5,
        "GS": 39.2,
        "HD": 31.5,
        "HON": 4.3,
        "JNJ": 3.5,
        "JPM": 32.1,
        "LLY": 32.5,
        "MA": 14.5,
        "META": 7.4,
        "MSFT": 13.7,
        "MU": 18.6,
        "NFLX": 11.0,
        "NKE": 21.9,
        "NVDA": 15.0,
        "QCOM": 39.5,
        "SMCI": 36.6,
        "T": 5.7,
        "TSLA": 36.5,
        "UNH": 39.5,
        "V": 3.9,
        "WMT": 35.9,
        "XOM": 36.9,
    })
    max_position_pct: float = 0.10          # max 10% of portfolio per position
    max_portfolio_exposure: float = 0.80    # max total deployed capital
    max_positions: int = 20
    max_sector_exposure: float = 0.25       # max 25% of portfolio in any single sector
    kelly_fraction: float = 0.25
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    min_confidence: float = 0.40
    sentiment_weight: float = 0.10
    allow_short: bool = True                # enable short selling on DOWN signals
    short_size_factor: float = 0.80         # shorts get 80% of normal size (unlimited loss risk)
    slippage_pct: float = 0.001             # 0.1% per trade simulated slippage
    retrain_interval_hours: int = 168       # retrain weekly when bot is running overnight
    model_staleness_hours: int = 72         # skip models older than 3 days in autonomous mode
    # Dynamic Kelly sizing — scales position size by rolling model accuracy
    dynamic_sizing: bool = True
    dynamic_sizing_window: int = 20         # rolling trade window for win-rate estimate
    dynamic_max_scale: float = 1.5          # max multiplier when model is hot (>70% win rate)
    dynamic_min_scale: float = 0.5          # min multiplier when model is cold (<40% win rate)
    # Daily trend filter — only enter in direction of the daily trend
    daily_trend_filter: bool = True         # skip hourly signals that fight the daily trend
    daily_trend_min_score: int = 2          # minimum daily trend score to allow directional entry


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    bot: BotConfig = field(default_factory=BotConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    tickers: List[str] = field(default_factory=lambda: [
        # Tier 1 — 25%+ annual return at 10% position size
        "MU", "SMCI", "APP", "COIN", "AMD", "LRCX", "QCOM", "AMAT",
        "AVGO", "ANET", "CRWD", "CRM", "ADBE", "TSLA",
        # Tier 2 — 15-25% annual return
        "META", "CAT", "GS", "GE", "AMZN", "XOM", "UNH", "HD",
        "NVDA", "NKE", "JPM", "CVX", "DIS", "HON", "ABBV",
        "WMT", "NFLX", "T", "GOOGL",
        # Tier 3 — 7-15% annual return (diversification)
        "BAC", "V", "LLY", "JNJ", "MA", "COST",
    ])


CONFIG = Config()
