import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    default_period: str = "60d"
    default_interval: str = "5m"
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
    target_bars: int = 12          # predict return over next N bars
    target_threshold: float = 0.003  # 0.3% move = UP or DOWN, else FLAT
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    n_splits: int = 5              # time-series CV folds
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
        "tree_method": "hist",
        "random_state": 42,
    })
    model_path: str = "models/xgb_model.json"
    scaler_path: str = "models/scaler.pkl"


@dataclass
class SentimentConfig:
    model_name: str = "ProsusAI/finbert"
    max_length: int = 512
    news_lookback_hours: int = 24
    sentiment_decay: float = 0.9   # weight recent news more


@dataclass
class BotConfig:
    paper_trading: bool = True
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    poll_interval_seconds: int = 30
    max_position_pct: float = 0.20   # max 20% of portfolio in one ticker
    kelly_fraction: float = 0.25     # fractional Kelly (conservative)
    stop_loss_pct: float = 0.02      # 2% stop loss
    take_profit_pct: float = 0.04    # 4% take profit
    min_confidence: float = 0.55     # minimum ML confidence to trade
    sentiment_weight: float = 0.15   # how much sentiment adjusts position size


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    bot: BotConfig = field(default_factory=BotConfig)
    tickers: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"])


CONFIG = Config()
