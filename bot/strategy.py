import pandas as pd
import numpy as np
from typing import Dict, Optional
from ml.model import RSIModel
from ml.features import get_feature_matrix
from data.indicators import add_all_indicators
from config import CONFIG


def compute_position_size(
    confidence: float,
    sentiment_score: float,
    portfolio_value: float,
    signal: str,
) -> float:
    if signal != "UP":
        return 0.0

    base_size = confidence * CONFIG.bot.max_position_pct
    sentiment_adj = sentiment_score * CONFIG.bot.sentiment_weight
    size_pct = float(np.clip(base_size + sentiment_adj, 0.0, CONFIG.bot.max_position_pct))
    return portfolio_value * size_pct


def generate_signal(
    df: pd.DataFrame,
    model: RSIModel,
    sentiment_score: float = 0.0,
) -> Optional[Dict]:
    if len(df) < 60:
        return None

    features = get_feature_matrix(df, sentiment_score)
    last_row = features.iloc[-1]

    if last_row.isna().any():
        return None

    prediction = model.predict_single(last_row)
    return {
        "signal": prediction["signal"],
        "confidence": prediction["confidence"],
        "probs": prediction["probs"],
        "sentiment_score": sentiment_score,
        "price": float(df["Close"].iloc[-1]),
        "rsi_14": float(df["RSI_14"].iloc[-1]) if "RSI_14" in df.columns else None,
        "macd_hist": float(df["MACD_hist"].iloc[-1]) if "MACD_hist" in df.columns else None,
    }
