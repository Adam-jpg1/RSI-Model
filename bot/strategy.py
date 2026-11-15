import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

from config import CONFIG
from data.fetcher import get_daily_data
from data.indicators import add_all_indicators, add_daily_trend_features
from ml.features import get_feature_matrix
from ml.model import RSIModel

# Per-hour confidence thresholds derived from ToD IC analysis (Spearman abs-IC across features).
# Open (9 ET) has 31% higher IC than post-lunch (13 ET) → lower threshold to admit more signals.
# Values taper from 0.33 at open to 0.40 at peak-noise hour, then recover into close.
_TOD_CONF: Dict[int, float] = {
    9:  0.33,   # 9:30 AM — open: IC=0.054, reversal+gap features dominant
    10: 0.35,
    11: 0.36,
    12: 0.38,
    13: 0.40,   # 1:30 PM — lunch lull: IC=0.041, worst hour
    14: 0.38,
    15: 0.36,   # 3:30 PM — close run-up
}


def get_dynamic_confidence(dt: datetime = None, base_conf: float = None) -> float:
    """
    Return a time-of-day-adjusted confidence threshold.

    Outside market hours or for non-ET-aware callers, falls back to base_conf
    (or CONFIG.bot.min_confidence). Used only for the live bot — backtests should
    continue using fixed per-ticker thresholds from param files.
    """
    base = base_conf if base_conf is not None else CONFIG.bot.min_confidence
    if dt is None:
        dt = datetime.now()

    # Convert to ET if timezone-aware
    try:
        import zoneinfo
        et_tz = zoneinfo.ZoneInfo("America/New_York")
        if dt.tzinfo is not None:
            dt = dt.astimezone(et_tz)
        hour = dt.hour
    except Exception:
        hour = dt.hour  # fallback: assume already ET

    if hour not in _TOD_CONF:
        return base  # pre-open or post-close — use base

    tod_conf = _TOD_CONF[hour]
    # Scale relative to base: if base was tuned to something other than 0.40,
    # preserve the same proportional adjustment.
    reference = 0.40
    adjustment = tod_conf - reference
    return round(float(np.clip(base + adjustment, 0.33, 0.70)), 3)


def compute_position_size(
    confidence: float,
    sentiment_score: float,
    portfolio_value: float,
    signal: str,
) -> float:
    """Dollar allocation for one trade. Returns 0 for FLAT signals."""
    if signal not in ("UP", "DOWN"):
        return 0.0

    base_pct = CONFIG.bot.max_position_pct  # Full size at threshold; gate is already applied
    sentiment_adj = sentiment_score * CONFIG.bot.sentiment_weight

    if signal == "DOWN":
        base_pct *= CONFIG.bot.short_size_factor  # 80% sizing for shorts

    size_pct = float(np.clip(base_pct + sentiment_adj, 0.0, CONFIG.bot.max_position_pct))
    return portfolio_value * size_pct


def generate_signal(
    df: pd.DataFrame,
    model: RSIModel,
    sentiment_score: float = 0.0,
    daily_df: pd.DataFrame = None,
) -> Optional[Dict]:
    if len(df) < 60:
        return None

    # Merge daily trend features when available
    working_df = df
    if daily_df is not None and len(daily_df) > 1:
        try:
            working_df = add_daily_trend_features(df.copy(), daily_df)
        except Exception:
            working_df = df

    features = get_feature_matrix(working_df, sentiment_score)
    last_row = features.iloc[-1]

    # Daily features may have NaN early in the series (SMA200 needs 200 days).
    # Fill them with neutral values so prediction still runs.
    daily_fill = {c: 0.0 for c in last_row.index if c.startswith("d_")}
    if daily_fill:
        last_row = last_row.fillna(daily_fill)

    # If non-daily features still have NaN, skip — indicator warmup incomplete.
    non_daily = last_row[[c for c in last_row.index if not c.startswith("d_")]]
    if non_daily.isna().any():
        return None

    d_trend = float(working_df["d_trend_score"].iloc[-1]) if "d_trend_score" in working_df.columns else 0.0

    prediction = model.predict_single(last_row)
    signal = prediction["signal"]

    # Hard daily-trend gate: suppress signals that fight the prevailing daily trend
    if CONFIG.bot.daily_trend_filter:
        min_score = CONFIG.bot.daily_trend_min_score
        if signal == "UP" and d_trend < -min_score:
            signal = "FLAT"
        elif signal == "DOWN" and d_trend > min_score:
            signal = "FLAT"

    return {
        "signal": signal,
        "raw_signal": prediction["signal"],   # original model output before gate
        "confidence": prediction["confidence"],
        "probs": prediction["probs"],
        "sentiment_score": sentiment_score,
        "price": float(df["Close"].iloc[-1]),
        "rsi_14": float(df["RSI_14"].iloc[-1]) if "RSI_14" in df.columns else None,
        "macd_hist": float(df["MACD_hist"].iloc[-1]) if "MACD_hist" in df.columns else None,
        "adx": float(df["ADX"].iloc[-1]) if "ADX" in df.columns else None,
        "d_trend": d_trend,
    }
