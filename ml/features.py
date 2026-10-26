import pandas as pd
import numpy as np
from typing import Tuple, List
from config import CONFIG


FEATURE_COLS = [
    "RSI_7", "RSI_14", "RSI_21",
    "MACD", "MACD_signal", "MACD_hist", "MACD_hist_slope",
    "BB_pct", "BB_bandwidth",
    "ATR_pct",
    "Stoch_K", "Stoch_D",
    "CCI", "Williams_R",
    "OBV_ratio",
    "price_vs_vwap",
    "volume_ratio",
    "return_1", "return_3", "return_5", "return_10",
    "volatility_10", "volatility_20",
    "EMA_9_21_cross",
    "price_vs_SMA20", "price_vs_SMA50",
    "RSI_14_slope",
    "SMA_9_21_cross",
]


def build_target(df: pd.DataFrame, n_bars: int = None, threshold: float = None) -> pd.Series:
    n_bars = n_bars or CONFIG.ml.target_bars
    threshold = threshold or CONFIG.ml.target_threshold
    future_return = df["Close"].shift(-n_bars) / df["Close"] - 1.0
    labels = pd.Series(1, index=df.index, dtype=int)  # FLAT
    labels[future_return > threshold] = 2              # UP
    labels[future_return < -threshold] = 0             # DOWN
    return labels


def get_feature_matrix(df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
    available = [c for c in FEATURE_COLS if c in df.columns]
    features = df[available].copy()
    features["sentiment"] = sentiment_score
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    return features


def prepare_dataset(
    df: pd.DataFrame, sentiment_score: float = 0.0
) -> Tuple[pd.DataFrame, pd.Series]:
    features = get_feature_matrix(df, sentiment_score)
    target = build_target(df)
    mask = features.notna().all(axis=1) & target.notna()
    mask &= np.arange(len(df)) < len(df) - CONFIG.ml.target_bars
    return features[mask], target[mask]


def time_series_split(
    X: pd.DataFrame, y: pd.Series, n_splits: int = None
) -> List[Tuple]:
    n_splits = n_splits or CONFIG.ml.n_splits
    n = len(X)
    fold_size = n // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        val_end = fold_size * (i + 1)
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        splits.append((X_train, y_train, X_val, y_val))
    return splits


CLASS_LABELS = {0: "DOWN", 1: "FLAT", 2: "UP"}
