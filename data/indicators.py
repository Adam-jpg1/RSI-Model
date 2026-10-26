import pandas as pd
import numpy as np
from config import CONFIG


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(series: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(series: pd.Series, period: int, n_std: float):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / sma.replace(0, np.nan)
    return upper, lower, pct_b, bandwidth


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(com=period - 1, min_periods=period).mean()


def compute_stochastic(df: pd.DataFrame, k_period: int, d_period: int):
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100.0 * (df["Close"] - low_min) / denom
    d = k.rolling(window=d_period).mean()
    return k, d


def compute_cci(df: pd.DataFrame, period: int) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (typical_price - sma) / (0.015 * mad.replace(0, np.nan))


def compute_williams_r(df: pd.DataFrame, period: int) -> pd.Series:
    high_max = df["High"].rolling(window=period).max()
    low_min = df["Low"].rolling(window=period).min()
    denom = (high_max - low_min).replace(0, np.nan)
    return -100.0 * (high_max - df["Close"]) / denom


def compute_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    obv = (direction * df["Volume"]).cumsum()
    return obv


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tp_vol = (typical_price * df["Volume"]).cumsum()
    cum_vol = df["Volume"].cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    cfg = CONFIG.data
    df = df.copy()
    close = df["Close"]

    for p in cfg.rsi_periods:
        df[f"RSI_{p}"] = compute_rsi(close, p)

    for p in cfg.sma_periods:
        df[f"SMA_{p}"] = close.rolling(window=p).mean()

    for p in cfg.ema_periods:
        df[f"EMA_{p}"] = close.ewm(span=p, adjust=False).mean()

    macd, macd_sig, macd_hist = compute_macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    df["MACD"] = macd
    df["MACD_signal"] = macd_sig
    df["MACD_hist"] = macd_hist

    bb_upper, bb_lower, bb_pct, bb_bw = compute_bollinger_bands(close, cfg.bb_period, cfg.bb_std)
    df["BB_upper"] = bb_upper
    df["BB_lower"] = bb_lower
    df["BB_pct"] = bb_pct
    df["BB_bandwidth"] = bb_bw

    df["ATR"] = compute_atr(df, cfg.atr_period)
    df["ATR_pct"] = df["ATR"] / close

    stoch_k, stoch_d = compute_stochastic(df, cfg.stoch_k, cfg.stoch_d)
    df["Stoch_K"] = stoch_k
    df["Stoch_D"] = stoch_d

    df["CCI"] = compute_cci(df, cfg.cci_period)
    df["Williams_R"] = compute_williams_r(df, cfg.williams_period)

    obv = compute_obv(df)
    df["OBV"] = obv
    df["OBV_SMA"] = obv.rolling(window=cfg.obv_sma).mean()
    df["OBV_ratio"] = obv / df["OBV_SMA"].replace(0, np.nan)

    df["VWAP"] = compute_vwap(df)
    df["price_vs_vwap"] = (close - df["VWAP"]) / df["VWAP"]

    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(window=20).mean()

    for n in [1, 3, 5, 10, 20]:
        df[f"return_{n}"] = close.pct_change(n)

    df["volatility_10"] = close.pct_change().rolling(window=10).std()
    df["volatility_20"] = close.pct_change().rolling(window=20).std()

    if len(cfg.sma_periods) >= 2:
        df["SMA_9_21_cross"] = df.get("SMA_9", df["EMA_9"]) - df.get("SMA_20", df["SMA_20"])

    df["EMA_9_21_cross"] = df["EMA_9"] - df["EMA_21"]
    df["price_vs_SMA20"] = (close - df["SMA_20"]) / df["SMA_20"]
    df["price_vs_SMA50"] = (close - df["SMA_50"]) / df["SMA_50"]

    df["RSI_14_slope"] = df["RSI_14"].diff(3)
    df["MACD_hist_slope"] = df["MACD_hist"].diff(3)

    return df
