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
    "SMA_9_21_cross",
    "ADX", "DI_plus", "DI_minus",
    "roc_5", "roc_10", "roc_20",
    "stoch_k_slope", "volume_accel",
    # Daily trend features (optional — only present when daily data is merged in)
    "d_rsi_14", "d_macd_hist", "d_adx",
    "d_price_vs_sma50", "d_price_vs_sma200", "d_trend_score",
    # Extended daily context (momentum, volatility, range position)
    "d_mom_5", "d_mom_20", "d_rsi_slope", "d_macd_slope",
    "d_vol_ratio", "d_atr_pct", "d_bb_pct", "d_range_pct",
    # Multi-timeframe alignment + relative strength vs SPY + VIX regime
    "d_trend_align", "d_rs_vs_spy_20",
    "d_vix", "d_vix_regime",
    # Weekly trend context (only present when daily data is merged in)
    "w_rsi_14", "w_macd_hist", "w_price_vs_sma20", "w_trend_score", "w_mom_4",
    # SPY cross-market context (only present when add_market_features is called)
    "spy_rsi_14", "spy_macd_hist", "spy_bb_pct", "spy_price_vs_sma20", "spy_volume_ratio",
    # Earnings proximity (only present when add_earnings_features is called)
    "days_to_earnings", "days_since_earnings", "near_earnings",
    # Intra-bar OHLC structure — buying/selling pressure not captured by close-only indicators
    "candle_body", "candle_body_ratio", "lower_shadow",
    # Sector relative strength (only present when add_sector_features is called)
    "sector_rs_5", "sector_rs_20", "sector_rs_slope",
    # Cross-asset macro regime features (populated by add_cross_asset_features)
    "ca_tlt_ret5", "ca_tlt_vs_ma20",
    "ca_hyg_ret5", "ca_hyg_tlt_spr",
    "ca_gld_ret5", "ca_uup_ret5",
    "ca_risk_score",
    "ca_vts9d_ratio", "ca_vts3m_ratio", "ca_vts_regime",
    # Post-earnings drift proxy (populated by add_pead_features)
    # pead_phase_drift covers days 7-20 post-event: IC=0.19-0.23 (AMD/GOOGL/APP)
    "pead_gap_score", "pead_days_since", "pead_bull_active",
    "pead_phase_early", "pead_phase_drift", "pead_drift_score",
    # Intraday VWAP slope — 10-bar slope has IC≈0.022 across diverse tickers
    "vwap_id_slope_10", "vwap_id_accel",
    # Price-Volume Divergence (10-bar) — IC≈0.032; raw price×volume confirmation signal
    "pvd_10",
    # Session range width — IC≈0.023, proxy for intraday institutional volatility
    "sess_range_width",
    # Options expiration day (3rd Friday) — IC≈0.051; gamma pin dissolution drives momentum
    "opex_friday",
    # Post-OpEx drift — IC≈0.030; mean-reversion in 1-3 days after expiration
    "post_opex",
    # Cumulative tick proxy — IC≈0.021; rolling bar-direction sum proxies order flow
    "cum_tick_10", "cum_tick_20",
    # Time-of-day × MACD — IC≈0.038; afternoon MACD signal stronger than morning
    "tod_macd",
    # Month-end institutional rebalancing — IC≈0.025; first days UP / last days DOWN
    "month_first_days", "month_last_days",
    # Macro calendar events (FOMC + NFP) — IC 0.05-0.06; computed by add_macro_event_features()
    "is_fomc_day", "fomc_post_decision", "is_fomc_next", "is_nfp_day",
    # Day-of-week — avg IC: Fri=+0.097, Wed=-0.076, Thu=-0.044, Tue=+0.056 (add_calendar_features)
    "is_mon", "is_tue", "is_wed", "is_thu", "is_fri",
    "opex_monthly", "opex_thursday",
    # Circular seasonal encoding (add_calendar_features) — continuous year-wrap safe
    # week_of_year SHAP #8 (IC: TSLA=+0.078, GOOGL=+0.054); April delta=+0.125, Feb=-0.156
    "week_sin", "week_cos",
    "month_sin", "month_cos",
    # is_q_earnings_month: Jan/Apr/Jul/Oct earnings season avg +0.08 class delta
    # is_close_hour: 20h UTC (4PM ET close bar) avg -0.054 class delta
    "is_q_earnings_month", "is_close_hour",
    # Cross-sectional rank vs portfolio peers — mean-reversion for 5/6 tickers (add_cross_sectional_features)
    "cs_rank_1d", "cs_rank_5d", "cs_rank_10d", "peer_spread_5d",
    # Bar microstructure — HOW price moved within each bar (add_microstructure_features)
    # buying_pressure avg IC=-0.024; tick_imbal_z avg IC=-0.025; both -0.05-0.08 in CHOPPY regime
    "buying_pressure", "bar_efficiency", "wick_imbalance",
    "momentum_efficiency_10", "tick_imbal_z",
    # Regime detection + interaction features (add_regime_features)
    # In CHOPPY: pvd_choppy IC=+0.117, tick_choppy IC=-0.077, opex_choppy IC=+0.086
    "is_trending", "is_choppy", "regime_score",
    "pvd_choppy", "pvd_trending", "tick_choppy", "opex_choppy",
    # Lead-lag returns: external leaders (add_peer_features + add_market_features)
    # SPY->MRVL IC=-0.084 (1h), NVDA->TSLA IC=-0.053 (1h), NVDA->AMD IC=+0.049 (6h)
    "nvda_ret6_lag1", "nvda_ret6_lag2", "nvda_ret6_lag3",
    "spy_ret6_lag1",  "spy_ret6_lag2",  "spy_ret6_lag3",
    # Lead-lag returns: portfolio cross-tickers (add_cross_sectional_features)
    # APP->GOOGL IC=+0.087 (1h), SNOW->TSLA IC=-0.077 (1h), MRVL->AMD IC=+0.057 avg
    "app_ret6_lag1",   "app_ret6_lag2",   "app_ret6_lag3",
    "amd_ret6_lag1",   "amd_ret6_lag2",   "amd_ret6_lag3",
    "googl_ret6_lag1", "googl_ret6_lag2", "googl_ret6_lag3",
    "snow_ret6_lag1",  "snow_ret6_lag2",  "snow_ret6_lag3",
    "mrvl_ret6_lag1",  "mrvl_ret6_lag2",  "mrvl_ret6_lag3",
    "tsla_ret6_lag1",  "tsla_ret6_lag2",  "tsla_ret6_lag3",
]

# Lagged versions of these indicators give XGBoost sequential context.
# IC analysis identified lag-3 of most features as near-zero (noise) — keep only
# lag-3 for MACD_hist and price_vs_vwap which showed consistent IC > 0.02.
_LAG_COLS        = ["MACD_hist", "price_vs_vwap", "RSI_14", "RSI_7", "volume_ratio", "BB_pct", "Stoch_K", "candle_body_ratio"]
_LAG_COLS_FULL   = {"MACD_hist", "price_vs_vwap"}  # these get lag 1-3; rest get lag 1-2
_LAG_PERIODS     = [1, 2, 3]  # only applied to _LAG_COLS_FULL; others use [1, 2]

CLASS_LABELS = {0: "DOWN", 1: "FLAT", 2: "UP"}


def target_bars_for_interval(interval: str) -> int:
    """Return the number of bars matching a ~6-hour prediction horizon for the given interval."""
    return CONFIG.ml.target_bars_by_interval.get(interval, CONFIG.ml.target_bars)


def build_target(df: pd.DataFrame, n_bars: int = None, threshold: float = None) -> pd.Series:
    n_bars = n_bars or CONFIG.ml.target_bars
    future_return = df["Close"].shift(-n_bars) / df["Close"] - 1.0

    # Percentile-based thresholds guarantee balanced classes (≈33/33/33) regardless of
    # the ticker's volatility regime. This removes the need for class weighting.
    valid = future_return.dropna()
    lo = float(valid.quantile(CONFIG.ml.label_pct_low))
    hi = float(valid.quantile(CONFIG.ml.label_pct_high))

    labels = pd.Series(1, index=df.index, dtype=int)  # FLAT
    labels[future_return > hi] = 2                     # UP
    labels[future_return < lo] = 0                     # DOWN
    return labels


def build_target_first_touch(
    df: pd.DataFrame,
    take_profit: float = None,
    stop_loss: float = None,
    max_bars: int = None,
) -> pd.Series:
    """
    First-touch target: label 2 (UP) if price hits take_profit before stop_loss
    within max_bars, label 0 (DOWN) if stop_loss hits first, label 1 (FLAT) if neither.

    This directly maps to trading outcomes, unlike percentile targets which only
    predict relative return rank. Expects 'High' and 'Low' columns in df.
    """
    take = take_profit if take_profit is not None else CONFIG.bot.take_profit_pct
    stop = stop_loss if stop_loss is not None else CONFIG.bot.stop_loss_pct
    n = max_bars if max_bars is not None else CONFIG.ml.target_bars * 4  # ~24h horizon

    closes = df["Close"].values
    highs = df["High"].values if "High" in df.columns else closes
    lows = df["Low"].values if "Low" in df.columns else closes
    labels = np.ones(len(df), dtype=int)  # FLAT default

    for i in range(len(df) - 1):
        entry = closes[i]
        tp_level = entry * (1 + take)
        sl_level = entry * (1 - stop)
        end = min(i + 1 + n, len(df))
        for j in range(i + 1, end):
            if highs[j] >= tp_level:
                labels[i] = 2  # UP — take profit hit first
                break
            if lows[j] <= sl_level:
                labels[i] = 0  # DOWN — stop loss hit first
                break
        # else: neither hit within n bars → stays FLAT

    return pd.Series(labels, index=df.index)


def get_feature_matrix(df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
    available = [c for c in FEATURE_COLS if c in df.columns]
    features = df[available].copy()

    # Lagged indicator values — gives XGBoost context about indicator direction/momentum
    for col in _LAG_COLS:
        if col in df.columns:
            periods = _LAG_PERIODS if col in _LAG_COLS_FULL else [1, 2]
            for lag in periods:
                features[f"{col}_lag{lag}"] = df[col].shift(lag)

    # ORB (Opening Range Breakout) features — price position relative to first bar of day
    _orb_cols = ["price_vs_orb_high", "price_vs_orb_low", "intraday_return", "orb_width", "overnight_gap"]
    for col in _orb_cols:
        if col in df.columns:
            features[col] = df[col]

    # Time-of-day, day-of-week, and seasonality — intraday/seasonal patterns are real and free
    idx = df.index
    if hasattr(idx, "hour"):
        features["hour_of_day"] = idx.hour
        features["day_of_week"] = idx.dayofweek
        features["month_of_year"] = idx.month
        features["week_of_year"] = idx.isocalendar().week.astype(int).values if hasattr(idx, "isocalendar") else idx.week
    elif hasattr(idx, "to_series"):
        dt = idx.to_series()
        features["hour_of_day"] = dt.dt.hour.values
        features["day_of_week"] = dt.dt.dayofweek.values
        features["month_of_year"] = dt.dt.month.values
        features["week_of_year"] = dt.dt.isocalendar().week.astype(int).values

    # Volatility-normalized overnight gap — ticker-agnostic Z-score of gap size (_add_orb_features)
    _orb_z = ["overnight_gap_z"]
    for col in _orb_z:
        if col in df.columns:
            features[col] = df[col]

    # Opening-hour interaction features — concentrate reversal signal at the FIRST bar of each day.
    # overnight_gap IC = -0.124 at first bar (p=0.005) but only +0.014 across all bars.
    # The day-broadcast overnight_gap is diluted by intraday bars where it carries no edge.
    # Date-boundary detection is DST-safe (works for both UTC 13:30 EDT and 14:30 EST opens).
    if "overnight_gap" in df.columns:
        _dates = pd.to_datetime(df.index).date
        _date_s = pd.Series(_dates, index=df.index)
        is_first_bar = _date_s.ne(_date_s.shift(1)).astype(float).values
        features["open_gap_x"] = df["overnight_gap"].values * is_first_bar
        if "MACD_hist_slope" in df.columns:
            features["open_macd_slope_x"] = df["MACD_hist_slope"].values * is_first_bar

    # Cross-asset lagged RSI — peer overbought/oversold at T-lag predicts target return at T+6h.
    # IC: spy_rsi14_lag1 ≈ -0.085 for MRVL/TSLA; nvda_rsi14_lag6 ≈ -0.086 for APP;
    # xlk_rsi14_lag1 ≈ -0.061 for SNOW; vix_rsi14_lag1 ≈ +0.062 for MRVL.
    # Negative IC = overbought peer → target falls (mean-reversion within sector).
    # Positive VIX IC = peak fear → stocks recover.
    _cross_rsi = [
        ("spy_rsi_14",  [1, 2, 3]),
        ("nvda_rsi_14", [1, 2, 6]),
        ("xlk_rsi_14",  [1, 2, 3]),
        ("vix_rsi_14",  [1, 2]),
    ]
    for col, lags in _cross_rsi:
        if col in df.columns:
            for lag in lags:
                features[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Composite: candle body direction × volume — captures institutional-strength moves
    if "candle_body_ratio" in df.columns and "volume_ratio" in df.columns:
        features["bar_momentum"] = (
            df["candle_body_ratio"] * df["volume_ratio"].clip(0, 5)
        ).fillna(0.0)

    features["sentiment"] = sentiment_score
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    return features


def compute_class_weights(y: pd.Series) -> np.ndarray:
    """Inverse-frequency weights so UP/DOWN/FLAT each contribute equally."""
    counts = np.bincount(y, minlength=3).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights_per_class = len(y) / (3.0 * counts)
    return weights_per_class[y.values]


def prepare_dataset(
    df: pd.DataFrame, sentiment_score: float = 0.0, interval: str = "1h"
) -> Tuple[pd.DataFrame, pd.Series]:
    n_bars = target_bars_for_interval(interval)
    features = get_feature_matrix(df, sentiment_score)
    if CONFIG.ml.use_first_touch_target:
        max_bars = n_bars * 8  # ~48h lookahead — better UP/DOWN class balance
        target = build_target_first_touch(df, max_bars=max_bars)
        mask = features.notna().all(axis=1) & target.notna()
        mask &= np.arange(len(df)) < len(df) - max_bars
    else:
        target = build_target(df, n_bars=n_bars)
        mask = features.notna().all(axis=1) & target.notna()
        mask &= np.arange(len(df)) < len(df) - n_bars
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
