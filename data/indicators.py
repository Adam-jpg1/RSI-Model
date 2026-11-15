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


def compute_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP reset each calendar day — cleaner price anchor for slope computation."""
    if "Volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = typical_price * df["Volume"]
    try:
        date_key = pd.Series(df.index.date, index=df.index)
    except AttributeError:
        date_key = pd.Series(
            [pd.Timestamp(ts).date() for ts in df.index], index=df.index
        )
    cum_pv = pv.groupby(date_key).cumsum()
    cum_v = df["Volume"].groupby(date_key).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)


def compute_adx(df: pd.DataFrame, period: int = 14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    dm_plus = pd.Series(dm_plus, index=df.index)
    dm_minus = pd.Series(dm_minus, index=df.index)

    atr = compute_atr(df, period)
    di_plus = 100 * dm_plus.ewm(com=period - 1, min_periods=period).mean() / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(com=period - 1, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan))
    adx = dx.ewm(com=period - 1, min_periods=period).mean()
    return adx, di_plus, di_minus


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

    # Explicit slope/momentum features for the strongest indicators
    df["vwap_dev_slope"] = df["price_vs_vwap"].diff(3)      # VWAP deviation trend
    df["stoch_k_slope"] = df["Stoch_K"].diff(3)             # stochastic momentum
    df["volume_accel"] = df["volume_ratio"].diff(3)         # volume acceleration
    df["bb_pct_slope"] = df["BB_pct"].diff(3)               # BB position trend

    adx, di_plus, di_minus = compute_adx(df, cfg.atr_period)
    df["ADX"] = adx
    df["DI_plus"] = di_plus
    df["DI_minus"] = di_minus

    for n in [5, 10, 20]:
        df[f"roc_{n}"] = close.pct_change(n) * 100

    df["volume_obv_trend"] = df["OBV"].diff(5)

    # Candle-body features: intrabar structure not captured by close-only indicators
    if "Open" in df.columns and "High" in df.columns and "Low" in df.columns:
        range_ = (df["High"] - df["Low"]).replace(0, np.nan)
        body = df["Close"] - df["Open"]
        df["candle_body"]       = body / df["Open"].replace(0, np.nan)
        df["candle_body_ratio"] = body / range_          # +1=full bull, -1=full bear, 0=doji
        df["upper_shadow"]      = (df["High"] - df[["Open", "Close"]].max(axis=1)) / range_
        df["lower_shadow"]      = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / range_

    df = _add_orb_features(df)

    # --- Novel edge features -------------------------------------------
    # 1. Intraday VWAP slope: captures institutional order-flow direction.
    #    Computed on intraday-reset VWAP to avoid multi-week drift contaminating slope.
    if "High" in df.columns and "Volume" in df.columns:
        vwap_id = compute_intraday_vwap(df)
        atr_safe = df["ATR"].replace(0, np.nan)
        df["vwap_id_slope_5"]  = vwap_id.diff(5)  / atr_safe   # 5-bar slope, ATR-normalised
        df["vwap_id_slope_10"] = vwap_id.diff(10) / atr_safe   # 10-bar slope
        df["vwap_id_accel"]    = df["vwap_id_slope_5"] - df["vwap_id_slope_10"]

    # 2. Price-Volume Divergence (PVD): +1 when price & volume confirm, -1 when diverge.
    #    Rolling sums give a regime score: persistently aligned = trend, diverge = reversal.
    if "volume_ratio" in df.columns:
        price_dir = np.sign(close.diff()).fillna(0)
        vol_dir   = np.sign(df["volume_ratio"] - 1.0)           # +1 above avg, -1 below
        pvd_bar   = (price_dir * vol_dir).replace(0, np.nan).ffill().fillna(0)
        df["pvd_3"]  = pvd_bar.rolling(3).sum()                 # -3..+3
        df["pvd_10"] = pvd_bar.rolling(10).sum()                # -10..+10

    # 3. Session range position: where is price within today's running high/low?
    if "High" in df.columns:
        try:
            date_key = pd.Series(df.index.date, index=df.index)
        except AttributeError:
            date_key = pd.Series(
                [pd.Timestamp(ts).date() for ts in df.index], index=df.index
            )
        sess_high = df["High"].groupby(date_key).cummax()
        sess_low  = df["Low"].groupby(date_key).cummin()
        sess_range = (sess_high - sess_low).replace(0, np.nan)
        df["sess_range_pos"] = (close - sess_low) / sess_range
        df["sess_range_width"] = sess_range / close

    # 4. Volatility compression patterns (NR4 / NR7):
    #    Current bar has the smallest high–low range of the last 4 or 7 bars →
    #    classic breakout precursor (Crabel 1990).  Combined with directional
    #    features, XGBoost can learn whether compression precedes an UP or DOWN break.
    if "High" in df.columns:
        range_ = (df["High"] - df["Low"]).replace(0, np.nan)
        atr_safe = df["ATR"].replace(0, np.nan)
        range_norm = range_ / atr_safe                            # ATR-normalised bar range
        df["nr4"]          = (range_norm == range_norm.rolling(4).min()).astype(float)
        df["nr7"]          = (range_norm == range_norm.rolling(7).min()).astype(float)
        df["range_rank_10"] = range_norm.rolling(10).rank(pct=True)  # 0=narrowest, 1=widest
        # Inside bar: full range contained within previous bar → compression before breakout
        df["inside_bar"] = (
            (df["High"] <= df["High"].shift(1)) & (df["Low"] >= df["Low"].shift(1))
        ).astype(float)

    # --- Novel edge features: time-of-day microstructure & options expiration ---

    # 5. Options expiration week (OpEx): 3rd Friday of each month drives stock pinning
    #    near round-number strikes. After OpEx, gamma exposure releases and momentum
    #    is freer. IC hypothesis: directional signals are stronger post-OpEx.
    try:
        try:
            bar_dates_pd = pd.DatetimeIndex(df.index)
        except Exception:
            bar_dates_pd = pd.DatetimeIndex([pd.Timestamp(ts) for ts in df.index])

        # 3rd Friday of month: Fridays where (day - 1) // 7 == 2 (0-indexed week)
        def _is_opex_friday(dt):
            return dt.weekday() == 4 and 15 <= dt.day <= 21  # 3rd Friday

        opex_fridays = {d.date() for d in bar_dates_pd if _is_opex_friday(d)}

        opex_week_arr = np.zeros(len(df))
        opex_friday_arr = np.zeros(len(df))
        post_opex_arr = np.zeros(len(df))

        for i, ts in enumerate(df.index):
            try:
                d = pd.Timestamp(ts).date()
            except Exception:
                continue
            # Flag: within 5 trading days BEFORE an OpEx Friday (inclusive)
            for of in opex_fridays:
                delta = (of - d).days
                if 0 <= delta <= 5:
                    opex_week_arr[i] = 1.0
                if delta == 0:
                    opex_friday_arr[i] = 1.0
                if -3 <= delta < 0:  # 1-3 days after OpEx
                    post_opex_arr[i] = 1.0

        df["opex_week"]    = opex_week_arr
        df["opex_friday"]  = opex_friday_arr
        df["post_opex"]    = post_opex_arr
    except Exception:
        pass

    # 6. Cumulative tick proxy: rolling sum of bar direction (close > open = +1).
    #    Acts as a free order-flow proxy — more "up-close" bars = accumulation.
    #    IC ≈ 0.03–0.05 in liquid growth tech stocks.
    if "Open" in df.columns:
        try:
            bar_tick = np.sign(df["Close"] - df["Open"]).fillna(0)
            df["cum_tick_10"] = bar_tick.rolling(10).sum()   # -10..+10
            df["cum_tick_20"] = bar_tick.rolling(20).sum()   # -20..+20
        except Exception:
            pass

    # 7b. Month-end rebalancing effect: index funds rebalance on last/first trading days.
    #     Last day: selling losers (mean-reversion pressure). First day: new cash inflows (buying).
    #     IC hypothesis: first days of month predict UP, last days neutral-to-down.
    try:
        try:
            ts_series = pd.DatetimeIndex(df.index)
        except Exception:
            ts_series = pd.DatetimeIndex([pd.Timestamp(ts) for ts in df.index])

        # Approximate: day 1-2 of month = first days; day >= 27 = last days
        day_of_month = ts_series.day
        is_month_first = (day_of_month <= 2).astype(float)
        is_month_last  = (day_of_month >= 27).astype(float)

        df["month_first_days"] = pd.Series(is_month_first, index=df.index)
        df["month_last_days"]  = pd.Series(is_month_last,  index=df.index)
    except Exception:
        pass

    # 7. Time-of-day conditional MACD signal:
    #    Morning momentum (9-11h) tends to be gap extension / institutional open.
    #    Afternoon momentum (14-16h) tends to be directional into the close.
    #    Interaction captures the regime difference between opening and closing range.
    try:
        try:
            bar_hours = pd.DatetimeIndex(df.index).hour
        except Exception:
            bar_hours = np.array([pd.Timestamp(ts).hour for ts in df.index])

        hour_norm = pd.Series(bar_hours, index=df.index).clip(9, 16)
        # Normalize to 0=open, 1=close (9→0, 16→1)
        hour_norm = (hour_norm - 9.0) / 7.0

        if "MACD_hist" in df.columns:
            df["tod_macd"] = df["MACD_hist"] * hour_norm  # stronger signal near close
        if "return_1" in df.columns:
            df["tod_return"] = df["return_1"] * hour_norm
        if "volume_ratio" in df.columns:
            df["tod_volume"] = df["volume_ratio"] * hour_norm
    except Exception:
        pass

    # 8. Time-normalized relative volume (RVOL): compares volume to the typical volume
    #    for the same hour-of-day — detects unusual institutional activity better than
    #    plain volume_ratio which averages across all hours.
    try:
        try:
            bar_hours_s = pd.DatetimeIndex(df.index).hour
        except Exception:
            bar_hours_s = np.array([pd.Timestamp(ts).hour for ts in df.index])
        hour_series = pd.Series(bar_hours_s, index=df.index)
        rvol_tod = pd.Series(1.0, index=df.index)
        for h in range(9, 17):
            mask = hour_series == h
            if mask.sum() > 5:
                hour_vol = df.loc[mask, "Volume"]
                hour_avg = hour_vol.rolling(20, min_periods=5).mean().replace(0, np.nan)
                rvol_tod.loc[mask] = (hour_vol / hour_avg).clip(0, 10).fillna(1.0)
        df["rvol_tod"] = rvol_tod
    except Exception:
        pass

    # 9. RSI divergence: price at N-bar high but RSI below its N-bar high (bear div).
    #    Classic momentum exhaustion signal — predicts mean-reversion when IC > 0.03.
    if "RSI_14" in df.columns:
        try:
            win = 14
            price_hi = close.rolling(win).max()
            rsi_hi   = df["RSI_14"].rolling(win).max()
            # Bearish divergence: price at high, RSI lower than its recent high
            at_price_high = (close >= price_hi * 0.995)
            rsi_lagging   = (df["RSI_14"] < rsi_hi * 0.95)
            df["rsi_bear_div"] = (at_price_high & rsi_lagging).astype(float)
            # Bullish divergence: price at low, RSI above its recent low
            price_lo = close.rolling(win).min()
            rsi_lo   = df["RSI_14"].rolling(win).min()
            at_price_low  = (close <= price_lo * 1.005)
            rsi_leading   = (df["RSI_14"] > rsi_lo * 1.05)
            df["rsi_bull_div"] = (at_price_low & rsi_leading).astype(float)
        except Exception:
            pass

    return df


def _add_orb_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Opening Range Breakout features based on the first bar of each trading day.
    All features are computable from known data at bar evaluation time — no lookahead.
    """
    if "High" not in df.columns or "Low" not in df.columns:
        return df

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    opens = df["Open"].values if "Open" in df.columns else closes

    try:
        dates_arr = np.array(df.index.date)
    except AttributeError:
        dates_arr = np.array([pd.Timestamp(ts).date() for ts in df.index])

    n = len(df)

    # Map each date to the index of its first bar (chronologically)
    first_bar_of_day: dict = {}
    for i in range(n):
        d = dates_arr[i]
        if d not in first_bar_of_day:
            first_bar_of_day[d] = i

    # Map each date to the last bar's close (for computing next day's overnight gap)
    last_close_of_day: dict = {}
    for i in range(n):
        d = dates_arr[i]
        last_close_of_day[d] = closes[i]  # last write wins → last bar of that day

    sorted_days = sorted(first_bar_of_day.keys())

    orb_high_arr = np.full(n, np.nan)
    orb_low_arr = np.full(n, np.nan)
    intraday_ret_arr = np.full(n, np.nan)
    orb_width_arr = np.full(n, np.nan)
    overnight_gap_arr = np.full(n, np.nan)

    for day_idx, day in enumerate(sorted_days):
        first_i = first_bar_of_day[day]
        day_orb_high = highs[first_i]
        day_orb_low = lows[first_i]
        day_open = opens[first_i]

        prev_close = None
        if day_idx > 0:
            prev_close = last_close_of_day.get(sorted_days[day_idx - 1])

        for i in range(first_i, n):
            if dates_arr[i] != day:
                break
            orb_high_arr[i] = day_orb_high
            orb_low_arr[i] = day_orb_low
            if day_open != 0:
                intraday_ret_arr[i] = (closes[i] - day_open) / day_open
            if day_orb_high != 0:
                orb_width_arr[i] = (day_orb_high - day_orb_low) / day_orb_high
            if prev_close and prev_close != 0:
                overnight_gap_arr[i] = (day_open - prev_close) / prev_close

    with np.errstate(divide="ignore", invalid="ignore"):
        df["price_vs_orb_high"] = np.where(
            orb_high_arr > 0, (closes - orb_high_arr) / orb_high_arr, np.nan
        )
        df["price_vs_orb_low"] = np.where(
            orb_low_arr > 0, (closes - orb_low_arr) / orb_low_arr, np.nan
        )

    df["intraday_return"] = intraday_ret_arr
    df["orb_width"] = orb_width_arr
    df["overnight_gap"] = overnight_gap_arr

    # Volatility-normalized gap: Z-score relative to recent 20-day gap distribution.
    # Identifies structurally unusual gaps (earnings, news) independent of ticker vol.
    _gap_std = df["overnight_gap"].rolling(20, min_periods=5).std()
    df["overnight_gap_z"] = (df["overnight_gap"] / _gap_std.replace(0, np.nan)).fillna(0.0)

    return df


def add_daily_trend_features(hourly_df: pd.DataFrame, daily_df: pd.DataFrame,
                             compute_weekly: bool = False,
                             spy_daily_df: pd.DataFrame = None,
                             vix_daily_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Merge previous-day trend indicators into an hourly dataframe.

    Uses a date-keyed lookup with 1-day shift so each hourly bar only sees
    the prior calendar day's daily close — no look-ahead bias.

    Added columns (prefix d_):
        d_rsi_14, d_macd_hist, d_adx, d_price_vs_sma50,
        d_price_vs_sma200, d_trend_score (-3 … +3)
        d_rs_vs_spy_5, d_rs_vs_spy_20  (relative return vs SPY, when spy_daily_df given)
        d_trend_align  (count of MAs pointing up: 0-4)
        d_vix, d_vix_vs_ma20, d_vix_regime  (volatility regime, when vix_daily_df given)

    compute_weekly: if True, also adds weekly trend features (w_*).
        Off by default because weekly features add noise for XGBoost when
        training data is limited (< ~1500 bars per fold).
    """
    import bisect

    daily = add_all_indicators(daily_df.copy())

    close_d = daily["Close"]
    sma200 = close_d.rolling(200).mean()

    d_rsi  = daily.get("RSI_14",          pd.Series(50.0, index=daily.index))
    d_macd = daily.get("MACD_hist",       pd.Series(0.0,  index=daily.index))
    d_adx  = daily.get("ADX",             pd.Series(20.0, index=daily.index))
    d_sma50 = daily.get("price_vs_SMA50", pd.Series(0.0,  index=daily.index))
    d_sma200 = (close_d / sma200 - 1.0).fillna(0.0)

    trend_score = (
        np.where(d_rsi > 55, 1, np.where(d_rsi < 45, -1, 0)).astype(float)
        + np.sign(d_macd.fillna(0).values)
        + np.sign(d_sma50.fillna(0).values)
    )

    # Additional daily momentum / volatility features
    d_mom_5  = close_d.pct_change(5).fillna(0.0)
    d_mom_20 = close_d.pct_change(20).fillna(0.0)
    d_rsi_slope  = d_rsi.diff(3).fillna(0.0)
    d_macd_slope = d_macd.diff(3).fillna(0.0)
    d_vol_ratio  = daily.get("volume_ratio", pd.Series(1.0, index=daily.index)).fillna(1.0)
    d_atr_pct    = daily.get("ATR_pct",      pd.Series(0.0, index=daily.index)).fillna(0.0)
    d_bb_pct     = daily.get("BB_pct",       pd.Series(0.5, index=daily.index)).fillna(0.5)

    # 52-week range position (0=at low, 1=at high)
    high_52w = close_d.rolling(252, min_periods=20).max()
    low_52w  = close_d.rolling(252, min_periods=20).min()
    range_52w = (high_52w - low_52w).replace(0, np.nan)
    d_range_pct = ((close_d - low_52w) / range_52w).fillna(0.5)

    # Multi-timeframe trend alignment: count of MA slopes pointing up (0–4)
    # EMA5 > EMA10 > EMA20 > EMA50 — each counts 1 point
    ema5  = close_d.ewm(span=5,  adjust=False).mean()
    ema10 = close_d.ewm(span=10, adjust=False).mean()
    ema20 = close_d.ewm(span=20, adjust=False).mean()
    ema50 = close_d.ewm(span=50, adjust=False).mean()
    d_trend_align = (
        (ema5  > ema5.shift(1)).astype(float)
        + (ema10 > ema10.shift(1)).astype(float)
        + (ema20 > ema20.shift(1)).astype(float)
        + (ema50 > ema50.shift(1)).astype(float)
    ).fillna(2.0)

    # Relative strength vs SPY: stock return minus SPY return over 5 and 20 days
    d_rs_vs_spy_5  = pd.Series(0.0, index=daily.index)
    d_rs_vs_spy_20 = pd.Series(0.0, index=daily.index)
    if spy_daily_df is not None and len(spy_daily_df) >= 25:
        spy_close = spy_daily_df["Close"].reindex(close_d.index, method="ffill")
        stock_ret5  = close_d.pct_change(5).fillna(0.0)
        stock_ret20 = close_d.pct_change(20).fillna(0.0)
        spy_ret5    = spy_close.pct_change(5).fillna(0.0)
        spy_ret20   = spy_close.pct_change(20).fillna(0.0)
        d_rs_vs_spy_5  = (stock_ret5  - spy_ret5).fillna(0.0)
        d_rs_vs_spy_20 = (stock_ret20 - spy_ret20).fillna(0.0)

    # VIX regime features: volatility level, trend, regime bucket
    d_vix        = pd.Series(20.0, index=daily.index)  # neutral default
    d_vix_vs_ma20 = pd.Series(0.0, index=daily.index)
    d_vix_regime  = pd.Series(1.0, index=daily.index)  # 0=low, 1=normal, 2=high
    if vix_daily_df is not None and len(vix_daily_df) >= 25:
        vix_close = vix_daily_df["Close"].reindex(close_d.index, method="ffill").fillna(20.0)
        vix_ma20  = vix_close.rolling(20, min_periods=5).mean().fillna(20.0)
        d_vix         = (vix_close / 30.0).clip(0.0, 3.0)      # normalized, 1.0 = VIX@30
        d_vix_vs_ma20 = ((vix_close - vix_ma20) / vix_ma20).fillna(0.0)
        d_vix_regime  = pd.Series(
            np.where(vix_close < 15, 0.0, np.where(vix_close > 25, 2.0, 1.0)),
            index=close_d.index
        )

    daily_feat = pd.DataFrame({
        "d_rsi_14":          d_rsi.values,
        "d_macd_hist":       d_macd.values,
        "d_adx":             d_adx.values,
        "d_price_vs_sma50":  d_sma50.values,
        "d_price_vs_sma200": d_sma200.values,
        "d_trend_score":     trend_score,
        # New: daily momentum / volatility context
        "d_mom_5":           d_mom_5.values,
        "d_mom_20":          d_mom_20.values,
        "d_rsi_slope":       d_rsi_slope.values,
        "d_macd_slope":      d_macd_slope.values,
        "d_vol_ratio":       d_vol_ratio.values,
        "d_atr_pct":         d_atr_pct.values,
        "d_bb_pct":          d_bb_pct.values,
        "d_range_pct":       d_range_pct.values,
        "d_trend_align":     d_trend_align.values,
        "d_rs_vs_spy_5":     d_rs_vs_spy_5.values,
        "d_rs_vs_spy_20":    d_rs_vs_spy_20.values,
        "d_vix":             d_vix.values,
        "d_vix_vs_ma20":     d_vix_vs_ma20.values,
        "d_vix_regime":      d_vix_regime.values,
    }, index=daily.index)

    # Shift by 1 row: today's hourly bars use yesterday's daily close
    daily_feat = daily_feat.shift(1).dropna()

    # Build sorted list of (date, row_dict) for O(log n) lookup
    def _to_date(ts):
        try:
            return pd.Timestamp(ts).date()
        except Exception:
            return None

    sorted_dates = []
    date_vals: dict = {}
    for ts, row in daily_feat.iterrows():
        d = _to_date(ts)
        if d is not None and d not in date_vals:
            sorted_dates.append(d)
            date_vals[d] = row.to_dict()
    sorted_dates.sort()

    d_cols = list(daily_feat.columns)
    n = len(hourly_df)
    arrays = {c: np.zeros(n) for c in d_cols}

    for i, ts in enumerate(hourly_df.index):
        d = _to_date(ts)
        if d is None:
            continue
        pos = bisect.bisect_right(sorted_dates, d) - 1
        if pos >= 0:
            row = date_vals[sorted_dates[pos]]
            for c in d_cols:
                arrays[c][i] = row[c]

    result = hourly_df.copy()
    for c in d_cols:
        result[c] = arrays[c]

    # --- Weekly trend features (w_*) --- off by default, enable with compute_weekly=True
    if not compute_weekly:
        return result

    try:
        weekly_close = daily["Close"].resample("W-FRI").last().dropna()
        if len(weekly_close) >= 30:
            w_rsi = compute_rsi(weekly_close, period=14).fillna(50.0)
            _, _, w_macd_h = compute_macd(weekly_close, fast=12, slow=26, signal=9)
            w_macd_h = w_macd_h.fillna(0.0)
            w_sma20 = weekly_close.rolling(20).mean()
            w_price_vs_sma20 = (weekly_close / w_sma20 - 1.0).fillna(0.0)
            w_mom_4 = weekly_close.pct_change(4).fillna(0.0)
            w_trend_score = (
                np.where(w_rsi > 55, 1, np.where(w_rsi < 45, -1, 0)).astype(float)
                + np.sign(w_macd_h.fillna(0).values)
                + np.sign(w_price_vs_sma20.fillna(0).values)
            )
            weekly_feat = pd.DataFrame({
                "w_rsi_14":         w_rsi.values,
                "w_macd_hist":      w_macd_h.values,
                "w_price_vs_sma20": w_price_vs_sma20.values,
                "w_trend_score":    w_trend_score,
                "w_mom_4":          w_mom_4.values,
            }, index=weekly_close.index)
            # Shift 1 week: hourly bar sees previous completed week only
            weekly_feat = weekly_feat.shift(1).dropna()

            sorted_wdates = []
            wdate_vals: dict = {}
            for ts, row in weekly_feat.iterrows():
                d = _to_date(ts)
                if d is not None and d not in wdate_vals:
                    sorted_wdates.append(d)
                    wdate_vals[d] = row.to_dict()
            sorted_wdates.sort()

            w_cols = list(weekly_feat.columns)
            w_arrays = {c: np.zeros(n) for c in w_cols}
            for i, ts in enumerate(hourly_df.index):
                d = _to_date(ts)
                if d is None:
                    continue
                pos = bisect.bisect_right(sorted_wdates, d) - 1
                if pos >= 0:
                    row = wdate_vals[sorted_wdates[pos]]
                    for c in w_cols:
                        w_arrays[c][i] = row[c]
            for c in w_cols:
                result[c] = w_arrays[c]
    except Exception:
        pass  # weekly features are additive; skip if resampling fails

    return result


def add_earnings_features(df: pd.DataFrame, earnings_dates: list) -> pd.DataFrame:
    """Add earnings proximity features.

    days_to_earnings:  trading days until next earnings (capped at 63)
    days_since_earnings: trading days since last earnings (capped at 63)
    near_earnings: 1 if within 5 trading days before or 2 after earnings

    These help the model discount signals near high-uncertainty events.
    """
    if not earnings_dates:
        return df

    try:
        import bisect
        from datetime import date

        def _to_date_obj(ts):
            try:
                return pd.Timestamp(ts).date()
            except Exception:
                return None

        bar_dates = [_to_date_obj(ts) for ts in df.index]
        sorted_ed = sorted(earnings_dates)
        n = len(df)

        days_to = np.full(n, 63.0)
        days_since = np.full(n, 63.0)
        near = np.zeros(n, dtype=float)

        for i, d in enumerate(bar_dates):
            if d is None:
                continue
            # Next earnings (at or after d)
            pos_next = bisect.bisect_left(sorted_ed, d)
            if pos_next < len(sorted_ed):
                delta_next = (sorted_ed[pos_next] - d).days
                days_to[i] = min(delta_next, 63.0)
            # Last earnings (strictly before d)
            pos_prev = bisect.bisect_left(sorted_ed, d) - 1
            if pos_prev >= 0:
                delta_prev = (d - sorted_ed[pos_prev]).days
                days_since[i] = min(delta_prev, 63.0)
            # Near-earnings flag: within 5 calendar days before or 2 after
            near[i] = 1.0 if days_to[i] <= 5 or days_since[i] <= 2 else 0.0

        df["days_to_earnings"] = days_to
        df["days_since_earnings"] = days_since
        df["near_earnings"] = near
    except Exception:
        pass
    return df


def add_market_features(df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """Add SPY cross-market features to df.

    Computes key SPY indicators and merges them into df by timestamp
    using forward-fill so the model always has the most recent SPY context.
    Features added: spy_rsi_14, spy_macd_hist, spy_bb_pct,
                    spy_price_vs_sma20, spy_volume_ratio.
    """
    if spy_df is None or len(spy_df) < 30:
        return df
    try:
        spy = spy_df.copy()
        spy["spy_rsi_14"]         = compute_rsi(spy["Close"], 14)
        _, _, spy_mh              = compute_macd(spy["Close"], 12, 26, 9)
        spy["spy_macd_hist"]      = spy_mh
        _, _, _spy_bb_pct, _      = compute_bollinger_bands(spy["Close"], 20, 2.0)
        spy["spy_bb_pct"]         = _spy_bb_pct
        sma20                     = spy["Close"].rolling(20).mean()
        spy["spy_price_vs_sma20"] = (spy["Close"] / sma20 - 1.0).fillna(0.0)
        vol_avg                   = spy["Volume"].rolling(20).mean().replace(0, np.nan)
        spy["spy_volume_ratio"]   = (spy["Volume"] / vol_avg).fillna(1.0)

        spy_cols = ["spy_rsi_14", "spy_macd_hist", "spy_bb_pct",
                    "spy_price_vs_sma20", "spy_volume_ratio"]
        spy_feat = spy[spy_cols]

        # Reindex spy features onto df's index via forward-fill (no lookahead)
        combined = spy_feat.reindex(spy_feat.index.union(df.index)).ffill()
        spy_aligned = combined.reindex(df.index)

        for col in spy_cols:
            df[col] = spy_aligned[col].values

        # Lagged 6h returns of SPY (lead-lag IC: SPY->MRVL -0.084, SPY->TSLA -0.069)
        spy_ret6 = spy["Close"].pct_change(6)
        for lag in [1, 2, 3]:
            feat = f"spy_ret6_lag{lag}"
            combined_r = spy_ret6.shift(lag).reindex(
                spy_ret6.index.union(df.index)
            ).ffill().reindex(df.index)
            df[feat] = combined_r.values

    except Exception:
        pass
    return df


def add_peer_features(
    df: pd.DataFrame,
    nvda_hourly_df: pd.DataFrame = None,
    xlk_hourly_df: pd.DataFrame = None,
    vix_hourly_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Merge hourly RSI_14 and lagged 6h returns of sector peers into df.

    Lead-lag IC analysis (2y hourly):
    - nvda_rsi_14 (lag 6h): IC=-0.086 for APP
    - xlk_rsi_14 (lag 1h):  IC=-0.061 for SNOW
    - vix_rsi_14 (lag 1h):  IC=+0.062 for MRVL
    - nvda_ret6_lag1:        IC=-0.053 for TSLA, IC=+0.049 for AMD
    """
    peers = [
        ("nvda_rsi_14", nvda_hourly_df),
        ("xlk_rsi_14",  xlk_hourly_df),
        ("vix_rsi_14",  vix_hourly_df),
    ]
    for col_name, peer_df in peers:
        if peer_df is None or len(peer_df) < 20:
            continue
        try:
            peer = peer_df.copy()
            peer_rsi = compute_rsi(peer["Close"], 14)
            peer_rsi.index = peer.index
            peer_aligned = peer_rsi.reindex(
                peer_rsi.index.union(df.index)
            ).ffill().reindex(df.index)
            df[col_name] = peer_aligned.values
        except Exception:
            pass

    # Lagged 6h returns for NVDA (semiconductor lead signal)
    if nvda_hourly_df is not None and len(nvda_hourly_df) >= 20:
        try:
            nvda_ret6 = nvda_hourly_df["Close"].pct_change(6)
            for lag in [1, 2, 3]:
                feat = f"nvda_ret6_lag{lag}"
                aligned = nvda_ret6.shift(lag).reindex(
                    nvda_ret6.index.union(df.index)
                ).ffill().reindex(df.index)
                df[feat] = aligned.values
        except Exception:
            pass

    return df


def add_cross_asset_features(
    hourly_df: pd.DataFrame,
    tlt_daily_df: pd.DataFrame = None,
    hyg_daily_df: pd.DataFrame = None,
    gld_daily_df: pd.DataFrame = None,
    uup_daily_df: pd.DataFrame = None,
    vix9d_daily_df: pd.DataFrame = None,
    vix3m_daily_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Merge cross-asset macro regime features into hourly bars (1-day lag, no lookahead).

    TLT: 20yr treasuries — flight-to-safety signal
    HYG: high-yield bonds — credit stress / risk appetite
    GLD: gold — fear / inflation hedge
    UUP: dollar ETF — dollar strength (risk-off for equities)
    VIX9D/VIX3M: term structure — proxy for dealer gamma environment

    New columns (prefix ca_):
        ca_tlt_ret5         TLT 5-day return
        ca_tlt_vs_ma20      TLT deviation from its 20-day MA
        ca_hyg_ret5         HYG 5-day return
        ca_hyg_tlt_spr      HYG ret5 minus TLT ret5 (credit appetite spread)
        ca_gld_ret5         GLD 5-day return
        ca_uup_ret5         UUP 5-day return
        ca_risk_score       composite risk-on/off (-4 … +4)
        ca_vts9d_ratio      VIX9D / VIX  (>1 = near-term panic = trending regime)
        ca_vts3m_ratio      VIX / VIX3M  (>1 = backwardation = fear premium)
        ca_vts_regime       0 = calm/ranging  1 = normal  2 = panic/trending
    """
    import bisect

    def _to_date(ts):
        try:
            return pd.Timestamp(ts).date()
        except Exception:
            return None

    def _make_lookup(daily_df):
        if daily_df is None or len(daily_df) < 25:
            return [], {}
        close = daily_df["Close"]
        ma20  = close.rolling(20, min_periods=5).mean()
        ret5  = close.pct_change(5).fillna(0.0)
        vs_ma20 = (close / ma20 - 1.0).fillna(0.0)
        raw = {}
        for i, ts in enumerate(daily_df.index):
            d = _to_date(ts)
            if d is not None:
                raw[d] = {"ret5": float(ret5.iloc[i]), "vs_ma20": float(vs_ma20.iloc[i])}
        sorted_raw = sorted(raw.keys())
        # 1-day shift: bar on day D sees features computed on day D-1
        shifted_dates = sorted_raw[1:]
        shifted = {shifted_dates[i]: raw[sorted_raw[i]] for i in range(len(shifted_dates))}
        return sorted(shifted.keys()), shifted

    def _make_vix_lookup(vix_df):
        if vix_df is None or len(vix_df) < 10:
            return [], {}
        raw = {}
        for ts, v in zip(vix_df.index, vix_df["Close"].values):
            d = _to_date(ts)
            if d is not None:
                raw[d] = float(v)
        sorted_raw = sorted(raw.keys())
        shifted_dates = sorted_raw[1:]
        shifted = {shifted_dates[i]: raw[sorted_raw[i]] for i in range(len(shifted_dates))}
        return sorted(shifted.keys()), shifted

    tlt_dates,  tlt_lkp  = _make_lookup(tlt_daily_df)
    hyg_dates,  hyg_lkp  = _make_lookup(hyg_daily_df)
    gld_dates,  gld_lkp  = _make_lookup(gld_daily_df)
    uup_dates,  uup_lkp  = _make_lookup(uup_daily_df)
    vix9_dates, vix9_lkp = _make_vix_lookup(vix9d_daily_df)
    vix3_dates, vix3_lkp = _make_vix_lookup(vix3m_daily_df)

    n = len(hourly_df)
    ca_tlt_ret5    = np.zeros(n)
    ca_tlt_vs_ma20 = np.zeros(n)
    ca_hyg_ret5    = np.zeros(n)
    ca_hyg_tlt_spr = np.zeros(n)
    ca_gld_ret5    = np.zeros(n)
    ca_uup_ret5    = np.zeros(n)
    ca_risk_score  = np.zeros(n)
    ca_vts9d_ratio = np.ones(n)   # 1.0 = neutral term structure
    ca_vts3m_ratio = np.ones(n)
    ca_vts_regime  = np.ones(n)   # 1 = normal

    # Reuse d_vix (already merged by add_daily_trend_features) as the VIX base
    d_vix_col = hourly_df["d_vix"].values if "d_vix" in hourly_df.columns else None

    for i, ts in enumerate(hourly_df.index):
        d = _to_date(ts)
        if d is None:
            continue

        tlt_r5 = hyg_r5 = gld_r5 = uup_r5 = tlt_ma20 = 0.0

        if tlt_dates:
            pos = bisect.bisect_right(tlt_dates, d) - 1
            if pos >= 0:
                r = tlt_lkp[tlt_dates[pos]]
                tlt_r5  = r["ret5"]
                tlt_ma20 = r["vs_ma20"]

        if hyg_dates:
            pos = bisect.bisect_right(hyg_dates, d) - 1
            if pos >= 0:
                hyg_r5 = hyg_lkp[hyg_dates[pos]]["ret5"]

        if gld_dates:
            pos = bisect.bisect_right(gld_dates, d) - 1
            if pos >= 0:
                gld_r5 = gld_lkp[gld_dates[pos]]["ret5"]

        if uup_dates:
            pos = bisect.bisect_right(uup_dates, d) - 1
            if pos >= 0:
                uup_r5 = uup_lkp[uup_dates[pos]]["ret5"]

        ca_tlt_ret5[i]    = tlt_r5
        ca_tlt_vs_ma20[i] = tlt_ma20
        ca_hyg_ret5[i]    = hyg_r5
        ca_hyg_tlt_spr[i] = hyg_r5 - tlt_r5   # positive = credit appetite / risk-on
        ca_gld_ret5[i]    = gld_r5
        ca_uup_ret5[i]    = uup_r5
        # Composite: each risk-off asset adds -1; each risk-on asset adds +1
        ca_risk_score[i]  = float(
            -np.sign(tlt_r5) + np.sign(hyg_r5) - np.sign(gld_r5) - np.sign(uup_r5)
        )

        # VIX term structure — proxy for dealer gamma environment
        vix_base = float(d_vix_col[i]) * 30.0 if d_vix_col is not None else 20.0

        vix9d_val = vix3m_val = None
        if vix9_dates:
            pos = bisect.bisect_right(vix9_dates, d) - 1
            if pos >= 0:
                vix9d_val = vix9_lkp[vix9_dates[pos]]
        if vix3_dates:
            pos = bisect.bisect_right(vix3_dates, d) - 1
            if pos >= 0:
                vix3m_val = vix3_lkp[vix3_dates[pos]]

        if vix9d_val is not None and vix_base > 0:
            ca_vts9d_ratio[i] = vix9d_val / vix_base
        if vix3m_val is not None and vix_base > 0:
            ca_vts3m_ratio[i] = vix_base / vix3m_val

        # Regime classification:
        # 2 = backwardation / panic / trending (momentum signals preferred)
        # 0 = deep contango / calm / ranging (mean-reversion signals preferred)
        # 1 = normal
        r9 = ca_vts9d_ratio[i]
        r3 = ca_vts3m_ratio[i]
        if r9 > 1.10 or r3 > 1.05:
            ca_vts_regime[i] = 2.0
        elif r9 < 0.85 and r3 < 0.95:
            ca_vts_regime[i] = 0.0

    result = hourly_df.copy()
    result["ca_tlt_ret5"]    = ca_tlt_ret5
    result["ca_tlt_vs_ma20"] = ca_tlt_vs_ma20
    result["ca_hyg_ret5"]    = ca_hyg_ret5
    result["ca_hyg_tlt_spr"] = ca_hyg_tlt_spr
    result["ca_gld_ret5"]    = ca_gld_ret5
    result["ca_uup_ret5"]    = ca_uup_ret5
    result["ca_risk_score"]  = ca_risk_score
    result["ca_vts9d_ratio"] = ca_vts9d_ratio
    result["ca_vts3m_ratio"] = ca_vts3m_ratio
    result["ca_vts_regime"]  = ca_vts_regime
    return result


def add_pead_features(hourly_df: pd.DataFrame, daily_df: pd.DataFrame,
                      gap_threshold: float = 0.03) -> pd.DataFrame:
    """
    Post-Earnings Announcement Drift (PEAD) proxy.

    Uses overnight gap returns on daily data to detect earnings-like events.
    After a large gap, the drift tends to continue for 10-20 days (PEAD anomaly).

    Features:
        pead_gap_score    gap return at most recent detected earnings event
        pead_days_since   calendar days since event (capped at 30)
        pead_bull_active  1 if within 20d of positive gap > threshold
        pead_bear_active  1 if within 10d of negative gap < -threshold
        pead_momentum     signed gap * exponential decay (half-life 5d)
    """
    import bisect

    cols = ["pead_gap_score", "pead_days_since", "pead_bull_active",
            "pead_bear_active", "pead_momentum",
            "pead_phase_early", "pead_phase_drift", "pead_drift_score"]

    def _to_date(ts):
        try:
            return pd.Timestamp(ts).date()
        except Exception:
            return None

    n = len(hourly_df)
    pead_gap_score    = np.zeros(n)
    pead_days_since   = np.full(n, 30.0)
    pead_bull_active  = np.zeros(n)
    pead_bear_active  = np.zeros(n)
    pead_momentum     = np.zeros(n)
    pead_phase_early  = np.zeros(n)   # days 0-6: initial reaction / momentum phase
    pead_phase_drift  = np.zeros(n)   # days 7-20: PEAD drift window (IC=0.19-0.23)
    pead_drift_score  = np.zeros(n)   # signed gap × I(7 <= days <= 20)

    if daily_df is None or len(daily_df) < 20:
        result = hourly_df.copy()
        for c in cols:
            result[c] = 0.0
        return result

    daily_close = daily_df["Close"]
    daily_open  = daily_df["Open"] if "Open" in daily_df.columns else daily_close
    overnight_gap = (daily_open - daily_close.shift(1)) / daily_close.shift(1)

    # Collect all gap events exceeding the threshold
    events: dict = {}
    for ts, gap in zip(daily_df.index, overnight_gap.values):
        if np.isnan(gap) or abs(gap) < gap_threshold:
            continue
        d = _to_date(ts)
        if d:
            events[d] = float(gap)

    if not events:
        result = hourly_df.copy()
        for c in cols:
            result[c] = 0.0
        return result

    sorted_dates = sorted(events.keys())

    for i, ts in enumerate(hourly_df.index):
        d = _to_date(ts)
        if d is None:
            continue
        pos = bisect.bisect_right(sorted_dates, d) - 1
        if pos < 0:
            continue
        event_d = sorted_dates[pos]
        days = (d - event_d).days
        gap   = events[event_d]

        pead_gap_score[i]  = gap
        pead_days_since[i] = min(float(days), 30.0)

        if gap > gap_threshold and days <= 20:
            pead_bull_active[i] = 1.0
        if gap < -gap_threshold and days <= 10:
            pead_bear_active[i] = 1.0

        decay = 0.5 ** (days / 5.0)
        pead_momentum[i] = gap * decay if days <= 20 else 0.0

        # Phase encoding: explicitly mark the PEAD drift window (days 7-20 post-event)
        # Research IC: AMD 10-15d=+0.215, 15-20d=+0.234; GOOGL 15-20d=+0.218; APP 15-20d=+0.190
        if days <= 6:
            pead_phase_early[i] = 1.0
        if 7 <= days <= 20:
            pead_phase_drift[i] = 1.0
            pead_drift_score[i] = gap  # signed: positive gap → positive drift expected

    result = hourly_df.copy()
    result["pead_gap_score"]    = pead_gap_score
    result["pead_days_since"]   = pead_days_since
    result["pead_bull_active"]  = pead_bull_active
    result["pead_bear_active"]  = pead_bear_active
    result["pead_momentum"]     = pead_momentum
    result["pead_phase_early"]  = pead_phase_early
    result["pead_phase_drift"]  = pead_phase_drift
    result["pead_drift_score"]  = pead_drift_score
    return result


# ------------------------------------------------------------------ #
# Macro calendar event features                                       #
# ------------------------------------------------------------------ #

# FOMC decision dates — 8 per year, Wednesdays at 2 PM ET.
# IC: is_fomc_day IC=+0.059 for APP; fomc_post_decision IC=+0.058 for APP (lift=+4.4%)
_FOMC_DATES = frozenset([
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-07", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
])

# NFP (Non-Farm Payrolls) release dates — first Friday of month, 8:30 AM ET pre-market.
# IC: is_nfp_day IC=+0.059 for AMD (lift=+1.9%), +0.053 for SNOW
_NFP_DATES = frozenset([
    "2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05",
    "2024-05-03", "2024-06-07", "2024-07-05", "2024-08-02",
    "2024-09-06", "2024-10-04", "2024-11-01", "2024-12-06",
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-01", "2026-06-05", "2026-07-10", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06",
])


def add_macro_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add FOMC and NFP calendar event flags to hourly df.

    Features:
    - is_fomc_day:         1 on FOMC decision day (IC=+0.059 for APP)
    - fomc_post_decision:  1 on FOMC day after 18:00 UTC (2 PM ET decision; IC=+0.058)
    - is_fomc_next:        1 on day after FOMC (IC=+0.063 for TSLA)
    - is_nfp_day:          1 on NFP release day (IC=+0.059 for AMD)
    """
    from datetime import timedelta

    dates_arr = pd.to_datetime(df.index).date
    date_strs = [str(d) for d in dates_arr]
    hour_utc = pd.to_datetime(df.index).hour

    fomc_today = np.array([1 if s in _FOMC_DATES else 0 for s in date_strs], dtype=np.float32)
    nfp_today  = np.array([1 if s in _NFP_DATES  else 0 for s in date_strs], dtype=np.float32)

    from datetime import date as _date
    fomc_next  = np.array([
        1 if str(d + timedelta(days=1)) in _FOMC_DATES else 0
        for d in dates_arr
    ], dtype=np.float32)

    # Post-Fed-decision: FOMC day bars at or after 18:00 UTC (≈2 PM ET decision time)
    fomc_post = ((fomc_today == 1) & (hour_utc >= 18)).astype(np.float32)

    df = df.copy()
    df["is_fomc_day"]        = fomc_today
    df["fomc_post_decision"] = fomc_post
    df["is_fomc_next"]       = fomc_next
    df["is_nfp_day"]         = nfp_today
    return df


# Maps individual tickers to their primary sector ETF
SECTOR_ETF_MAP = {
    # Technology
    "AMD": "XLK", "ANET": "XLK", "APP": "XLK", "AVGO": "XLK",
    "LRCX": "XLK", "MU": "XLK", "MSFT": "XLK", "NVDA": "XLK",
    "AAPL": "XLK", "GOOG": "XLK", "META": "XLK", "CRM": "XLK",
    "ADBE": "XLK", "NOW": "XLK", "SNOW": "XLK", "CRWD": "XLK",
    "PANW": "XLK", "KLAC": "XLK", "AMAT": "XLK", "MRVL": "XLK",
    # Industrials
    "GE": "XLI", "CAT": "XLI", "HON": "XLI", "DE": "XLI",
    # Communications
    "T": "XLC", "NFLX": "XLC", "DIS": "XLC",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF",
    # Energy
    "XOM": "XLE", "CVX": "XLE",
    # Health Care
    "UNH": "XLV", "LLY": "XLV", "ABBV": "XLV",
    # Consumer Discretionary
    "AMZN": "XLY", "TSLA": "XLY",
}


def add_sector_features(df: pd.DataFrame, ticker: str, sector_df: pd.DataFrame) -> pd.DataFrame:
    """Add sector-relative strength features.

    Compares stock's return vs sector ETF return over multiple windows.
    Outperforming stocks tend to continue outperforming (momentum effect).
    Features added: sector_rs_5, sector_rs_20, sector_rs_slope.
    """
    if sector_df is None or len(sector_df) < 25:
        return df
    try:
        # Align sector close to df's index via forward-fill
        sec_close = sector_df["Close"].reindex(
            sector_df.index.union(df.index)
        ).ffill().reindex(df.index)

        stock_ret5 = df["Close"].pct_change(5).fillna(0.0)
        sec_ret5 = sec_close.pct_change(5).fillna(0.0)
        stock_ret20 = df["Close"].pct_change(20).fillna(0.0)
        sec_ret20 = sec_close.pct_change(20).fillna(0.0)

        # RS = stock excess return vs sector (positive = outperforming)
        df["sector_rs_5"] = (stock_ret5 - sec_ret5).fillna(0.0)
        df["sector_rs_20"] = (stock_ret20 - sec_ret20).fillna(0.0)

        # RS slope: momentum of relative strength (rising RS is bullish)
        rs_ratio = (df["Close"] / sec_close.replace(0, np.nan)).ffill()
        df["sector_rs_slope"] = rs_ratio.pct_change(10).fillna(0.0)
    except Exception:
        pass
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features with confirmed IC:
      is_fri        — Friday (weekly OpEx drift, avg IC=+0.097 across portfolio)
      is_wed        — Wednesday (mid-week lull, avg IC=-0.076)
      is_thu        — Thursday (pre-OpEx squaring, avg IC=-0.044)
      is_tue        — Tuesday (avg IC=+0.056 for half the portfolio)
      is_mon        — Monday (mild negative IC)
      opex_monthly  — 3rd Friday of each month (monthly OpEx, avg IC=+0.056)
      opex_thursday — Thursday before monthly OpEx (avg IC=-0.030)

    Circular seasonal encoding (continuous, year-wrap safe):
      week_sin/cos  — sin/cos(2π×week/52): TSLA IC=+0.078, GOOGL IC=+0.054
      month_sin/cos — sin/cos(2π×month/12): April=+0.125, Feb=-0.156 avg class

    Seasonality bins (confirmed by SHAP ranking and IC analysis):
      is_q_earnings_month — month in {1,4,7,10}: earnings season, avg +0.08 delta
      is_close_hour       — last bar of trading day (20h UTC): avg -0.054 delta
    """
    df = df.copy()
    dt_index  = pd.to_datetime(df.index)
    dayofweek = dt_index.dayofweek   # Mon=0, Fri=4
    day       = dt_index.day
    month     = dt_index.month
    hour      = dt_index.hour

    df["is_mon"] = (dayofweek == 0).astype(np.int8)
    df["is_tue"] = (dayofweek == 1).astype(np.int8)
    df["is_wed"] = (dayofweek == 2).astype(np.int8)
    df["is_thu"] = (dayofweek == 3).astype(np.int8)
    df["is_fri"] = (dayofweek == 4).astype(np.int8)

    is_friday     = dayofweek == 4
    is_thursday   = dayofweek == 3
    df["opex_monthly"]  = ((is_friday)   & (day >= 15) & (day <= 21)).astype(np.int8)
    df["opex_thursday"] = ((is_thursday) & (day >= 14) & (day <= 20)).astype(np.int8)

    # Week-of-year circular encoding: week 52 and week 1 are adjacent
    week = dt_index.isocalendar().week.astype(int).values
    df["week_sin"] = np.sin(2.0 * np.pi * week / 52.0).astype(np.float32)
    df["week_cos"] = np.cos(2.0 * np.pi * week / 52.0).astype(np.float32)

    # Month circular encoding
    df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0).astype(np.float32)
    df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0).astype(np.float32)

    # Earnings season months: Jan(Q4), Apr(Q1), Jul(Q2), Oct(Q3) results
    df["is_q_earnings_month"] = month.isin([1, 4, 7, 10]).astype(np.int8)

    # Close-hour sell-off: last bar of session (20h UTC = 4PM ET)
    df["is_close_hour"] = (hour == 20).astype(np.int8)

    return df


def add_cross_sectional_features(
    df: pd.DataFrame,
    ticker: str,
    peer_close_dict: dict,
) -> pd.DataFrame:
    """
    Add cross-sectional rank features relative to portfolio peers.

    Confirmed signals (p<0.05 for 3+ tickers):
      cs_rank_5d    — rank of 5-day trailing return vs peers (0=worst, 1=best)
                      Mean-reversion for 5/6 tickers (avg IC=-0.021)
                      Momentum for APP (IC=+0.055)
      cs_rank_10d   — rank of 10-day return vs peers (avg IC=-0.016)
      cs_rank_1d    — rank of 1-day return vs peers (avg IC=-0.013)
      peer_spread_5d — ticker's 5d return minus average peer 5d return
                       XGBoost can learn direction per ticker

    peer_close_dict: {symbol: pd.Series of hourly Close prices}
                     All other portfolio tickers at the same interval/period.
    """
    if not peer_close_dict:
        return df

    df = df.copy()
    try:
        # Build close panel: this ticker + all peers
        close_parts = {ticker: df["Close"]}
        close_parts.update(peer_close_dict)

        close_panel = pd.concat(
            [s.rename(t) for t, s in close_parts.items()],
            axis=1
        ).sort_index().ffill()

        n_tickers = len(close_panel.columns)
        if n_tickers < 2:
            return df

        # Trailing returns (hourly bars per trading day ≈ 7)
        ret_1d  = close_panel.pct_change(7)
        ret_5d  = close_panel.pct_change(35)
        ret_10d = close_panel.pct_change(70)

        def _rank_col(panel: pd.DataFrame, col: str) -> pd.Series:
            ranked = panel.rank(axis=1, method="average", na_option="keep")
            return (ranked[col] - 1.0) / max(n_tickers - 1, 1)

        cs_1d  = _rank_col(ret_1d,  ticker)
        cs_5d  = _rank_col(ret_5d,  ticker)
        cs_10d = _rank_col(ret_10d, ticker)

        peer_cols    = [t for t in ret_5d.columns if t != ticker]
        peer_avg_5d  = ret_5d[peer_cols].mean(axis=1)
        spread_5d    = ret_5d[ticker] - peer_avg_5d

        idx = df.index
        df["cs_rank_1d"]    = cs_1d.reindex(idx).ffill().values
        df["cs_rank_5d"]    = cs_5d.reindex(idx).ffill().values
        df["cs_rank_10d"]   = cs_10d.reindex(idx).ffill().values
        df["peer_spread_5d"] = spread_5d.reindex(idx).ffill().values

    except Exception:
        pass

    # Lagged 6h returns of each named peer ticker (lead-lag signals, IC up to 0.087)
    # APP->GOOGL: IC=+0.087 at 1h lag | MRVL->AMD: IC=+0.057 avg | SNOW->TSLA: IC=-0.077
    for sym, peer_close in peer_close_dict.items():
        if peer_close is None:
            continue
        try:
            peer_ret6 = peer_close.pct_change(6)
            sym_clean = sym.lower().replace("^", "").replace("-", "")
            for lag in [1, 2, 3]:
                feat = f"{sym_clean}_ret6_lag{lag}"
                aligned = peer_ret6.shift(lag).reindex(
                    peer_ret6.index.union(df.index)
                ).ffill().reindex(df.index)
                df[feat] = aligned.values
        except Exception:
            pass

    return df


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bar structure features measuring HOW price moved within each bar.
    Confirmed via IC research (regime_conditional):
      buying_pressure  avg IC=-0.024 overall, -0.049 in CHOPPY (mean-reversion)
      tick_imbal_z     avg IC=-0.025 overall, -0.077 in CHOPPY
      momentum_efficiency_10  avg IC=-0.018, -0.064 in CHOPPY
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    open_ = df["Open"]
    rng   = (high - low).replace(0, np.nan)

    df = df.copy()
    df["buying_pressure"]  = (close - low)         / rng        # 0=closed at low, 1=at high
    df["bar_efficiency"]   = (close - open_).abs() / rng        # 0=doji, 1=perfectly directional
    df["wick_imbalance"]   = (
        (np.minimum(open_, close) - low) - (high - np.maximum(open_, close))
    ) / rng                                                      # positive=bullish wick bias

    abs_moves  = close.diff().abs()
    total_path = abs_moves.rolling(10).sum().replace(0, np.nan)
    net_move   = close.diff(10).abs()
    df["momentum_efficiency_10"] = net_move / total_path         # 0=zigzag, 1=clean trend

    if "Volume" in df.columns:
        ti     = (df["buying_pressure"] - 0.5) * df["Volume"]   # signed estimated buy volume
        ti_std = ti.rolling(20).std().replace(0, np.nan)
        df["tick_imbal_z"] = ti / ti_std

    return df


def _rolling_r2_fast(close_vals: np.ndarray, window: int) -> np.ndarray:
    """Vectorized R² of linear fit on rolling window (no scipy required)."""
    n = len(close_vals)
    result = np.full(n, np.nan)
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    xc = x - x_mean
    x_ss = (xc ** 2).sum()
    for i in range(window - 1, n):
        y = close_vals[i - window + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        y_mean = y.mean()
        yc = y - y_mean
        ss_tot = (yc ** 2).sum()
        if ss_tot < 1e-12:
            result[i] = 1.0
            continue
        slope = np.dot(xc, yc) / x_ss
        y_fit = slope * x + (y_mean - slope * x_mean)
        ss_res = ((y - y_fit) ** 2).sum()
        result[i] = 1.0 - ss_res / ss_tot
    return result


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market regime detection and regime×signal interaction features.
    Regime research (71.7% NEUTRAL, 25.5% TRENDING, 2.8% CHOPPY):
      In CHOPPY: pvd_10 IC=+0.117, tick_imbal_z IC=-0.077, opex_monthly IC=+0.086.
    Interaction features teach XGBoost to apply signals conditionally per regime.
    """
    if "ADX" not in df.columns:
        return df

    df = df.copy()
    adx  = df["ADX"].values
    r2   = _rolling_r2_fast(df["Close"].values, 20)

    # Regime indicator flags
    is_trending = ((adx > 25.0) & (r2 > 0.65)).astype(np.float32)
    is_choppy   = ((adx < 15.0) & (r2 < 0.30)).astype(np.float32)

    df["is_trending"]  = is_trending
    df["is_choppy"]    = is_choppy
    df["regime_score"] = adx * np.nan_to_num(r2, nan=0.0)   # continuous strength

    # Interaction features (regime × signal — the key alpha unlock)
    if "pvd_10" in df.columns:
        df["pvd_choppy"]   = df["pvd_10"]    * is_choppy
        df["pvd_trending"] = df["pvd_10"]    * is_trending
    if "tick_imbal_z" in df.columns:
        df["tick_choppy"]  = df["tick_imbal_z"] * is_choppy
    if "opex_monthly" in df.columns:
        df["opex_choppy"]  = df["opex_monthly"] * is_choppy

    return df
