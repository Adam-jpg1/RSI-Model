import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from data.fetcher import get_price_data, get_daily_data
from data.indicators import (add_all_indicators, add_daily_trend_features,
                              add_market_features, add_earnings_features,
                              add_cross_asset_features, add_pead_features,
                              add_peer_features, add_macro_event_features,
                              add_calendar_features, add_cross_sectional_features,
                              add_microstructure_features, add_regime_features)
from ml.features import (get_feature_matrix, prepare_dataset, target_bars_for_interval,
                          build_target, compute_class_weights)
from ml.model import RSIModel
from backtest.engine import _run_simulation, print_backtest_report
from config import CONFIG


def _build_spy_signals(period: str = "2y", interval: str = "1h") -> dict:
    """
    Compute a simple SPY intraday trend signal for each hourly bar.
    Returns {iso_timestamp_str: "UP" | "DOWN" | "FLAT"}.
    Signal is based on SPY close vs 20-bar SMA + RSI(14).
    """
    try:
        spy = get_price_data("SPY", period=period, interval=interval)
        spy = add_all_indicators(spy)
    except Exception:
        return {}

    sma20 = spy["Close"].rolling(20).mean()
    rsi14 = spy.get("RSI_14", pd.Series(50.0, index=spy.index))

    result = {}
    for ts in spy.index:
        close = float(spy.loc[ts, "Close"])
        sma = float(sma20.loc[ts]) if not pd.isna(sma20.loc[ts]) else close
        rsi = float(rsi14.loc[ts]) if not pd.isna(rsi14.loc[ts]) else 50.0
        ts_str = str(ts)
        if close > sma and rsi > 50:
            result[ts_str] = "UP"
        elif close < sma and rsi < 50:
            result[ts_str] = "DOWN"
        else:
            result[ts_str] = "FLAT"
    return result


def _build_daily_signals(daily_df: pd.DataFrame, train_end_date, xgb_params: dict = None,
                         n_days_ahead: int = 5) -> dict:
    """
    Train a daily RSIModel and return a date→signal dict for all subsequent dates.
    Signals are shifted 1 day to avoid lookahead (signal from day D applied on day D+1).

    n_days_ahead=5 (1 week) predicts multi-day regime direction rather than
    next-day noise — better suited for filtering intraday entries.
    """
    if daily_df is None or len(daily_df) < 60:
        return {}

    if hasattr(train_end_date, "date"):
        cutoff = train_end_date.date()
    else:
        cutoff = pd.Timestamp(str(train_end_date)[:10]).date()

    daily_train = daily_df[pd.DatetimeIndex(daily_df.index).date <= cutoff]
    if len(daily_train) < 50:
        return {}

    features_d = get_feature_matrix(daily_train, sentiment_score=0.0).dropna()
    labels_d = build_target(daily_train, n_bars=n_days_ahead)
    valid_idx = features_d.index.intersection(labels_d.dropna().index)
    features_d = features_d.loc[valid_idx]
    labels_d = labels_d.loc[valid_idx]

    if len(features_d) < 40:
        return {}

    orig_xgb = None
    if xgb_params:
        orig_xgb = {k: v for k, v in CONFIG.ml.xgb_params.items()}
        CONFIG.ml.xgb_params.update(xgb_params)

    # Early stopping for the daily model — same as hourly
    X_val_es = y_val_es = None
    if len(features_d) > 100:
        val_cut = int(len(features_d) * 0.85)
        X_val_es = features_d.iloc[val_cut:]
        y_val_es = labels_d.iloc[val_cut:]
        features_d = features_d.iloc[:val_cut]
        labels_d = labels_d.iloc[:val_cut]

    daily_model = RSIModel()
    daily_model.build(early_stopping_rounds=15)
    daily_model.fit(features_d, labels_d, X_val=X_val_es, y_val=y_val_es)

    if orig_xgb is not None:
        CONFIG.ml.xgb_params.clear()
        CONFIG.ml.xgb_params.update(orig_xgb)

    features_all = get_feature_matrix(daily_df, sentiment_score=0.0).dropna()
    try:
        preds_raw = {}
        for tidx, trow in features_all.iterrows():
            try:
                pred = daily_model.predict_single(trow)
                preds_raw[str(tidx)[:10]] = pred["signal"]
            except Exception:
                pass
    except Exception:
        return {}

    sorted_dates = sorted(preds_raw.keys())
    shifted = {}
    for i in range(1, len(sorted_dates)):
        shifted[sorted_dates[i]] = preds_raw[sorted_dates[i - 1]]

    return shifted


def _build_daily_signals_with_conf(daily_df: pd.DataFrame, train_end_date,
                                   xgb_params: dict = None, n_days_ahead: int = 5) -> dict:
    """Like _build_daily_signals but returns {date: {"signal": str, "conf": float}}."""
    if daily_df is None or len(daily_df) < 60:
        return {}

    if hasattr(train_end_date, "date"):
        cutoff = train_end_date.date()
    else:
        cutoff = pd.Timestamp(str(train_end_date)[:10]).date()

    daily_train = daily_df[pd.DatetimeIndex(daily_df.index).date <= cutoff]
    if len(daily_train) < 50:
        return {}

    features_d = get_feature_matrix(daily_train, sentiment_score=0.0).dropna()
    labels_d = build_target(daily_train, n_bars=n_days_ahead)
    valid_idx = features_d.index.intersection(labels_d.dropna().index)
    features_d = features_d.loc[valid_idx]
    labels_d = labels_d.loc[valid_idx]
    if len(features_d) < 40:
        return {}

    orig_xgb = None
    if xgb_params:
        orig_xgb = {k: v for k, v in CONFIG.ml.xgb_params.items()}
        CONFIG.ml.xgb_params.update(xgb_params)

    X_val_es = y_val_es = None
    if len(features_d) > 100:
        val_cut = int(len(features_d) * 0.85)
        X_val_es = features_d.iloc[val_cut:]
        y_val_es = labels_d.iloc[val_cut:]
        features_d = features_d.iloc[:val_cut]
        labels_d = labels_d.iloc[:val_cut]

    daily_model = RSIModel()
    daily_model.build(early_stopping_rounds=15)
    daily_model.fit(features_d, labels_d, X_val=X_val_es, y_val=y_val_es)

    if orig_xgb is not None:
        CONFIG.ml.xgb_params.clear()
        CONFIG.ml.xgb_params.update(orig_xgb)

    features_all = get_feature_matrix(daily_df, sentiment_score=0.0).dropna()
    preds_raw = {}
    for tidx, trow in features_all.iterrows():
        try:
            pred = daily_model.predict_single(trow)
            preds_raw[str(tidx)[:10]] = {"signal": pred["signal"], "conf": pred["confidence"]}
        except Exception:
            pass

    sorted_dates = sorted(preds_raw.keys())
    shifted = {}
    for i in range(1, len(sorted_dates)):
        shifted[sorted_dates[i]] = preds_raw[sorted_dates[i - 1]]
    return shifted


def walk_forward_backtest(
    ticker: str,
    n_folds: int = 4,
    period: str = "2y",
    interval: str = "1h",
    adx_filter_min: float = None,
    initial_capital: float = 100_000.0,
    xgb_params: dict = None,
    daily_confirm: bool = False,
    d_trend_gate: bool = False,
    d_trend_min_score: int = 2,
    ignore_flat_exit: bool = False,
    trailing_stop_pct: float = 0.0,
    allow_short: bool = None,
    min_conviction_spread: float = 0.0,
    session_hours: frozenset = None,
    max_hold_bars: int = 0,
    recency_halflife: int = 0,
    spy_align: bool = False,
    atr_tp_mult: float = 0.0,
    atr_sl_mult: float = 0.0,
    min_vol_ratio: float = 0.0,
    vwap_align: bool = False,
    min_signal_bars: int = 1,
    use_spy_context: bool = True,
    use_earnings_features: bool = True,
    reentry_after_tp: bool = False,
    regime_gate: bool = False,
    breakeven_pct: float = 0.0,
    partial_tp_pct: float = 0.0,
    partial_tp_ratio: float = 0.5,
    vol_target_atr: float = 0.0,
    daily_min_conf: float = 0.0,
    use_sector_features: bool = False,
    daily_n_days_ahead: int = 5,
    n_ensemble: int = 1,
    min_confidence: float = None,
) -> Dict:
    """
    Walk-forward validation: train on expanding window, test on next unseen chunk.
    Gives n_folds independent out-of-sample test periods for robust evaluation.
    """
    from ml.optimizer import load_best_params
    best = load_best_params(ticker, daily_confirm=daily_confirm)
    orig_tp = CONFIG.bot.take_profit_pct
    orig_sl = CONFIG.bot.stop_loss_pct
    orig_conf = CONFIG.bot.min_confidence
    if best:
        if xgb_params is None:
            xgb_params = best["xgb_params"]
        if adx_filter_min is None:
            adx_filter_min = best["adx_filter_min"]
        # Apply optimized TP/SL if present in saved params (don't override explicit args)
        if "take_profit" in best and orig_tp == CONFIG.bot.take_profit_pct:
            CONFIG.bot.take_profit_pct = best["take_profit"]
        if "stop_loss" in best and orig_sl == CONFIG.bot.stop_loss_pct:
            CONFIG.bot.stop_loss_pct = best["stop_loss"]
        # Load per-ticker optimal hold bars from params (only if caller didn't specify one)
        if "max_hold_bars" in best and max_hold_bars == 0:
            max_hold_bars = best["max_hold_bars"]
        # Load per-ticker optimal confidence: explicit caller arg takes priority,
        # then saved params, then leave global CONFIG alone.
        if min_confidence is None and "min_confidence" in best:
            min_confidence = best["min_confidence"]
        # Load per-ticker partial TP from params (only if caller didn't enable one)
        if "partial_tp_pct" in best and partial_tp_pct == 0.0:
            partial_tp_pct = best["partial_tp_pct"]
    if min_confidence is not None:
        CONFIG.bot.min_confidence = min_confidence
    if adx_filter_min is None:
        adx_filter_min = 0.0

    # Temporarily apply tuned XGBoost params for all fold models in this run
    orig_xgb = None
    if xgb_params:
        orig_xgb = {k: v for k, v in CONFIG.ml.xgb_params.items()}
        CONFIG.ml.xgb_params.update(xgb_params)

    # Build SPY alignment signals once (not per fold — same SPY data for all)
    spy_sigs = None
    if spy_align and ticker != "SPY":
        try:
            spy_sigs = _build_spy_signals(period=period, interval=interval)
        except Exception:
            pass

    print(f"\n[wf] Fetching {ticker} data ({period}, {interval})...")
    df = get_price_data(ticker, period=period, interval=interval)
    df = add_all_indicators(df)
    daily_df_raw = None
    spy_daily_raw = None
    vix_daily_raw = None
    try:
        # Fetch 5yr of daily data when using daily_confirm (needs more history for model training)
        # otherwise 2yr matches the hourly window and is sufficient for d_* trend features
        daily_period = "5y" if daily_confirm else period
        daily_df_raw = get_daily_data(ticker, period=daily_period)
        daily_df_raw = add_all_indicators(daily_df_raw)
        # Fetch SPY + VIX daily for regime features
        if ticker != "SPY":
            try:
                spy_daily_raw = get_daily_data("SPY", period=daily_period)
            except Exception:
                spy_daily_raw = None
        try:
            vix_daily_raw = get_daily_data("^VIX", period=daily_period)
        except Exception:
            vix_daily_raw = None
        df = add_daily_trend_features(df, daily_df_raw,
                                      compute_weekly=True,
                                      spy_daily_df=spy_daily_raw,
                                      vix_daily_df=vix_daily_raw)
        # Cross-asset macro regime features
        try:
            ca_assets = {}
            for sym in ["TLT", "HYG", "GLD", "UUP", "^VIX9D", "^VIX3M"]:
                try:
                    ca_assets[sym] = get_daily_data(sym, period=daily_period)
                except Exception:
                    ca_assets[sym] = None
            df = add_cross_asset_features(
                df,
                tlt_daily_df=ca_assets.get("TLT"),
                hyg_daily_df=ca_assets.get("HYG"),
                gld_daily_df=ca_assets.get("GLD"),
                uup_daily_df=ca_assets.get("UUP"),
                vix9d_daily_df=ca_assets.get("^VIX9D"),
                vix3m_daily_df=ca_assets.get("^VIX3M"),
            )
        except Exception:
            pass
        # PEAD proxy
        try:
            df = add_pead_features(df, daily_df_raw)
        except Exception:
            pass
    except Exception:
        pass

    # SPY cross-market context features
    if use_spy_context and ticker != "SPY":
        try:
            spy_raw = get_price_data("SPY", period=period, interval=interval)
            spy_raw = add_all_indicators(spy_raw)
            df = add_market_features(df, spy_raw)
        except Exception:
            pass

    # Earnings proximity features
    if use_earnings_features:
        try:
            from data.fetcher import get_earnings_dates
            edates = get_earnings_dates(ticker)
            df = add_earnings_features(df, edates)
        except Exception:
            pass

    # Sector relative strength features
    if use_sector_features:
        try:
            from data.indicators import add_sector_features, SECTOR_ETF_MAP
            sector_etf = SECTOR_ETF_MAP.get(ticker.upper())
            if sector_etf:
                sector_raw = get_price_data(sector_etf, period=period, interval=interval)
                df = add_sector_features(df, ticker, sector_raw)
        except Exception:
            pass

    # Peer cross-asset RSI features (lagged in get_feature_matrix)
    # IC: nvda_rsi14_lag6 ≈ -0.086 for APP; xlk_rsi14_lag1 ≈ -0.061 for SNOW
    try:
        peer_assets = {}
        for sym in ["NVDA", "XLK", "^VIX"]:
            if sym != ticker:
                try:
                    peer_assets[sym] = get_price_data(sym, period=period, interval=interval)
                except Exception:
                    peer_assets[sym] = None
        df = add_peer_features(
            df,
            nvda_hourly_df=peer_assets.get("NVDA"),
            xlk_hourly_df=peer_assets.get("XLK"),
            vix_hourly_df=peer_assets.get("^VIX"),
        )
    except Exception:
        pass

    # Macro calendar event features — FOMC/NFP dates known in advance (zero lookahead)
    # IC: is_fomc_day IC=+0.059 for APP; is_nfp_day IC=+0.059 for AMD
    try:
        df = add_macro_event_features(df)
    except Exception:
        pass
    # Day-of-week and OpEx features — avg IC: Fri=+0.097, Wed=-0.076, Thu=-0.044
    try:
        df = add_calendar_features(df)
    except Exception:
        pass
    try:
        df = add_microstructure_features(df)
    except Exception:
        pass
    try:
        df = add_regime_features(df)
    except Exception:
        pass

    features_full = get_feature_matrix(df, sentiment_score=0.0)
    features_full = features_full.dropna()

    # Align df to the cleaned feature index
    df_aligned = df.loc[features_full.index]

    n = len(features_full)
    fold_size = n // (n_folds + 1)

    all_trades: List[pd.Series] = []
    fold_results = []

    for fold in range(1, n_folds + 1):
        train_end = fold * fold_size
        test_start = train_end
        test_end = min((fold + 1) * fold_size, n)

        X_tr = features_full.iloc[:train_end]
        y_tr = _build_labels(df_aligned.iloc[:train_end], interval=interval,
                             use_first_touch=CONFIG.ml.use_first_touch_target)
        mask = X_tr.notna().all(axis=1) & y_tr.notna()
        X_tr, y_tr = X_tr[mask], y_tr[mask]

        # Class weights ensure UP/DOWN/FLAT each contribute equally regardless of
        # slight imbalance. Without this, small folds default to predicting FLAT.
        sample_weight = compute_class_weights(y_tr)
        if recency_halflife > 0 and len(X_tr) > 1:
            ages = np.arange(len(X_tr) - 1, -1, -1, dtype=float)
            decay = 0.5 ** (1.0 / recency_halflife)
            recency_w = decay ** ages / (decay ** ages).mean()
            sample_weight = sample_weight * recency_w

        # Early stopping: hold out last 15% of training data as a validation set.
        # XGBoost stops adding trees when val-loss stops improving, preventing overfit.
        X_val_es = y_val_es = None
        if len(X_tr) > 150:
            val_cut = int(len(X_tr) * 0.85)
            X_val_es = X_tr.iloc[val_cut:]
            y_val_es = y_tr.iloc[val_cut:]
            X_tr = X_tr.iloc[:val_cut]
            y_tr = y_tr.iloc[:val_cut]
            if sample_weight is not None:
                sample_weight = sample_weight[:val_cut]

        model = RSIModel()
        model.build(early_stopping_rounds=20)
        model.fit(X_tr, y_tr, X_val=X_val_es, y_val=y_val_es,
                  sample_weight=sample_weight)
        for seed_offset in range(1, n_ensemble):
            model.add_ensemble_member(X_tr, y_tr, X_val=X_val_es, y_val=y_val_es,
                                      seed=42 + seed_offset * 7)

        feat_test = features_full.iloc[test_start:test_end]
        df_test = df_aligned.iloc[test_start:test_end]

        # Experiment B: build daily signals gate for this fold's test period
        daily_sigs = None
        if daily_confirm and daily_df_raw is not None:
            train_end_ts = features_full.index[train_end - 1]
            if daily_min_conf > 0:
                daily_sigs = _build_daily_signals_with_conf(daily_df_raw, train_end_ts, xgb_params,
                                                            n_days_ahead=daily_n_days_ahead)
            else:
                daily_sigs = _build_daily_signals(daily_df_raw, train_end_ts, xgb_params,
                                                  n_days_ahead=daily_n_days_ahead)

        _allow_short = CONFIG.bot.allow_short if allow_short is None else allow_short
        result = _run_simulation(df_test, feat_test, model, initial_capital, ticker,
                                 adx_filter_min, allow_short=_allow_short,
                                 daily_signals=daily_sigs,
                                 d_trend_gate=d_trend_gate,
                                 d_trend_min_score=d_trend_min_score,
                                 ignore_flat_exit=ignore_flat_exit,
                                 trailing_stop_pct=trailing_stop_pct,
                                 min_conviction_spread=min_conviction_spread,
                                 session_hours=session_hours,
                                 max_hold_bars=max_hold_bars,
                                 spy_signals=spy_sigs,
                                 atr_tp_mult=atr_tp_mult,
                                 atr_sl_mult=atr_sl_mult,
                                 min_vol_ratio=min_vol_ratio,
                                 vwap_align=vwap_align,
                                 min_signal_bars=min_signal_bars,
                                 reentry_after_tp=reentry_after_tp,
                                 regime_gate=regime_gate,
                                 breakeven_pct=breakeven_pct,
                                 partial_tp_pct=partial_tp_pct,
                                 partial_tp_ratio=partial_tp_ratio,
                                 vol_target_atr=vol_target_atr,
                                 daily_min_conf=daily_min_conf)
        m = result["metrics"]
        fold_results.append(result)

        # Aggregate PnL from both long exits (SELL) and short covers (COVER)
        exits = result["trades"]
        exits = exits[exits["type"].isin(["SELL", "COVER"])] if len(exits) else pd.DataFrame()
        if len(exits):
            all_trades.append(exits["pnl_pct"])

        adx_tag = f"  ADX>{adx_filter_min:.0f}" if adx_filter_min > 0 else ""
        print(f"  Fold {fold}: {test_end-test_start} bars  "
              f"{m['total_trades']} trades  "
              f"{m['win_rate_pct']:.1f}% win  "
              f"{result['total_return_pct']:+.2f}% return{adx_tag}")

    if orig_xgb is not None:
        CONFIG.ml.xgb_params.clear()
        CONFIG.ml.xgb_params.update(orig_xgb)
    CONFIG.bot.take_profit_pct = orig_tp
    CONFIG.bot.stop_loss_pct = orig_sl
    CONFIG.bot.min_confidence = orig_conf

    combined_pnl = pd.concat(all_trades) if all_trades else pd.Series(dtype=float)
    total_trades = sum(r["metrics"]["total_trades"] for r in fold_results)
    total_return = sum(r["total_return_pct"] for r in fold_results)
    avg_win_rate = float(np.mean([r["metrics"]["win_rate_pct"] for r in fold_results
                                  if r["metrics"]["total_trades"] > 0]))

    return {
        "ticker": ticker,
        "n_folds": n_folds,
        "fold_results": fold_results,
        "combined_pnl": combined_pnl,
        "total_trades": total_trades,
        "total_return_pct": round(total_return, 2),
        "avg_win_rate": round(avg_win_rate, 1),
        "adx_filter_min": adx_filter_min,
        "adx_from_optuna": best is not None,
    }


def _build_labels(df_slice: pd.DataFrame, interval: str = "1h",
                  use_first_touch: bool = False) -> pd.Series:
    from ml.features import build_target, build_target_first_touch, target_bars_for_interval
    if use_first_touch:
        n = target_bars_for_interval(interval) * 8  # ~48h lookahead
        return build_target_first_touch(df_slice, max_bars=n)
    return build_target(df_slice, n_bars=target_bars_for_interval(interval))


def yearly_backtest(
    ticker: str,
    period: str = "2y",
    interval: str = "1h",
    adx_filter_min: float = None,
    initial_capital: float = 100_000.0,
    xgb_params: dict = None,
    daily_confirm: bool = False,
    allow_short: bool = None,
    max_hold_bars: int = 0,
) -> Dict:
    """
    Train on all data before each calendar year, then test on that year.
    Returns {year: {return_pct, trades, win_rate, test_bars, train_bars}} or
            {year: {skip: True, reason: str}}.
    """
    from ml.optimizer import load_best_params
    best = load_best_params(ticker, daily_confirm=daily_confirm)
    if best:
        if xgb_params is None:
            xgb_params = best["xgb_params"]
        if adx_filter_min is None:
            adx_filter_min = best["adx_filter_min"]
    if adx_filter_min is None:
        adx_filter_min = 0.0

    orig_xgb = None
    if xgb_params:
        orig_xgb = {k: v for k, v in CONFIG.ml.xgb_params.items()}
        CONFIG.ml.xgb_params.update(xgb_params)

    df = get_price_data(ticker, period=period, interval=interval)
    df = add_all_indicators(df)
    daily_df_raw = None
    spy_daily_raw = None
    vix_daily_raw = None
    try:
        daily_period = "5y" if daily_confirm else period
        daily_df_raw = get_daily_data(ticker, period=daily_period)
        daily_df_raw = add_all_indicators(daily_df_raw)
        if ticker != "SPY":
            try:
                spy_daily_raw = get_daily_data("SPY", period=daily_period)
            except Exception:
                spy_daily_raw = None
        try:
            vix_daily_raw = get_daily_data("^VIX", period=daily_period)
        except Exception:
            vix_daily_raw = None
        df = add_daily_trend_features(df, daily_df_raw,
                                      spy_daily_df=spy_daily_raw,
                                      vix_daily_df=vix_daily_raw)
    except Exception:
        pass
    try:
        peer_assets2 = {}
        for sym in ["NVDA", "XLK", "^VIX"]:
            if sym != ticker:
                try:
                    peer_assets2[sym] = get_price_data(sym, period=period, interval=interval)
                except Exception:
                    peer_assets2[sym] = None
        df = add_peer_features(df,
                               nvda_hourly_df=peer_assets2.get("NVDA"),
                               xlk_hourly_df=peer_assets2.get("XLK"),
                               vix_hourly_df=peer_assets2.get("^VIX"))
    except Exception:
        pass
    try:
        df = add_macro_event_features(df)
    except Exception:
        pass
    try:
        df = add_calendar_features(df)
    except Exception:
        pass
    try:
        df = add_microstructure_features(df)
    except Exception:
        pass
    try:
        df = add_regime_features(df)
    except Exception:
        pass

    features_full = get_feature_matrix(df, sentiment_score=0.0).dropna()
    df_aligned = df.loc[features_full.index]
    tz = features_full.index.tz

    def _ts(dt_str):
        ts = pd.Timestamp(dt_str)
        return ts.tz_localize(tz) if tz and ts.tzinfo is None else ts

    years = sorted(set(features_full.index.year))
    year_results = {}

    for year in years:
        year_start = _ts(f"{year}-01-01")
        year_end = _ts(f"{year}-12-31 23:59:59")

        train_mask = features_full.index < year_start
        test_mask = (features_full.index >= year_start) & (features_full.index <= year_end)

        X_tr = features_full[train_mask]
        df_tr = df_aligned[train_mask]
        X_te = features_full[test_mask]
        df_te = df_aligned[test_mask]

        if len(X_tr) < 300:
            year_results[year] = {"skip": True, "reason": f"only {len(X_tr)} train bars"}
            continue
        if len(X_te) < 50:
            year_results[year] = {"skip": True, "reason": f"only {len(X_te)} test bars"}
            continue

        y_tr = _build_labels(df_tr, interval=interval,
                             use_first_touch=CONFIG.ml.use_first_touch_target)
        valid_mask = X_tr.notna().all(axis=1) & y_tr.notna()
        X_tr_f, y_tr_f = X_tr[valid_mask], y_tr[valid_mask]

        model = RSIModel()
        model.build()
        model.fit(X_tr_f, y_tr_f)

        daily_sigs = None
        if daily_confirm and daily_df_raw is not None and len(X_tr_f) > 0:
            daily_sigs = _build_daily_signals(daily_df_raw, X_tr_f.index[-1], xgb_params)

        _allow_short = CONFIG.bot.allow_short if allow_short is None else allow_short
        result = _run_simulation(
            df_te, X_te, model, initial_capital, ticker,
            adx_filter_min,
            allow_short=_allow_short,
            daily_signals=daily_sigs,
            max_hold_bars=max_hold_bars,
        )

        m = result["metrics"]
        year_results[year] = {
            "return_pct": round(result["total_return_pct"], 2),
            "trades": m["total_trades"],
            "win_rate": round(m["win_rate_pct"], 1),
            "test_bars": len(X_te),
            "train_bars": len(X_tr),
        }

    if orig_xgb is not None:
        CONFIG.ml.xgb_params.clear()
        CONFIG.ml.xgb_params.update(orig_xgb)

    return year_results


def bootstrap_confidence(pnl_list: pd.Series, n_bootstrap: int = 10_000) -> Dict:
    """Bootstrap 95% CIs on trade expectancy and win rate."""
    if len(pnl_list) < 3:
        return {"error": f"Need ≥3 trades, got {len(pnl_list)}"}

    trades = pnl_list.values.astype(float)
    rng = np.random.default_rng(42)

    boot_means = np.zeros(n_bootstrap)
    boot_win_rates = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(trades, size=len(trades), replace=True)
        boot_means[i] = sample.mean()
        boot_win_rates[i] = (sample > 0).mean()

    p_profitable = float((boot_means > 0).mean())

    # Confidence label
    if p_profitable >= 0.97:
        confidence_label = "HIGH"
    elif p_profitable >= 0.90:
        confidence_label = "MODERATE"
    elif p_profitable >= 0.75:
        confidence_label = "LOW"
    else:
        confidence_label = "INSUFFICIENT"

    return {
        "n_trades": len(trades),
        "mean_return_pct": round(float(trades.mean()), 3),
        "expectancy_95ci": (round(float(np.percentile(boot_means, 2.5)), 3),
                            round(float(np.percentile(boot_means, 97.5)), 3)),
        "p_profitable": round(p_profitable, 4),
        "win_rate_pct": round(float((trades > 0).mean() * 100), 1),
        "win_rate_95ci": (round(float(np.percentile(boot_win_rates * 100, 2.5)), 1),
                          round(float(np.percentile(boot_win_rates * 100, 97.5)), 1)),
        "confidence_label": confidence_label,
    }


def run_adx_comparison(ticker: str, adx_thresholds: List[float] = None) -> None:
    """Compare win rate and trade count across ADX filter values."""
    if adx_thresholds is None:
        adx_thresholds = [0, 15, 20, 25, 30]

    print(f"\n{'='*62}")
    print(f"  ADX FILTER COMPARISON: {ticker}")
    print(f"  {'ADX min':>8}  {'Trades':>7}  {'Win%':>6}  {'Return':>8}  {'PF':>6}")
    print(f"{'='*62}")

    for adx_min in adx_thresholds:
        result = walk_forward_backtest(ticker, n_folds=4, adx_filter_min=adx_min)
        total_t = result["total_trades"]
        win_r = result["avg_win_rate"]
        ret = result["total_return_pct"]

        all_sells = pd.concat([
            r["trades"][r["trades"]["type"].isin(["SELL", "COVER"])]
            for r in result["fold_results"]
            if len(r["trades"]) and "type" in r["trades"].columns
        ]) if result["fold_results"] else pd.DataFrame()

        pf = 0.0
        if len(all_sells):
            gross_profit = all_sells[all_sells["pnl"] > 0]["pnl"].sum()
            gross_loss = abs(all_sells[all_sells["pnl"] <= 0]["pnl"].sum())
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        label = " (no filter)" if adx_min == 0 else ""
        print(f"  {adx_min:>8.0f}  {total_t:>7}  {win_r:>6.1f}%  {ret:>+7.2f}%  {pf:>6.2f}{label}")
    print(f"{'='*62}")


def print_simulation_report(ticker: str, wf: Dict, boot: Dict) -> None:
    sep = "=" * 58
    print(f"\n{sep}")
    adx_src = " (Optuna-tuned)" if wf.get("adx_from_optuna") else ""
    print(f"  WALK-FORWARD SIMULATION: {ticker}  ({wf['n_folds']} folds)")
    if wf["adx_filter_min"] > 0:
        print(f"  ADX filter: >{wf['adx_filter_min']:.0f} (trending markets only){adx_src}")
    print(sep)
    print(f"  Total trades   : {wf['total_trades']}")
    print(f"  Avg win rate   : {wf['avg_win_rate']:.1f}%")
    print(f"  Combined return: {wf['total_return_pct']:+.2f}%")
    print()
    if "error" in boot:
        print(f"  Bootstrap: {boot['error']}")
    else:
        lo, hi = boot["expectancy_95ci"]
        wlo, whi = boot["win_rate_95ci"]
        print(f"  Mean trade return : {boot['mean_return_pct']:+.3f}%")
        print(f"  Expectancy 95% CI : [{lo:+.3f}%, {hi:+.3f}%]")
        print(f"  P(profitable)     : {boot['p_profitable']*100:.1f}%")
        print(f"  Win rate          : {boot['win_rate_pct']:.1f}%  95% CI [{wlo:.1f}%, {whi:.1f}%]")
        print(f"  Confidence        : {boot['confidence_label']}")
    print(sep)
