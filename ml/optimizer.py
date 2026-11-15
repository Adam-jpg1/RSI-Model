import json
import logging
import os
from typing import Dict, Optional

import numpy as np

from config import CONFIG

logger = logging.getLogger(__name__)

_PARAMS_DIR = "models"


def _params_path(ticker: str, daily_confirm: bool = False) -> str:
    suffix = "_dc" if daily_confirm else ""
    return os.path.join(_PARAMS_DIR, f"params_{ticker}{suffix}.json")


def save_best_params(ticker: str, params: Dict, daily_confirm: bool = False):
    os.makedirs(_PARAMS_DIR, exist_ok=True)
    path = _params_path(ticker, daily_confirm)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"[optuna] Saved best params -> {path}")


def load_best_params(ticker: str, daily_confirm: bool = False) -> Optional[Dict]:
    # Prefer daily_confirm-specific params when available; fall back to base params
    if daily_confirm:
        dc_path = _params_path(ticker, daily_confirm=True)
        if os.path.exists(dc_path):
            with open(dc_path) as f:
                return json.load(f)
    path = _params_path(ticker, daily_confirm=False)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def optimize_ticker(
    ticker: str,
    n_trials: int = 60,
    n_folds: int = 4,
    period: str = "2y",
    interval: str = "1h",
    daily_confirm: bool = False,
    allow_short: bool = None,
    n_ensemble: int = 1,
    peer_close_dict: dict = None,
) -> Dict:
    """
    Optuna Bayesian search over XGBoost hyperparameters and ADX threshold for one ticker.
    Objective = mean(win_rate × profit_factor × drawdown_penalty) across walk-forward folds.

    When daily_confirm=True, optimizes hyperparameters for the daily_confirm gate scenario
    and saves to params_{ticker}_dc.json.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Run: pip install optuna")

    from data.fetcher import get_price_data, get_daily_data
    from data.indicators import (add_all_indicators, add_daily_trend_features,
                                  add_peer_features, add_macro_event_features,
                                  add_market_features, add_cross_asset_features,
                                  add_pead_features, add_calendar_features,
                                  add_cross_sectional_features,
                                  add_microstructure_features, add_regime_features,
                                  add_earnings_features)
    from data.fetcher import get_earnings_dates
    from ml.features import get_feature_matrix, build_target, target_bars_for_interval
    from ml.model import RSIModel
    from backtest.engine import _run_simulation
    from backtest.simulate import _build_daily_signals

    # When targeting dc+hold6+long combo, force matching settings so the
    # optimizer tunes parameters for the same scenario we actually trade.
    _allow_short = False if daily_confirm else (CONFIG.bot.allow_short if allow_short is None else allow_short)
    _tune_hold_bars = daily_confirm  # only tune hold bars for the dc scenario
    dc_label = " [daily_confirm+hold?+long]" if daily_confirm else ""
    print(f"\n[optuna] Optimizing {ticker}{dc_label} — {n_trials} trials, {n_folds} folds, interval={interval}")

    # Disable confidence threshold during optimization: the confidence filter creates a
    # catch-22 where random-param models can't generate signals above 0.52, scoring -999
    # on every trial so Optuna can't learn.  We optimize raw signal quality via Calmar
    # ratio and let the production confidence threshold act as a separate deployment gate.
    _orig_conf = CONFIG.bot.min_confidence
    CONFIG.bot.min_confidence = 0.0

    df = get_price_data(ticker, period=period, interval=interval)
    df = add_all_indicators(df)
    daily_df_raw = None
    spy_daily_raw = None
    try:
        daily_period = "5y" if daily_confirm else period
        daily_df_raw = get_daily_data(ticker, period=daily_period)
        daily_df_raw = add_all_indicators(daily_df_raw)
        if ticker != "SPY":
            try:
                spy_daily_raw = get_daily_data("SPY", period=daily_period)
            except Exception:
                spy_daily_raw = None
        vix_daily_raw = None
        try:
            vix_daily_raw = get_daily_data("^VIX", period=daily_period)
        except Exception:
            pass
        df = add_daily_trend_features(df, daily_df_raw,
                                      spy_daily_df=spy_daily_raw,
                                      vix_daily_df=vix_daily_raw)
        # Cross-asset macro features
        try:
            ca_assets = {}
            for sym in ["TLT", "HYG", "GLD", "UUP"]:
                try:
                    ca_assets[sym] = get_daily_data(sym, period=period)
                except Exception:
                    ca_assets[sym] = None
            df = add_cross_asset_features(
                df,
                tlt_daily_df=ca_assets.get("TLT"),
                hyg_daily_df=ca_assets.get("HYG"),
                gld_daily_df=ca_assets.get("GLD"),
                uup_daily_df=ca_assets.get("UUP"),
            )
        except Exception:
            pass
        try:
            df = add_pead_features(df, daily_df_raw)
        except Exception:
            pass
    except Exception:
        pass
    # Earnings proximity: days_to_earnings, days_since_earnings, near_earnings
    # Pre-earnings countdown IC=0.17-0.37 (strongest signal found in PEAD research)
    try:
        edates = get_earnings_dates(ticker)
        if edates:
            df = add_earnings_features(df, edates)
    except Exception:
        pass
    # Peer cross-asset RSI + SPY context + macro events
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
    try:
        spy_hourly = get_price_data("SPY", period=period, interval=interval)
        spy_hourly = add_all_indicators(spy_hourly)
        df = add_market_features(df, spy_hourly)
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
    if peer_close_dict:
        try:
            df = add_cross_sectional_features(df, ticker, peer_close_dict)
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
    features_full = get_feature_matrix(df, 0.0).dropna()
    df_aligned = df.loc[features_full.index]

    n = len(features_full)
    if n < 200:
        raise ValueError(f"Not enough data for {ticker}: {n} rows")
    fold_size = n // (n_folds + 1)

    # Pre-build daily signals once per fold — shared across all trials.
    # Use existing saved params (if any) for the daily model so signals match
    # what walk_forward_backtest will produce.  Random default params often produce
    # all-FLAT daily predictions, which blocks every trial via the daily confirm gate.
    daily_sigs_per_fold: list = []
    if daily_confirm and daily_df_raw is not None:
        existing = load_best_params(ticker, daily_confirm=daily_confirm)
        _daily_xgb = existing.get("xgb_params") if existing else None
        _orig_daily_xgb = None
        if _daily_xgb:
            _orig_daily_xgb = {k: v for k, v in CONFIG.ml.xgb_params.items()}
            CONFIG.ml.xgb_params.update(_daily_xgb)
        try:
            for fold in range(1, n_folds + 1):
                train_end = fold * fold_size
                train_end_ts = features_full.index[train_end - 1]
                sigs = _build_daily_signals(daily_df_raw, train_end_ts)
                daily_sigs_per_fold.append(sigs)
        finally:
            if _orig_daily_xgb is not None:
                CONFIG.ml.xgb_params.clear()
                CONFIG.ml.xgb_params.update(_orig_daily_xgb)
        print(f"[optuna] Pre-built daily signals for {n_folds} folds")

    def objective(trial: "optuna.Trial") -> float:
        trial_xgb = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 30),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 5.0),
        }
        adx_min = trial.suggest_float("adx_filter_min", 0.0, 40.0)
        trial_tp = trial.suggest_float("take_profit", 0.012, 0.07)
        trial_sl = trial.suggest_float("stop_loss", 0.004, 0.025)
        trial_hold = trial.suggest_int("max_hold_bars", 2, 16) if _tune_hold_bars else 6
        # Tune confidence alongside model params so production threshold is co-optimized.
        # Range starts at 0.33 (3-class random baseline) to avoid the catch-22 where
        # random models can't meet 0.52 → 0 trades → -999 blocks all learning.
        trial_conf = trial.suggest_float("min_confidence", 0.33, 0.52) if daily_confirm else 0.0
        # Partial TP: take half position at this gain, let rest ride.
        # AT sweep found 1.5% optimal; search 0–2.5% with 0.0 (disabled) also valid.
        trial_partial_tp = trial.suggest_float("partial_tp_pct", 0.0, 0.025) if daily_confirm else 0.0

        orig_xgb = {k: CONFIG.ml.xgb_params[k] for k in trial_xgb if k in CONFIG.ml.xgb_params}
        orig_tp = CONFIG.bot.take_profit_pct
        orig_sl = CONFIG.bot.stop_loss_pct
        CONFIG.ml.xgb_params.update(trial_xgb)
        CONFIG.bot.take_profit_pct = trial_tp
        CONFIG.bot.stop_loss_pct = trial_sl
        CONFIG.bot.min_confidence = trial_conf

        # Track per-fold returns + combined equity for full-period Calmar
        fold_returns = []  # (ret_fraction, bars, n_trades)
        combined_equity = []  # equity values across all folds (for max_dd)
        total_trades_all = 0
        try:
            for fold in range(1, n_folds + 1):
                train_end = fold * fold_size
                test_start = train_end
                test_end = min((fold + 1) * fold_size, n)

                X_tr = features_full.iloc[:train_end]
                y_tr = build_target(df_aligned.iloc[:train_end],
                                    n_bars=target_bars_for_interval(interval))
                mask = X_tr.notna().all(axis=1) & y_tr.notna()
                X_tr, y_tr = X_tr[mask], y_tr[mask]

                if len(X_tr) < 100:
                    return -999.0

                # Early stopping validation split (last 15% of training data)
                X_val_es = y_val_es = None
                if len(X_tr) > 150:
                    val_cut = int(len(X_tr) * 0.85)
                    X_val_es = X_tr.iloc[val_cut:]
                    y_val_es = y_tr.iloc[val_cut:]
                    X_tr = X_tr.iloc[:val_cut]
                    y_tr = y_tr.iloc[:val_cut]

                model = RSIModel()
                model.build(early_stopping_rounds=20)
                try:
                    model.fit(X_tr, y_tr, X_val=X_val_es, y_val=y_val_es)
                    for seed_off in range(1, n_ensemble):
                        model.add_ensemble_member(X_tr, y_tr, X_val=X_val_es, y_val=y_val_es,
                                                  seed=42 + seed_off * 7)
                except Exception:
                    return -999.0

                feat_test = features_full.iloc[test_start:test_end]
                df_test = df_aligned.iloc[test_start:test_end]

                daily_sigs = daily_sigs_per_fold[fold - 1] if daily_sigs_per_fold else None
                result = _run_simulation(df_test, feat_test, model, 100_000.0, ticker, adx_min,
                                         allow_short=_allow_short,
                                         daily_signals=daily_sigs,
                                         max_hold_bars=trial_hold,
                                         partial_tp_pct=trial_partial_tp)
                m = result["metrics"]

                n_trades = m["total_trades"]
                if n_trades == 0:
                    # Fold blocked by regime gate — skip; equity stays flat (no trading)
                    fold_returns.append((0.0, test_end - test_start, 0))
                    continue
                if n_trades < 3:
                    # Too few trades for reliable stats — treat as small loss to discourage
                    fold_returns.append((-0.005, test_end - test_start, n_trades))
                    continue

                ret = result["total_return_pct"] / 100.0
                total_trades_all += n_trades
                fold_returns.append((ret, test_end - test_start, n_trades))

                # Accumulate equity for combined max-drawdown calculation
                eq = result["equity_curve"]
                if len(eq) > 0:
                    eq_vals = eq["equity"].values if "equity" in eq.columns else eq.values
                    combined_equity.extend(eq_vals.tolist())

        finally:
            CONFIG.ml.xgb_params.update(orig_xgb)
            CONFIG.bot.take_profit_pct = orig_tp
            CONFIG.bot.stop_loss_pct = orig_sl
            CONFIG.bot.min_confidence = 0.0  # restore to outer override (not original)

        if not fold_returns:
            return -999.0

        # Full-period Calmar: compound returns across folds, divide by combined max-dd.
        # This matches what BAA actually measures, avoiding the per-fold averaging bias
        # where good per-fold Calmar can mask a poor combined equity curve.
        compound = 1.0
        total_bars = 0
        for ret, bars, _ in fold_returns:
            compound *= (1.0 + ret)
            total_bars += bars

        years = max(total_bars / (252 * 6.5), 0.01)
        compound_return = compound - 1.0
        if compound_return > -1.0:
            combined_cagr = ((1.0 + compound_return) ** (1.0 / years) - 1.0) * 100.0
        else:
            combined_cagr = -100.0

        # Combined max drawdown over all fold equity curves
        if combined_equity:
            eq_arr = np.array(combined_equity)
            roll_max = np.maximum.accumulate(eq_arr)
            dd = (eq_arr - roll_max) / roll_max
            combined_max_dd = abs(float(dd.min())) * 100.0
        else:
            combined_max_dd = 1.0

        calmar = combined_cagr / max(combined_max_dd, 1.0)

        # Trade-count credibility weight: fully trust ≥(15 × n_folds) total trades
        trade_weight = min(total_trades_all / (15.0 * len(fold_returns)), 1.0)

        # Fold-consistency penalty: models that only trade in 1-2 out of 4 folds are
        # regime-specific and won't generalize. Require trades (>=5) in >=50% of folds.
        # Squaring the fraction strongly favors consistent models over concentrated ones.
        active_folds = sum(1 for _, _, n in fold_returns if n >= 5)
        fold_consistency = active_folds / max(len(fold_returns), 1)
        fold_penalty = fold_consistency ** 2   # 1 fold=0.0625, 2=0.25, 3=0.5625, 4=1.0

        return calmar * trade_weight * fold_penalty

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    finally:
        CONFIG.bot.min_confidence = _orig_conf

    best = study.best_params.copy()
    adx_min = round(best.pop("adx_filter_min", 0.0), 1)
    best_tp = round(best.pop("take_profit", CONFIG.bot.take_profit_pct), 4)
    best_sl = round(best.pop("stop_loss", CONFIG.bot.stop_loss_pct), 4)
    best_hold = int(best.pop("max_hold_bars", 6))
    best_conf = round(best.pop("min_confidence", _orig_conf), 3)
    best_partial_tp = round(best.pop("partial_tp_pct", 0.0), 4)

    # Strip non-XGBoost keys from xgb_params
    xgb_keys = set(CONFIG.ml.xgb_params.keys()) | {
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda",
    }
    best_xgb_only = {k: v for k, v in best.items() if k in xgb_keys}
    merged_xgb = {**CONFIG.ml.xgb_params, **best_xgb_only}

    best_params = {
        "ticker": ticker,
        "xgb_params": merged_xgb,
        "adx_filter_min": adx_min,
        "take_profit": best_tp,
        "stop_loss": best_sl,
        "max_hold_bars": best_hold,
        "min_confidence": best_conf,
        "partial_tp_pct": best_partial_tp,
        "best_score": round(study.best_value, 5),
        "n_trials": n_trials,
        "daily_confirm": daily_confirm,
    }
    save_best_params(ticker, best_params, daily_confirm=daily_confirm)

    print(
        f"[optuna] {ticker} score={best_params['best_score']:.4f}  "
        f"adx={adx_min:.1f}  tp={best_tp:.3f}  sl={best_sl:.3f}  hold={best_hold}  "
        f"conf={best_conf:.2f}  ptp={best_partial_tp:.4f}  "
        f"depth={best.get('max_depth')}  n_est={best.get('n_estimators')}  "
        f"lr={best.get('learning_rate', 0):.4f}"
    )
    return best_params
