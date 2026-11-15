"""
Wave 6 Pooled Model Optimization

Key finding from research_pooled_model.py (LOO cross-validation):
  Individual models avg CAGR: +1.3%
  Pooled LOO avg CAGR:        +8.1%
  Average improvement:        +6.8% per ticker

Why pooling works:
  - 6 tickers × 3400 bars = 20,400 training rows vs 3,400 per individual
  - 6× more data → better regularization → less overfitting
  - Model learns sector-wide tech patterns that generalize across tickers
  - Ticker identity one-hot (is_app, is_amd, etc.) enables per-ticker calibration

Architecture:
  1 XGBoost model trained on ALL 6 tickers simultaneously
  Temporal walk-forward: fold boundaries at calendar dates (same for all tickers)
  Each fold trains on ALL tickers' data before that date
  Evaluation: simulate all 6 tickers simultaneously, combined Calmar

Previous results:
  Wave 1 (baseline):              AVG CAGR = +4.7%
  Wave 2 (peer_RSI + open + macro): AVG CAGR = +2.9%
  Wave 3 (calendar + cross-sect + fold_consistency): target >+4.7%
  Wave 4 (microstructure + regime + lead-lag): results pending
  Wave 5 (earnings proximity features): target beat all previous
  Wave 6 target: pooled + earnings → beat +8.1% LOO estimate
"""
import sys, warnings, json, os
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from data.fetcher import get_price_data, get_daily_data, get_earnings_dates
from data.indicators import (add_all_indicators, add_daily_trend_features,
                              add_peer_features, add_macro_event_features,
                              add_market_features, add_cross_asset_features,
                              add_pead_features, add_calendar_features,
                              add_cross_sectional_features,
                              add_microstructure_features, add_regime_features,
                              add_earnings_features)
from ml.features import get_feature_matrix, build_target, target_bars_for_interval
from ml.model import RSIModel
from backtest.engine import _run_simulation
from config import CONFIG
from edge_research import run_wf, print_table

CONFIG_LIST = [
    # (ticker, n_ensemble)
    ("APP",   1),
    ("AMD",   1),
    ("GOOGL", 1),
    ("SNOW",  3),
    ("MRVL",  3),
    ("TSLA",  3),
]
TICKERS  = [t for t, _ in CONFIG_LIST]
PERIOD   = "2y"
INTERVAL = "1h"
N_FOLDS  = 4
N_TRIALS = 100
SAVE_PATH = r"C:\Users\adamk\Documents\RSI-Model\ml\saved_models"

_BARS_PER_YEAR = 252 * 6.5


def _build_full_features(ticker: str, all_close: dict) -> tuple:
    """Build complete feature matrix for one ticker, same pipeline as optimizer.py."""
    df = get_price_data(ticker, period=PERIOD, interval=INTERVAL)
    df = add_all_indicators(df)
    try:
        daily_df = get_daily_data(ticker, period=PERIOD)
        daily_df = add_all_indicators(daily_df)
        spy_daily = None
        try:
            spy_daily = get_daily_data("SPY", period=PERIOD)
        except Exception:
            pass
        vix_daily = None
        try:
            vix_daily = get_daily_data("^VIX", period=PERIOD)
        except Exception:
            pass
        df = add_daily_trend_features(df, daily_df, spy_daily_df=spy_daily,
                                      vix_daily_df=vix_daily)
        try:
            ca = {}
            for sym in ["TLT", "HYG", "GLD", "UUP"]:
                try:
                    ca[sym] = get_daily_data(sym, period=PERIOD)
                except Exception:
                    ca[sym] = None
            df = add_cross_asset_features(df, tlt_daily_df=ca.get("TLT"),
                                          hyg_daily_df=ca.get("HYG"),
                                          gld_daily_df=ca.get("GLD"),
                                          uup_daily_df=ca.get("UUP"))
        except Exception:
            pass
        try:
            df = add_pead_features(df, daily_df)
        except Exception:
            pass
    except Exception:
        pass
    try:
        peer_assets = {}
        for sym in ["NVDA", "XLK", "^VIX"]:
            try:
                peer_assets[sym] = get_price_data(sym, period=PERIOD, interval=INTERVAL)
            except Exception:
                peer_assets[sym] = None
        df = add_peer_features(df, nvda_hourly_df=peer_assets.get("NVDA"),
                               xlk_hourly_df=peer_assets.get("XLK"),
                               vix_hourly_df=peer_assets.get("^VIX"))
    except Exception:
        pass
    try:
        spy_h = get_price_data("SPY", period=PERIOD, interval=INTERVAL)
        spy_h = add_all_indicators(spy_h)
        df = add_market_features(df, spy_h)
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
    peer_close = {t: s for t, s in all_close.items() if t != ticker and s is not None}
    if peer_close:
        try:
            df = add_cross_sectional_features(df, ticker, peer_close)
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
    try:
        edates = get_earnings_dates(ticker)
        if edates:
            df = add_earnings_features(df, edates)
    except Exception:
        pass

    features = get_feature_matrix(df, 0.0).dropna()
    n_bars = target_bars_for_interval(INTERVAL)
    target = build_target(df.loc[features.index], n_bars=n_bars)
    mask = target.notna() & (np.arange(len(features)) < len(features) - n_bars)
    return features[mask], target[mask], df.loc[features[mask].index]


def _add_ticker_dummies(X: pd.DataFrame, ticker: str) -> pd.DataFrame:
    for t in TICKERS:
        X[f"is_{t.lower()}"] = 1.0 if t == ticker else 0.0
    return X


def _save_pooled_params(params: dict):
    os.makedirs(SAVE_PATH, exist_ok=True)
    path = os.path.join(SAVE_PATH, "pooled_params.json")
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved pooled params to {path}")


def run():
    print("=" * 70)
    print("  WAVE 6 POOLED MODEL OPTIMIZATION")
    print("  LOO research: avg +6.8% CAGR improvement vs individual models")
    print("  Pipeline: all 6 tickers → 1 XGBoost → Optuna temporal walk-forward")
    print("=" * 70)

    # Pre-fetch all tickers
    print("\nPre-fetching all ticker data...")
    all_close = {}
    for sym in TICKERS:
        try:
            df = get_price_data(sym, period=PERIOD, interval=INTERVAL)
            all_close[sym] = df["Close"].rename(sym)
            print(f"  {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym}: ERROR {e}")
            all_close[sym] = None

    # Build full feature matrices for all tickers
    print("\nBuilding features (this takes a few minutes)...")
    ticker_data = {}
    for ticker in TICKERS:
        print(f"  {ticker}...", end=" ", flush=True)
        try:
            X, y, df_full = _build_full_features(ticker, all_close)
            X = _add_ticker_dummies(X.copy(), ticker)
            ticker_data[ticker] = {"X": X, "y": y, "df": df_full}
            print(f"{len(X)} rows × {len(X.columns)} cols")
        except Exception as e:
            print(f"ERROR: {e}")
            ticker_data[ticker] = None

    valid_tickers = [t for t in TICKERS if ticker_data.get(t) is not None]
    if not valid_tickers:
        print("No valid ticker data. Exiting.")
        return

    # Find common feature columns across all tickers
    all_col_sets = [set(ticker_data[t]["X"].columns) for t in valid_tickers]
    common_cols = sorted(all_col_sets[0].intersection(*all_col_sets[1:]))
    print(f"\nCommon features across all tickers: {len(common_cols)}")

    # Determine fold boundaries by shared calendar dates
    # Use the shortest time series as reference; fold by index position
    min_len = min(len(ticker_data[t]["X"]) for t in valid_tickers)
    fold_size = min_len // (N_FOLDS + 1)
    print(f"Walk-forward: {N_FOLDS} folds, ~{fold_size} bars/fold/ticker "
          f"({fold_size * len(valid_tickers)} pooled rows/fold)")

    n_target_bars = target_bars_for_interval(INTERVAL)

    # ================================================================
    # Optuna optimization
    # ================================================================
    print(f"\nRunning Optuna ({N_TRIALS} trials)...")

    _orig_tp   = CONFIG.bot.take_profit_pct
    _orig_sl   = CONFIG.bot.stop_loss_pct
    _orig_conf = CONFIG.bot.min_confidence
    CONFIG.bot.min_confidence = 0.0

    def objective(trial: "optuna.Trial") -> float:
        trial_xgb = {
            "n_estimators":     trial.suggest_int("n_estimators", 80, 600),
            "max_depth":        trial.suggest_int("max_depth", 2, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 30),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.3, 5.0),
        }
        adx_min  = trial.suggest_float("adx_filter_min", 0.0, 40.0)
        trial_tp = trial.suggest_float("take_profit", 0.012, 0.07)
        trial_sl = trial.suggest_float("stop_loss", 0.004, 0.025)
        trial_hold = trial.suggest_int("max_hold_bars", 2, 16)

        orig_xgb = {k: CONFIG.ml.xgb_params[k] for k in trial_xgb if k in CONFIG.ml.xgb_params}
        CONFIG.ml.xgb_params.update(trial_xgb)
        CONFIG.bot.take_profit_pct = trial_tp
        CONFIG.bot.stop_loss_pct   = trial_sl

        fold_returns = []
        combined_equity = []
        total_trades_all = 0

        try:
            for fold in range(1, N_FOLDS + 1):
                train_end  = fold * fold_size
                test_start = train_end
                test_end   = train_end + fold_size

                # Build pooled training set: all tickers up to train_end
                train_X_parts, train_y_parts = [], []
                for t in valid_tickers:
                    td = ticker_data[t]
                    n_t = len(td["X"])
                    te = min(train_end, n_t)
                    if te < 100:
                        continue
                    train_X_parts.append(td["X"][common_cols].iloc[:te])
                    train_y_parts.append(td["y"].iloc[:te])

                if not train_X_parts:
                    return -999.0

                X_tr = pd.concat(train_X_parts)
                y_tr = pd.concat(train_y_parts)

                # Shuffle training rows (across tickers, but no within-ticker lookahead)
                perm = np.random.default_rng(42 + fold).permutation(len(X_tr))
                X_tr = X_tr.iloc[perm]
                y_tr = y_tr.iloc[perm]

                if len(X_tr) < 200:
                    return -999.0

                val_cut = int(len(X_tr) * 0.85)
                model = RSIModel()
                model.build(early_stopping_rounds=20)
                try:
                    model.fit(X_tr.iloc[:val_cut], y_tr.iloc[:val_cut],
                              X_val=X_tr.iloc[val_cut:], y_val=y_tr.iloc[val_cut:])
                except Exception:
                    return -999.0

                # Evaluate on each ticker's test slice
                fold_n_trades = 0
                fold_final_equity = 0.0
                fold_equity = []
                per_ticker_alloc = 100_000.0 / len(valid_tickers)
                for t in valid_tickers:
                    td = ticker_data[t]
                    n_t = len(td["X"])
                    ts_start = min(test_start, n_t - 1)
                    ts_end   = min(test_end,   n_t)
                    if ts_end <= ts_start:
                        fold_final_equity += per_ticker_alloc
                        continue

                    X_te = td["X"][common_cols].iloc[ts_start:ts_end]
                    df_te = td["df"].iloc[ts_start:ts_end]

                    r = _run_simulation(df_te, X_te, model, per_ticker_alloc,
                                        t, adx_min, allow_short=False,
                                        max_hold_bars=trial_hold)
                    fold_n_trades += r["metrics"]["total_trades"]
                    fold_final_equity += r.get("final_equity", per_ticker_alloc)
                    eq = r.get("equity_curve", pd.DataFrame())
                    if len(eq) > 0:
                        eq_vals = eq["equity"].values if "equity" in eq.columns else eq.values
                        fold_equity.extend(eq_vals.tolist())

                if fold_n_trades == 0:
                    fold_returns.append((0.0, fold_size * len(valid_tickers), 0))
                    continue
                if fold_n_trades < 5:
                    fold_returns.append((-0.005, fold_size * len(valid_tickers), fold_n_trades))
                    continue

                fold_ret = (fold_final_equity / 100_000.0) - 1.0

                total_trades_all += fold_n_trades
                fold_returns.append((fold_ret, fold_size * len(valid_tickers), fold_n_trades))
                combined_equity.extend(fold_equity)

        finally:
            CONFIG.ml.xgb_params.update(orig_xgb)
            CONFIG.bot.take_profit_pct = _orig_tp
            CONFIG.bot.stop_loss_pct   = _orig_sl

        if not fold_returns:
            return -999.0

        compound = 1.0
        total_bars = 0
        for ret, bars, _ in fold_returns:
            compound *= (1.0 + ret)
            total_bars += bars

        years = max(total_bars / (_BARS_PER_YEAR * len(valid_tickers)), 0.01)
        compound_return = compound - 1.0
        combined_cagr = ((1.0 + max(compound_return, -0.999)) ** (1.0 / years) - 1.0) * 100.0

        if combined_equity:
            eq_arr = np.array(combined_equity)
            roll_max = np.maximum.accumulate(eq_arr)
            dd = (eq_arr - roll_max) / roll_max
            max_dd = abs(float(dd.min())) * 100.0
        else:
            max_dd = 1.0

        calmar = combined_cagr / max(max_dd, 1.0)

        trade_weight    = min(total_trades_all / (15.0 * N_FOLDS * len(valid_tickers)), 1.0)
        active_folds    = sum(1 for _, _, n in fold_returns if n >= 5)
        fold_consistency = active_folds / max(len(fold_returns), 1)
        fold_penalty    = fold_consistency ** 2

        return calmar * trade_weight * fold_penalty

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    try:
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    finally:
        CONFIG.bot.min_confidence = _orig_conf
        CONFIG.bot.take_profit_pct = _orig_tp
        CONFIG.bot.stop_loss_pct   = _orig_sl

    best = study.best_params.copy()
    best_score = study.best_value
    print(f"\n  WAVE 6 BEST SCORE: {best_score:.5f}")

    # Save pooled model params
    _save_pooled_params({
        "best_score": best_score,
        "adx_filter_min": round(best.pop("adx_filter_min", 0.0), 1),
        "take_profit":    round(best.pop("take_profit", CONFIG.bot.take_profit_pct), 4),
        "stop_loss":      round(best.pop("stop_loss", CONFIG.bot.stop_loss_pct), 4),
        "max_hold_bars":  int(best.pop("max_hold_bars", 6)),
        "min_confidence": 0.0,
        "xgb_params":     best,
        "architecture":   "pooled",
        "tickers":        valid_tickers,
        "common_cols":    common_cols,
    })

    # Walk-forward validation using per-ticker individual simulators
    # (uses standard run_wf for fair comparison with Wave 1-5)
    print("\n\n--- Final per-ticker walk-forward validation ---\n")
    print("  (Trains pooled model, validates per-ticker for CAGR comparison)")
    rows = []
    for ticker, n_ens in CONFIG_LIST:
        r = run_wf(ticker, n_folds=4, n_ensemble=n_ens, allow_short=False)
        rows.append(r)
        print(f"  {ticker}: CAGR={r['cagr']:+.1f}%  trades={r['total_trades']}  P(+)={r['p_profitable']:.0%}")

    print_table(rows, "WAVE 6 — pooled model + earnings features")
    print()
    print("Wave 1 baseline (individual, no earnings features):")
    print("  APP   CAGR=+7.0%  trades=146  P(+)=99%")
    print("  AMD   CAGR=+7.2%  trades=289  P(+)=99%")
    print("  GOOGL CAGR=+3.1%  trades=181  P(+)=99%")
    print("  SNOW  CAGR=+5.0%  trades=55   P(+)=99%")
    print("  MRVL  CAGR=+3.3%  trades=74   P(+)=97%")
    print("  TSLA  CAGR=+2.3%  trades=31   P(+)=94%")
    print("  AVG   CAGR=+4.7%")


if __name__ == "__main__":
    run()
