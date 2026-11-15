"""
LightGBM vs XGBoost Comparison

XGBoost is our current baseline. LightGBM often outperforms it on tabular data:
  - Leaf-wise tree growth (vs. level-wise) captures complex patterns better
  - GOSS + EFB make it 5-10x faster without sacrificing accuracy
  - Native handling of NaN (no imputation needed)
  - Better at high-cardinality feature spaces (we have 144 features)

Hypothesis: LightGBM will produce higher IC on our feature set because:
  - Our 144 features have many low-IC signals that need nonlinear combinations
  - Leaf-wise growth finds rarer but stronger interaction patterns
  - Faster training -> more Optuna trials -> better hyperparameter tuning

Methodology:
  1. Train XGBoost and LightGBM with same walk-forward setup (4-fold expanding)
  2. Measure: Calmar ratio, CAGR, win rate, # trades
  3. Test: does LGBM consistently beat XGB, or is it ticker-specific?
  4. If LGBM wins: replace XGBoost in optimizer
  5. If ensemble: average XGB + LGBM probabilities (like we do for multi-seed XGB)
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

from data.fetcher import get_price_data, get_daily_data
from data.indicators import (add_all_indicators, add_daily_trend_features,
                              add_peer_features, add_macro_event_features,
                              add_market_features, add_cross_asset_features,
                              add_pead_features, add_calendar_features,
                              add_cross_sectional_features,
                              add_microstructure_features, add_regime_features)
from ml.features import get_feature_matrix, build_target, target_bars_for_interval
from ml.model import RSIModel
from ml.optimizer import load_best_params
from backtest.engine import _run_simulation
from config import CONFIG

TICKERS  = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD   = "2y"
INTERVAL = "1h"
N_FOLDS  = 4

try:
    import lightgbm as lgb
    LGBM_OK = True
except ImportError:
    print("lightgbm not installed — run: pip install lightgbm")
    LGBM_OK = False


def build_full_features(ticker: str, all_close: dict) -> tuple:
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
        df = add_daily_trend_features(df, daily_df,
                                      spy_daily_df=spy_daily,
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
            if sym != ticker:
                try:
                    peer_assets[sym] = get_price_data(sym, period=PERIOD, interval=INTERVAL)
                except Exception:
                    peer_assets[sym] = None
        df = add_peer_features(df,
                               nvda_hourly_df=peer_assets.get("NVDA"),
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
    peer_close = {t: s for t, s in all_close.items()
                  if t != ticker and s is not None}
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

    features = get_feature_matrix(df, 0.0).dropna()
    n_bars = target_bars_for_interval(INTERVAL)
    target = build_target(df.loc[features.index], n_bars=n_bars)
    mask = target.notna() & (np.arange(len(features)) < len(features) - n_bars)
    return features[mask], target[mask], df.loc[features[mask].index]


def run_xgb_fold(X_tr, y_tr, X_te, params):
    """Train XGBoost and get probabilities on test set."""
    orig_xgb = {k: CONFIG.ml.xgb_params[k] for k in params if k in CONFIG.ml.xgb_params}
    CONFIG.ml.xgb_params.update(params)
    try:
        model = RSIModel()
        model.build(early_stopping_rounds=20)
        val_cut = int(len(X_tr) * 0.85)
        model.fit(X_tr.iloc[:val_cut], y_tr.iloc[:val_cut],
                  X_val=X_tr.iloc[val_cut:], y_val=y_tr.iloc[val_cut:])
        _, probs = model.predict(X_te)
        return probs, model
    finally:
        CONFIG.ml.xgb_params.update(orig_xgb)


def run_lgbm_fold(X_tr, y_tr, X_te, params):
    """Train LightGBM and get probabilities on test set."""
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    feat_names = list(X_tr.columns)
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te[feat_names])

    val_cut = int(len(X_tr_scaled) * 0.85)
    dtrain = lgb.Dataset(X_tr_scaled[:val_cut], label=y_tr.values[:val_cut])
    dval   = lgb.Dataset(X_tr_scaled[val_cut:], label=y_tr.values[val_cut:], reference=dtrain)

    lgbm_params = {
        "objective": "multiclass",
        "num_class": 3,
        "num_leaves": params.get("num_leaves", 31),
        "max_depth": params.get("max_depth", 6),
        "learning_rate": params.get("learning_rate", 0.05),
        "subsample": params.get("subsample", 0.8),
        "colsample_bytree": params.get("colsample_bytree", 0.8),
        "min_child_samples": params.get("min_child_samples", 20),
        "reg_alpha": params.get("reg_alpha", 0.1),
        "reg_lambda": params.get("reg_lambda", 1.0),
        "n_estimators": params.get("n_estimators", 300),
        "verbose": -1,
        "random_state": 42,
        "metric": "multi_logloss",
    }
    callbacks = [lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)]
    model = lgb.train(lgbm_params, dtrain,
                      num_boost_round=lgbm_params["n_estimators"],
                      valid_sets=[dval],
                      callbacks=callbacks)
    probs = model.predict(X_te_scaled)  # shape: (n, 3)
    return probs, model, scaler, feat_names


def walk_forward(ticker: str, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame,
                 xgb_params: dict, lgbm_params: dict) -> dict:
    """Run 4-fold walk-forward for XGB vs LGBM."""
    n = len(X)
    fold_size = n // (N_FOLDS + 1)

    xgb_results = []
    lgbm_results = []

    adx_min = xgb_params.pop("adx_filter_min", 15.0)
    tp = xgb_params.pop("take_profit", CONFIG.bot.take_profit_pct)
    sl = xgb_params.pop("stop_loss", CONFIG.bot.stop_loss_pct)
    mc = xgb_params.pop("min_confidence", 0.0)

    orig_tp   = CONFIG.bot.take_profit_pct
    orig_sl   = CONFIG.bot.stop_loss_pct
    orig_conf = CONFIG.bot.min_confidence
    CONFIG.bot.take_profit_pct = tp
    CONFIG.bot.stop_loss_pct   = sl
    CONFIG.bot.min_confidence  = mc

    try:
        for fold in range(1, N_FOLDS + 1):
            train_end  = fold * fold_size
            test_start = train_end
            test_end   = min((fold + 1) * fold_size, n)

            X_tr = X.iloc[:train_end]
            y_tr = y.iloc[:train_end]
            X_te = X.iloc[test_start:test_end]
            df_te = df.iloc[test_start:test_end]

            fold_bars = test_end - test_start

            # XGBoost
            try:
                xgb_probs, xgb_model = run_xgb_fold(X_tr, y_tr, X_te, {**xgb_params})
                # Use _run_simulation with XGBoost model
                r_xgb = _run_simulation(df_te, X_te, xgb_model, 100_000.0,
                                        ticker, adx_min, allow_short=False)
                xgb_results.append((r_xgb, fold_bars))
            except Exception as e:
                print(f"    XGB fold {fold} error: {e}")
                xgb_results.append((None, fold_bars))

            # LightGBM — wrap in a simple runner that uses probabilities directly
            if LGBM_OK:
                try:
                    lgbm_probs, lgbm_model, lgbm_scaler, feat_names = run_lgbm_fold(
                        X_tr, y_tr, X_te, lgbm_params
                    )
                    # Build a mock model wrapper for _run_simulation
                    _CLASS_LABELS = {0: "DOWN", 1: "FLAT", 2: "UP"}

                    class LGBMWrapper:
                        def __init__(self, m, s, fn):
                            self.model = m
                            self._scaler = s
                            self._feat_names = fn
                            self._ensemble = []
                            self.feature_names = fn
                        def predict(self, X_df):
                            Xs = self._scaler.transform(X_df[self._feat_names])
                            probs = self.model.predict(Xs)
                            preds = np.argmax(probs, axis=1)
                            return preds, probs
                        def predict_single(self, feature_row):
                            X = pd.DataFrame([feature_row[self._feat_names]])
                            Xs = self._scaler.transform(X)
                            probs = self.model.predict(Xs)  # (1, 3)
                            label_idx = int(np.argmax(probs[0]))
                            return {
                                "signal": _CLASS_LABELS[label_idx],
                                "confidence": float(probs[0][label_idx]),
                                "probs": {_CLASS_LABELS[i]: float(p)
                                          for i, p in enumerate(probs[0])},
                            }

                    lgbm_wrapper = LGBMWrapper(lgbm_model, lgbm_scaler, feat_names)
                    r_lgbm = _run_simulation(df_te, X_te, lgbm_wrapper, 100_000.0,
                                            ticker, adx_min, allow_short=False)
                    lgbm_results.append((r_lgbm, fold_bars))
                except Exception as e:
                    print(f"    LGBM fold {fold} error: {e}")
                    lgbm_results.append((None, fold_bars))
    finally:
        CONFIG.bot.take_profit_pct = orig_tp
        CONFIG.bot.stop_loss_pct   = orig_sl
        CONFIG.bot.min_confidence  = orig_conf

    _BARS_PER_YEAR = 252 * 6.5

    def aggregate(fold_results_list):
        valid = [(r, n_bars) for r, n_bars in fold_results_list if r is not None]
        if not valid:
            return {"cagr": 0.0, "total_trades": 0, "win_rate": 0.0}
        total_trades = sum(r["metrics"].get("total_trades", 0) for r, _ in valid)
        cagrs = []
        for r, bars in valid:
            years = max(bars / _BARS_PER_YEAR, 0.01)
            tr = r.get("total_return_pct", 0.0) / 100.0
            cagrs.append(((1.0 + tr) ** (1.0 / years) - 1.0) * 100.0)
        avg_win = np.mean([r["metrics"].get("win_rate_pct", 0.0) for r, _ in valid
                           if r["metrics"].get("total_trades", 0) > 0]) if any(
                           r["metrics"].get("total_trades", 0) > 0 for r, _ in valid) else 0.0
        return {"cagr": np.mean(cagrs), "total_trades": total_trades, "win_rate": avg_win}

    return {
        "xgb":  aggregate(xgb_results),
        "lgbm": aggregate(lgbm_results),
    }


def run():
    if not LGBM_OK:
        return

    print("=" * 70)
    print("  LightGBM vs XGBoost Comparison")
    print(f"  Tickers: {TICKERS}  |  {N_FOLDS}-fold walk-forward")
    print("=" * 70)

    # Pre-fetch Close series
    print("\nPre-fetching Close series...")
    all_close = {}
    for sym in TICKERS:
        try:
            df_sym = get_price_data(sym, period=PERIOD, interval=INTERVAL)
            all_close[sym] = df_sym["Close"].rename(sym)
            print(f"  {sym}: {len(df_sym)} bars")
        except Exception as e:
            print(f"  {sym}: ERROR {e}")
            all_close[sym] = None

    results_table = []

    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")
        try:
            X, y, df = build_full_features(ticker, all_close)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        print(f"  Features: {X.shape[0]} rows x {X.shape[1]} cols")

        # Load saved XGBoost params
        saved = load_best_params(ticker)
        xgb_params = {}
        lgbm_params = {}
        if saved:
            xgb_params = {**saved.get("xgb_params", {})}
            xgb_params["adx_filter_min"] = saved.get("adx_filter_min", 15.0)
            xgb_params["take_profit"]    = saved.get("take_profit", CONFIG.bot.take_profit_pct)
            xgb_params["stop_loss"]      = saved.get("stop_loss", CONFIG.bot.stop_loss_pct)
            # Mirror key XGBoost params in LGBM (same scale, different algorithm)
            lgbm_params = {
                "max_depth":        min(xgb_params.get("max_depth", 4), 8),
                "num_leaves":       2 ** xgb_params.get("max_depth", 4),  # ~equivalent depth
                "learning_rate":    xgb_params.get("learning_rate", 0.05),
                "subsample":        xgb_params.get("subsample", 0.8),
                "colsample_bytree": xgb_params.get("colsample_bytree", 0.8),
                "reg_alpha":        xgb_params.get("reg_alpha", 0.1),
                "reg_lambda":       xgb_params.get("reg_lambda", 1.0),
                "min_child_samples": max(int(xgb_params.get("min_child_weight", 5)), 5),
                "n_estimators":     xgb_params.get("n_estimators", 300),
            }

        result = walk_forward(ticker, X, y, df, xgb_params, lgbm_params)
        xgb_r  = result["xgb"]
        lgbm_r = result["lgbm"]

        winner = "LGBM" if lgbm_r["cagr"] > xgb_r["cagr"] else "XGB "
        print(f"  XGB : CAGR={xgb_r['cagr']:+.1f}%  trades={xgb_r['total_trades']:>4}  "
              f"WR={xgb_r['win_rate']:.0%}")
        print(f"  LGBM: CAGR={lgbm_r['cagr']:+.1f}%  trades={lgbm_r['total_trades']:>4}  "
              f"WR={lgbm_r['win_rate']:.0%}  <-- {winner}")
        results_table.append({
            "ticker": ticker,
            "xgb_cagr":   xgb_r["cagr"],
            "lgbm_cagr":  lgbm_r["cagr"],
            "xgb_trades": xgb_r["total_trades"],
            "lgbm_trades": lgbm_r["total_trades"],
            "winner": winner.strip(),
        })

    print("\n\n" + "=" * 70)
    print("  SUMMARY: XGBoost vs LightGBM")
    print("=" * 70)
    print(f"\n  {'Ticker':<8} {'XGB CAGR':>10} {'LGBM CAGR':>10} {'Delta':>8} {'Winner':<6}")
    print("  " + "-" * 46)
    lgbm_wins = 0
    for r in results_table:
        delta = r["lgbm_cagr"] - r["xgb_cagr"]
        w = "LGBM" if r["winner"] == "LGBM" else "XGB"
        if w == "LGBM":
            lgbm_wins += 1
        print(f"  {r['ticker']:<8} {r['xgb_cagr']:>+10.1f}% {r['lgbm_cagr']:>+10.1f}%"
              f" {delta:>+8.1f}% {w:<6}")

    if results_table:
        avg_xgb  = np.mean([r["xgb_cagr"]  for r in results_table])
        avg_lgbm = np.mean([r["lgbm_cagr"] for r in results_table])
        print(f"\n  {'AVG':<8} {avg_xgb:>+10.1f}% {avg_lgbm:>+10.1f}% {avg_lgbm-avg_xgb:>+8.1f}%")
        print(f"\n  LGBM wins: {lgbm_wins}/{len(results_table)} tickers")

        if lgbm_wins >= 4:
            print("\n  RECOMMENDATION: Switch to LightGBM or use XGB+LGBM ensemble")
            print("  Next step: run Optuna with LightGBM objective function")
        elif lgbm_wins >= 2:
            print("\n  RECOMMENDATION: Try XGB+LGBM ensemble for mixed results")
        else:
            print("\n  RECOMMENDATION: Stick with XGBoost (LGBM not consistently better)")


if __name__ == "__main__":
    run()
