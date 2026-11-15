"""
SHAP Feature Importance Analysis

Standard XGBoost feature importance (split-count / gain) is misleading:
  - Correlated features split their importance across each other
  - High-cardinality features get artificially high split counts
  - Shows all 144 features at ~0.7-2%, making selection impossible

SHAP (SHapley Additive exPlanations) gives the true marginal contribution
of each feature to each prediction. Specifically:
  - mean(|SHAP|) per feature = average impact on model output
  - Zero SHAP = feature provides NO information to this model
  - Consistent rank across tickers = robust signal vs. ticker-specific noise

Plan:
  1. Train model on first 75% of data (using saved Wave 3 params)
  2. Compute SHAP on the remaining 25% (test set)
  3. Rank features by mean(|SHAP|) across UP/DOWN/FLAT classes
  4. Identify: top-20 features, zero-SHAP features (prunable), ticker-consistent signals

Output: a recommended FEATURE_COLS_PRUNED list removing near-zero contributors
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
from config import CONFIG

TICKERS  = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD   = "2y"
INTERVAL = "1h"

try:
    import shap
    SHAP_OK = True
except ImportError:
    print("shap not installed — run: pip install shap")
    SHAP_OK = False


def build_full_features(ticker: str, all_close: dict) -> tuple:
    """Build complete feature matrix for a ticker."""
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
    features = features[mask]
    target = target[mask]
    return features, target


def run():
    if not SHAP_OK:
        return

    print("=" * 70)
    print("  SHAP Feature Importance Analysis")
    print(f"  Tickers: {TICKERS}")
    print(f"  Goal: identify which of 144 features actually drive predictions")
    print("=" * 70)

    # Pre-fetch Close series for cross-sectional features
    print("\nPre-fetching Close series...")
    all_close = {}
    for sym in TICKERS:
        try:
            df = get_price_data(sym, period=PERIOD, interval=INTERVAL)
            all_close[sym] = df["Close"].rename(sym)
            print(f"  {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym}: ERROR {e}")
            all_close[sym] = None

    all_shap_scores = {}  # feature -> list of mean|shap| scores across tickers

    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")
        try:
            X, y = build_full_features(ticker, all_close)
        except Exception as e:
            print(f"  ERROR building features: {e}")
            continue
        print(f"  Feature matrix: {X.shape[0]} rows x {X.shape[1]} cols")

        # Train/test split: first 75% train, last 25% test
        split = int(len(X) * 0.75)
        X_tr, y_tr = X.iloc[:split], y.iloc[:split]
        X_te, y_te = X.iloc[split:], y.iloc[split:]

        # Apply saved params if available
        saved = load_best_params(ticker)
        orig_xgb = {}
        if saved and "xgb_params" in saved:
            orig_xgb = {k: CONFIG.ml.xgb_params[k]
                        for k in saved["xgb_params"]
                        if k in CONFIG.ml.xgb_params}
            CONFIG.ml.xgb_params.update(saved["xgb_params"])

        try:
            model = RSIModel()
            model.build(early_stopping_rounds=20)
            val_cut = int(len(X_tr) * 0.85)
            X_val_es = X_tr.iloc[val_cut:]
            y_val_es = y_tr.iloc[val_cut:]
            X_tr_fit = X_tr.iloc[:val_cut]
            y_tr_fit = y_tr.iloc[:val_cut]
            model.fit(X_tr_fit, y_tr_fit, X_val=X_val_es, y_val=y_val_es)

            feat_names = model.feature_names
            n_feat = len(feat_names)

            # Sample up to 500 test rows for SHAP (TreeExplainer is O(n))
            sample_size = min(500, len(X_te))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_te), size=sample_size, replace=False)
            X_shap = X_te.iloc[idx]

            X_shap_scaled = model.scaler.transform(X_shap[feat_names])
            explainer = shap.TreeExplainer(model.model)
            shap_vals = explainer.shap_values(X_shap_scaled)

            # shap_vals format varies by shap version:
            #   list of (n_samples, n_features) per class  — older shap
            #   ndarray (n_samples, n_features, n_classes)  — newer shap
            if isinstance(shap_vals, list):
                mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
            elif shap_vals.ndim == 3:
                # (n_samples, n_features, n_classes) → mean over samples and classes
                mean_abs_shap = np.abs(shap_vals).mean(axis=(0, 2))
            else:
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)

            feat_shap = pd.Series(mean_abs_shap, index=feat_names).sort_values(ascending=False)

            print(f"\n  Top 25 features by SHAP:")
            for i, (f, v) in enumerate(feat_shap.head(25).items(), 1):
                bar = "#" * int(v / feat_shap.iloc[0] * 30)
                print(f"    {i:>2}. {f:<40} {v:.5f} {bar}")

            zero_shap = (feat_shap == 0.0).sum()
            near_zero = (feat_shap < feat_shap.iloc[0] * 0.01).sum()
            print(f"\n  Total features: {n_feat}")
            print(f"  Zero SHAP:  {zero_shap}")
            print(f"  Near-zero (<1% of max): {near_zero}")

            # Accumulate scores
            for f, v in feat_shap.items():
                if f not in all_shap_scores:
                    all_shap_scores[f] = []
                all_shap_scores[f].append(float(v))

        except Exception as e:
            print(f"  SHAP error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if orig_xgb:
                CONFIG.ml.xgb_params.update(orig_xgb)

    # Cross-ticker consensus
    print("\n\n" + "=" * 70)
    print("  CROSS-TICKER SHAP CONSENSUS")
    print("  Features ranked by mean SHAP score across all tickers where present")
    print("=" * 70)

    consensus_df = pd.DataFrame([
        {
            "feature": f,
            "mean_shap": np.mean(scores),
            "n_tickers": len(scores),
            "min_shap": np.min(scores),
            "max_shap": np.max(scores),
        }
        for f, scores in all_shap_scores.items()
    ]).sort_values("mean_shap", ascending=False).reset_index(drop=True)

    print(f"\n  Top 40 by cross-ticker mean|SHAP|:")
    print(f"  {'Rank':<5} {'Feature':<40} {'MeanSHAP':>10} {'N':>4} {'Min':>10} {'Max':>10}")
    print("  " + "-" * 80)
    for i, row in consensus_df.head(40).iterrows():
        print(f"  {i+1:<5} {row['feature']:<40} {row['mean_shap']:>10.5f} "
              f"{row['n_tickers']:>4} {row['min_shap']:>10.5f} {row['max_shap']:>10.5f}")

    # Features that are consistently near-zero across all tickers (safe to prune)
    threshold = consensus_df["mean_shap"].quantile(0.10)  # bottom 10%
    prunable = consensus_df[
        (consensus_df["mean_shap"] < threshold) &
        (consensus_df["n_tickers"] >= 3)
    ].sort_values("mean_shap")

    print(f"\n  Potentially prunable (bottom 10% SHAP, present in 3+ tickers):")
    print(f"  Threshold: mean|SHAP| < {threshold:.5f}")
    if prunable.empty:
        print("  None found.")
    else:
        for _, row in prunable.iterrows():
            print(f"    {row['feature']:<40}  mean|SHAP|={row['mean_shap']:.5f}  n={row['n_tickers']}")

    # Features that appear ONLY for 1 ticker (possibly ticker-specific overfit)
    single_ticker = consensus_df[consensus_df["n_tickers"] == 1].sort_values("mean_shap", ascending=False)
    print(f"\n  Single-ticker features (potential overfit): {len(single_ticker)}")

    # Top 50 kept features (by cross-ticker mean SHAP)
    top50 = consensus_df[consensus_df["n_tickers"] >= 2].head(50)
    print(f"\n\n  RECOMMENDED CORE FEATURES (top-50 by cross-ticker SHAP, present in 2+ tickers):")
    print("  Add this list to FEATURE_COLS_PRUNED in ml/features.py")
    print()
    print("  FEATURE_COLS_PRUNED = [")
    for _, row in top50.iterrows():
        print(f"      \"{row['feature']}\",  # mean|SHAP|={row['mean_shap']:.5f}  n={row['n_tickers']}")
    print("  ]")

    # Save full results
    consensus_df.to_csv("shap_results.csv", index=False)
    print(f"\n  Full results saved to shap_results.csv")
    print(f"  Total unique features across tickers: {len(consensus_df)}")


if __name__ == "__main__":
    run()
