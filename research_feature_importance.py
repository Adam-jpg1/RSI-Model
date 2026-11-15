"""
Feature Importance Analysis

Uses XGBoost's built-in feature_importances_ (gain-based) to identify which
of the 100+ features are actually used by trained models.

Insight: if the model uses only 30-40 features, adding 60 unused ones increases
overfitting risk (N/M ratio for XGBoost regularization). A curated reduced
feature set may generalize better than the full 100+ set.

Output:
  - Per-ticker top 20 features by gain importance
  - Cross-ticker consensus: features important for 3+ tickers
  - Recommendation: features to keep vs drop
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import defaultdict

from data.fetcher import get_price_data, get_daily_data
from data.indicators import (add_all_indicators, add_daily_trend_features,
                              add_market_features, add_cross_asset_features,
                              add_pead_features, add_peer_features,
                              add_macro_event_features, add_calendar_features)
from ml.features import get_feature_matrix, build_target, target_bars_for_interval
from ml.model import RSIModel
from ml.optimizer import load_best_params

TICKERS = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD  = "2y"
INTERVAL = "1h"


def build_full_feature_df(ticker: str) -> pd.DataFrame:
    df = get_price_data(ticker, period=PERIOD, interval=INTERVAL)
    df = add_all_indicators(df)

    try:
        daily_df = get_daily_data(ticker, period="2y")
        df = add_daily_trend_features(df, daily_df)
    except Exception:
        pass
    try:
        for sym in ["NVDA", "XLK", "^VIX"]:
            if sym != ticker:
                try:
                    peer_df = get_price_data(sym, period=PERIOD, interval=INTERVAL)
                except Exception:
                    peer_df = None
            else:
                peer_df = None
        df = add_peer_features(df,
                                nvda_hourly_df=get_price_data("NVDA", period=PERIOD, interval=INTERVAL),
                                xlk_hourly_df=get_price_data("XLK",  period=PERIOD, interval=INTERVAL),
                                vix_hourly_df=get_price_data("^VIX", period=PERIOD, interval=INTERVAL))
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
        spy_h = get_price_data("SPY", period=PERIOD, interval=INTERVAL)
        df = add_market_features(df, spy_h)
    except Exception:
        pass
    try:
        df = add_cross_asset_features(df)
    except Exception:
        pass
    try:
        df = add_pead_features(df, ticker)
    except Exception:
        pass

    features = get_feature_matrix(df, 0.0).dropna()
    return features


def get_feature_importance(ticker: str, features: pd.DataFrame) -> pd.Series:
    """Train a model with best params and return feature importance (gain)."""
    params = load_best_params(ticker)
    if params is None:
        return pd.Series(dtype=float)

    target = build_target(
        get_price_data(ticker, period=PERIOD, interval=INTERVAL),
        n_bars=target_bars_for_interval(INTERVAL)
    )

    common = features.index.intersection(target.dropna().index)
    X = features.loc[common]
    y = target.loc[common]

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    if len(X) < 200:
        return pd.Series(dtype=float)

    from config import CONFIG
    orig_xgb = CONFIG.ml.xgb_params.copy()
    if "xgb_params" in params:
        CONFIG.ml.xgb_params.update(params["xgb_params"])

    model = RSIModel()
    model.build()
    model.fit(X, y)

    CONFIG.ml.xgb_params.update(orig_xgb)

    # Extract feature importances (gain = sum of gain from splits using this feature)
    # XGBoost assigns generic names (f0, f1, ...) after scaler strips column names;
    # map back to real feature names using model.feature_names
    feat_names = model.feature_names  # list of real column names
    importances = {}
    for clf in [model.model] + model._ensemble:
        booster = clf.get_booster()
        fi = booster.get_score(importance_type="gain")
        for fkey, score in fi.items():
            # fkey is 'fN' where N is the column index
            try:
                real_name = feat_names[int(fkey[1:])]
            except (ValueError, IndexError):
                real_name = fkey
            importances[real_name] = importances.get(real_name, 0.0) + score

    if not importances:
        return pd.Series(dtype=float)

    imp_series = pd.Series(importances)
    return (imp_series / imp_series.sum()).sort_values(ascending=False)


def run():
    print("=" * 70)
    print("  Feature Importance Analysis")
    print("=" * 70)

    all_importances = {}
    feature_counts = defaultdict(int)
    feature_total_imp = defaultdict(float)

    for ticker in TICKERS:
        print(f"\n[fi] {ticker}...", flush=True)
        try:
            features = build_full_feature_df(ticker)
            print(f"  {len(features.columns)} features, {len(features)} bars")
            imp = get_feature_importance(ticker, features)
            if imp.empty:
                print("  (no importances returned)")
                continue
            all_importances[ticker] = imp

            # Aggregate: count how many tickers use each feature in top-20
            top20 = set(imp.head(20).index)
            for feat in top20:
                feature_counts[feat] += 1
            for feat, v in imp.items():
                feature_total_imp[feat] += v

            print(f"  Top 10 features:")
            for feat, v in imp.head(10).items():
                print(f"    {feat:<30} {v:.4f} ({v*100:.1f}%)")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}")

    if not all_importances:
        print("No importances computed.")
        return

    print("\n\n" + "=" * 70)
    print("  CONSENSUS: Features in top-20 for 3+ tickers")
    print("=" * 70)
    print(f"  {'Feature':<32} {'Tickers':>8} {'Avg_imp%':>10} {'Tickers_list'}")
    print("  " + "-" * 65)

    consensus = [(f, c, feature_total_imp[f] / len(all_importances))
                 for f, c in feature_counts.items() if c >= 3]
    consensus.sort(key=lambda x: x[2], reverse=True)

    for feat, count, avg_imp in consensus:
        tickers_using = [t for t, imp in all_importances.items()
                         if feat in imp.head(20).index]
        print(f"  {feat:<32} {count:>8}  {avg_imp*100:>9.2f}%  {tickers_using}")

    print(f"\n\nFeatures appearing in top-20 for >=3 tickers: {len(consensus)}")

    # Zero-importance features (never used by any model)
    all_features = set()
    for imp in all_importances.values():
        all_features.update(imp.index)

    # Features with very low total importance across all tickers
    low_imp = [(f, feature_total_imp[f] / len(all_importances))
               for f in all_features
               if feature_total_imp.get(f, 0) / len(all_importances) < 0.001]
    low_imp.sort(key=lambda x: x[1])

    print(f"\nFeatures with avg importance <0.1% (potential noise): {len(low_imp)}")
    for feat, imp in low_imp[:20]:
        print(f"  {feat:<35} {imp*100:.4f}%")

    print("\n\nRECOMMENDATION:")
    keep = [f for f, _, _ in consensus]
    print(f"  Top consensus features (in top-20 for >=3 tickers): {len(keep)}")
    print(f"  Could reduce feature set from {len(all_features)} to ~{len(keep)+15} features")
    print(f"  (consensus + a few tickers' unique top features)")


if __name__ == "__main__":
    run()
