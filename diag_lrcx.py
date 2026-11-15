"""
Diagnostic: check why LRCX Optuna optimizer always returns -999.
Checks hourly model prediction distribution and daily gate interactions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from config import CONFIG

CONFIG.bot.min_confidence = 0.0

from data.fetcher import get_price_data, get_daily_data
from data.indicators import add_all_indicators, add_daily_trend_features
from ml.features import get_feature_matrix, build_target, target_bars_for_interval
from ml.model import RSIModel
from backtest.simulate import _build_daily_signals
from ml.optimizer import load_best_params

ticker = "LRCX"
interval = "1h"
n_folds = 4
period = "2y"

print(f"=== LRCX Hourly Model Diagnostic ===\n")

# Load data
df = get_price_data(ticker, period=period, interval=interval)
df = add_all_indicators(df)

daily_df_raw = get_daily_data(ticker, period="5y")
daily_df_raw = add_all_indicators(daily_df_raw)
spy_daily_raw = get_daily_data("SPY", period="5y")
try:
    vix_daily_raw = get_daily_data("^VIX", period="5y")
except Exception:
    vix_daily_raw = None

df = add_daily_trend_features(df, daily_df_raw, spy_daily_df=spy_daily_raw, vix_daily_df=vix_daily_raw)

features_full = get_feature_matrix(df, 0.0).dropna()
df_aligned = df.loc[features_full.index]

print(f"Total hourly rows after dropna: {len(features_full)}")
print(f"Date range: {features_full.index[0]} -> {features_full.index[-1]}")

n = len(features_full)
fold_size = n // (n_folds + 1)
print(f"Fold size: {fold_size} bars each\n")

# Check target label distribution for each fold's training data
print("=== Target label distribution per fold training set ===")
n_bars = target_bars_for_interval(interval)
for fold in range(1, n_folds + 1):
    train_end = fold * fold_size
    y_tr = build_target(df_aligned.iloc[:train_end], n_bars=n_bars)
    counts = y_tr.value_counts().sort_index()
    total = len(y_tr)
    print(f"Fold {fold} train (n={train_end}): DOWN={counts.get(0,0)} ({100*counts.get(0,0)/total:.0f}%)  "
          f"FLAT={counts.get(1,0)} ({100*counts.get(1,0)/total:.0f}%)  "
          f"UP={counts.get(2,0)} ({100*counts.get(2,0)/total:.0f}%)")

print()

# Pre-build daily signals using saved params (as optimizer does)
existing = load_best_params(ticker, daily_confirm=True)
_daily_xgb = existing.get("xgb_params") if existing else None
print(f"Existing params for daily model: {_daily_xgb is not None}")
if _daily_xgb:
    orig = {k: v for k, v in CONFIG.ml.xgb_params.items()}
    CONFIG.ml.xgb_params.update(_daily_xgb)

daily_sigs_per_fold = []
for fold in range(1, n_folds + 1):
    train_end = fold * fold_size
    train_end_ts = features_full.index[train_end - 1]
    sigs = _build_daily_signals(daily_df_raw, train_end_ts)
    daily_sigs_per_fold.append(sigs)

if _daily_xgb:
    CONFIG.ml.xgb_params.clear()
    CONFIG.ml.xgb_params.update(orig)

# For each fold, train a model and check prediction distribution
print("=== Hourly model prediction distribution per fold ===")
for fold in range(1, n_folds + 1):
    train_end = fold * fold_size
    test_start = train_end
    test_end = min((fold + 1) * fold_size, n)

    X_tr = features_full.iloc[:train_end]
    y_tr = build_target(df_aligned.iloc[:train_end], n_bars=n_bars)
    mask = X_tr.notna().all(axis=1) & y_tr.notna()
    X_tr, y_tr = X_tr[mask], y_tr[mask]

    # Early stopping split
    val_cut = int(len(X_tr) * 0.85)
    X_val_es = X_tr.iloc[val_cut:]
    y_val_es = y_tr.iloc[val_cut:]
    X_tr = X_tr.iloc[:val_cut]
    y_tr = y_tr.iloc[:val_cut]

    model = RSIModel()
    model.build(early_stopping_rounds=20)
    model.fit(X_tr, y_tr, X_val=X_val_es, y_val=y_val_es)

    feat_test = features_full.iloc[test_start:test_end]
    df_test = df_aligned.iloc[test_start:test_end]

    # Get predictions for all test rows
    preds_list = []
    probs_up_list = []
    for idx, row in feat_test.iterrows():
        try:
            pred = model.predict_single(row)
            preds_list.append(pred["signal"])
            probs_up_list.append(pred["probs"].get("UP", 0))
        except Exception as e:
            preds_list.append("ERROR")
            probs_up_list.append(0)

    pred_counts = {s: preds_list.count(s) for s in ["DOWN", "FLAT", "UP", "ERROR"]}
    up_probs = [p for p in probs_up_list if p > 0]

    # Check how many UP predictions survive daily gate
    daily_sigs = daily_sigs_per_fold[fold - 1]
    up_with_daily_ok = 0
    for idx, pred_signal, up_prob in zip(feat_test.index, preds_list, probs_up_list):
        if pred_signal != "UP":
            continue
        date_key = str(idx)[:10]
        daily_sig = daily_sigs.get(date_key)
        if daily_sig is None or daily_sig == "FLAT" or daily_sig == "UP":
            up_with_daily_ok += 1

    # Check ADX distribution for UP predictions
    adx_vals_at_up = []
    for idx, pred_signal in zip(feat_test.index, preds_list):
        if pred_signal == "UP" and "ADX" in df_aligned.columns:
            adx_val = float(df_aligned.loc[idx, "ADX"]) if idx in df_aligned.index else 0.0
            adx_vals_at_up.append(adx_val)

    print(f"\nFold {fold} test (bars {test_start}->{test_end}, n={test_end-test_start}):")
    print(f"  Predictions: DOWN={pred_counts['DOWN']} FLAT={pred_counts['FLAT']} UP={pred_counts['UP']} ERR={pred_counts['ERROR']}")
    print(f"  UP prob range: min={min(up_probs):.3f} max={max(up_probs):.3f} mean={np.mean(up_probs):.3f}" if up_probs else "  UP prob range: N/A")
    print(f"  UP signals passing daily gate: {up_with_daily_ok}")
    if adx_vals_at_up:
        print(f"  ADX at UP predictions: min={min(adx_vals_at_up):.1f} max={max(adx_vals_at_up):.1f} mean={np.mean(adx_vals_at_up):.1f}")

print("\n=== Done ===")
