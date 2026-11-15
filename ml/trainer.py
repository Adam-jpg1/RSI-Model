import pandas as pd
import numpy as np
from typing import Dict, List

from config import CONFIG
from data.fetcher import get_price_data, get_daily_data, get_stock_news
from data.indicators import (add_all_indicators, add_daily_trend_features,
                              add_cross_asset_features, add_pead_features)
from sentiment.analyzer import analyze_headlines
from ml.features import prepare_dataset, time_series_split, CLASS_LABELS, target_bars_for_interval, compute_class_weights
from ml.model import RSIModel
from ml.optimizer import load_best_params


def train_on_ticker(ticker: str, period: str = None, interval: str = "1h") -> Dict:
    if period is None:
        period = CONFIG.data.train_period
    print(f"\n[train] Fetching {ticker} ({period}, {interval})...")
    df = get_price_data(ticker, period=period, interval=interval)
    df = add_all_indicators(df)

    print(f"[train] Fetching daily data for multi-timeframe features...")
    try:
        daily_df = get_daily_data(ticker, period=period)
        try:
            spy_daily = get_daily_data("SPY", period=period)
        except Exception:
            spy_daily = None
        try:
            vix_daily = get_daily_data("^VIX", period=period)
        except Exception:
            vix_daily = None
        df = add_daily_trend_features(
            df, daily_df,
            compute_weekly=True,
            spy_daily_df=spy_daily,
            vix_daily_df=vix_daily,
        )
        spy_vix_note = f"SPY={'yes' if spy_daily is not None else 'no'} VIX={'yes' if vix_daily is not None else 'no'}"
        print(f"[train] Daily features merged ({spy_vix_note}, weekly=True)")
        # Cross-asset macro features
        try:
            ca = {}
            for sym in ["TLT", "HYG", "GLD", "UUP", "^VIX9D", "^VIX3M"]:
                try:
                    ca[sym] = get_daily_data(sym, period=period)
                except Exception:
                    ca[sym] = None
            df = add_cross_asset_features(
                df,
                tlt_daily_df=ca.get("TLT"), hyg_daily_df=ca.get("HYG"),
                gld_daily_df=ca.get("GLD"), uup_daily_df=ca.get("UUP"),
                vix9d_daily_df=ca.get("^VIX9D"), vix3m_daily_df=ca.get("^VIX3M"),
            )
            print(f"[train] Cross-asset features added (TLT/HYG/GLD/UUP/VIX9D/VIX3M)")
        except Exception as e:
            print(f"[train] Warning: cross-asset features failed: {e}")
        # PEAD proxy
        try:
            df = add_pead_features(df, daily_df)
            print(f"[train] PEAD features added")
        except Exception as e:
            print(f"[train] Warning: PEAD features failed: {e}")
    except Exception as e:
        print(f"[train] Warning: daily features unavailable for {ticker}: {e}")

    print(f"[train] Fetching news sentiment for {ticker}...")
    try:
        headlines = get_stock_news(ticker)
        sentiment = analyze_headlines(headlines)
    except Exception:
        sentiment = {"label": "neutral", "score": 0.0, "n_articles": 0}
    sentiment_score = sentiment["score"]
    print(f"[train] Sentiment: {sentiment['label']} ({sentiment_score:+.3f}, n={sentiment['n_articles']})")

    # Apply Optuna XGB params if available (ADX threshold + tuned architecture)
    best = load_best_params(ticker)
    if best:
        orig_params = {k: v for k, v in CONFIG.ml.xgb_params.items()}
        CONFIG.ml.xgb_params.update(best["xgb_params"])
        adx_note = f"  adx_opt={best['adx_filter_min']:.1f}"
        print(f"[train] Using Optuna params for {ticker} (score={best['best_score']:.4f}){adx_note}")
    else:
        orig_params = None

    n_bars = target_bars_for_interval(interval)
    print(f"[train] target_bars={n_bars} (interval={interval}, ~{n_bars * {'1h':60,'15m':15,'5m':5,'1d':1440}.get(interval,60)//60}h horizon)")
    X, y = prepare_dataset(df, sentiment_score=sentiment_score, interval=interval)
    counts = y.value_counts().to_dict()
    pcts = {CLASS_LABELS[k]: f"{v}({v/len(y)*100:.0f}%)" for k, v in counts.items()}
    print(f"[train] Dataset: {len(X)} samples, {X.shape[1]} features  labels: {pcts}")

    if len(X) < 200:
        if orig_params:
            CONFIG.ml.xgb_params.clear()
            CONFIG.ml.xgb_params.update(orig_params)
        raise ValueError(f"Not enough data for {ticker}: only {len(X)} samples")

    splits = time_series_split(X, y)
    cv_scores, cv_f1s, cv_up_prec = [], [], []
    for i, (X_tr, y_tr, X_v, y_v) in enumerate(splits):
        m = RSIModel()
        m.build()
        m.fit(X_tr, y_tr, sample_weight=compute_class_weights(y_tr))
        result = m.evaluate(X_v, y_v)
        cv_scores.append(result["accuracy"])
        cv_f1s.append(result["f1_macro"])
        up_prec = result["report"].get("UP", {}).get("precision", 0.0)
        cv_up_prec.append(up_prec)
        per_class = {k: (round(v["precision"], 3), round(v["recall"], 3))
                     for k, v in result["report"].items() if k in ("DOWN", "FLAT", "UP")}
        print(f"  Fold {i+1}: acc={result['accuracy']:.3f}  f1={result['f1_macro']:.3f}  "
              f"UP_prec={up_prec:.3f}  (prec,rec) {per_class}")

    mean_cv = float(np.mean(cv_scores))
    mean_f1 = float(np.mean(cv_f1s))
    mean_up_prec = float(np.mean(cv_up_prec))
    print(f"[train] CV mean — accuracy: {mean_cv:.4f}  f1_macro: {mean_f1:.4f}  UP_precision: {mean_up_prec:.4f}")

    n = len(X)
    split_idx = int(n * (CONFIG.ml.train_ratio + CONFIG.ml.val_ratio))
    n_bars = target_bars_for_interval(interval)  # used for final-model mask below
    X_train_full = X.iloc[:split_idx]
    y_train_full = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    final_model = RSIModel()
    final_model.build()
    final_model.fit(X_train_full, y_train_full, sample_weight=compute_class_weights(y_train_full))
    test_result = final_model.evaluate(X_test, y_test)
    print(f"[train] Test — accuracy: {test_result['accuracy']:.4f}  f1_macro: {test_result['f1_macro']:.4f}"
          f"  trees: {final_model.n_trees()}")

    for label in ("DOWN", "FLAT", "UP"):
        v = test_result["report"].get(label, {})
        print(f"  {label:4s}: precision={v.get('precision',0):.3f}  recall={v.get('recall',0):.3f}  f1={v.get('f1-score',0):.3f}")
    up_prec = test_result["report"].get("UP", {}).get("precision", 0)
    baseline = y_test.value_counts(normalize=True).get(2, 0)
    print(f"[train] UP precision vs base rate: {up_prec:.3f} vs {baseline:.3f} (+{up_prec - baseline:.3f})")

    fi = final_model.feature_importance(top_n=10)
    print(f"[train] Top 10 features:\n{fi.to_string(index=False)}")

    if orig_params:
        CONFIG.ml.xgb_params.clear()
        CONFIG.ml.xgb_params.update(orig_params)

    final_model.save(ticker=ticker)
    return {
        "ticker": ticker,
        "cv_accuracy": mean_cv,
        "cv_f1_macro": mean_f1,
        "test_accuracy": test_result["accuracy"],
        "test_f1_macro": test_result["f1_macro"],
        "report": test_result["report"],
        "feature_importance": fi,
        "model": final_model,
    }


def train_multi_ticker(tickers: List[str] = None) -> Dict[str, RSIModel]:
    tickers = tickers or CONFIG.tickers
    models = {}
    for t in tickers:
        result = train_on_ticker(t)
        models[t] = result["model"]
    return models
