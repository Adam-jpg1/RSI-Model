import pandas as pd
import numpy as np
from typing import List, Dict
from data.fetcher import get_price_data, get_stock_news
from data.indicators import add_all_indicators
from sentiment.analyzer import analyze_headlines
from ml.features import prepare_dataset, time_series_split
from ml.model import RSIModel
from config import CONFIG


def train_on_ticker(ticker: str, period: str = "180d", interval: str = "5m") -> Dict:
    print(f"\n[train] Fetching {ticker} price data ({period}, {interval})...")
    df = get_price_data(ticker, period=period, interval=interval)
    df = add_all_indicators(df)

    print(f"[train] Fetching news sentiment for {ticker}...")
    headlines = get_stock_news(ticker)
    sentiment = analyze_headlines(headlines)
    sentiment_score = sentiment["score"]
    print(f"[train] Sentiment: {sentiment['label']} ({sentiment_score:+.3f}, n={sentiment['n_articles']})")

    X, y = prepare_dataset(df, sentiment_score=sentiment_score)
    print(f"[train] Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"[train] Label distribution: {y.value_counts().to_dict()}")

    splits = time_series_split(X, y)
    cv_scores = []
    for i, (X_tr, y_tr, X_v, y_v) in enumerate(splits):
        m = RSIModel()
        m.build()
        m.fit(X_tr, y_tr, X_v, y_v)
        result = m.evaluate(X_v, y_v)
        cv_scores.append(result["accuracy"])
        print(f"  Fold {i+1}: accuracy={result['accuracy']:.4f}")

    mean_cv = float(np.mean(cv_scores))
    print(f"[train] CV mean accuracy: {mean_cv:.4f}")

    n = len(X)
    split_idx = int(n * (CONFIG.ml.train_ratio + CONFIG.ml.val_ratio))
    X_train_full = X.iloc[:split_idx]
    y_train_full = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    final_model = RSIModel()
    final_model.build()
    final_model.fit(X_train_full, y_train_full)
    test_result = final_model.evaluate(X_test, y_test)
    print(f"[train] Test accuracy: {test_result['accuracy']:.4f}")

    fi = final_model.feature_importance()
    print(f"[train] Top 5 features:\n{fi.head(5).to_string(index=False)}")

    final_model.save()
    return {
        "ticker": ticker,
        "cv_accuracy": mean_cv,
        "test_accuracy": test_result["accuracy"],
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
