import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional
from functools import lru_cache
from config import CONFIG


@lru_cache(maxsize=1)
def _load_pipeline():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "text-classification",
        model=CONFIG.sentiment.model_name,
        tokenizer=CONFIG.sentiment.model_name,
        top_k=None,
    )


_LABEL_MAP = {"positive": 1, "negative": -1, "neutral": 0}


def analyze_text(text: str) -> Dict:
    finbert = _load_pipeline()
    truncated = text[: CONFIG.sentiment.max_length]
    results = finbert(truncated)[0]
    top = max(results, key=lambda x: x["score"])
    polarity_scores = {r["label"].lower(): r["score"] for r in results}
    weighted_polarity = sum(
        _LABEL_MAP.get(label, 0) * score for label, score in polarity_scores.items()
    )
    return {
        "label": top["label"].lower(),
        "polarity": _LABEL_MAP.get(top["label"].lower(), 0),
        "confidence": round(top["score"], 4),
        "weighted_polarity": round(weighted_polarity, 4),
        "scores": polarity_scores,
    }


def analyze_headlines(headlines: List[Dict]) -> Dict:
    if not headlines:
        return {"score": 0.0, "label": "neutral", "n_articles": 0, "articles": []}

    cfg = CONFIG.sentiment
    results = []
    now_ts = datetime.now(timezone.utc).timestamp()

    for i, h in enumerate(headlines):
        text = h.get("text") or h.get("title", "")
        if not text:
            continue

        sentiment = analyze_text(text)

        pub_raw = h.get("pub_date", "")
        age_hours = _parse_age_hours(pub_raw, now_ts)
        decay_weight = cfg.sentiment_decay ** (age_hours / cfg.news_lookback_hours)

        results.append({
            "title": h.get("title", ""),
            "provider": h.get("provider", "unknown"),
            "pub_date": pub_raw,
            "age_hours": round(age_hours, 1),
            "sentiment": sentiment["label"],
            "polarity": sentiment["polarity"],
            "weighted_polarity": sentiment["weighted_polarity"],
            "confidence": sentiment["confidence"],
            "decay_weight": round(decay_weight, 4),
            "adjusted_score": round(sentiment["weighted_polarity"] * decay_weight * sentiment["confidence"], 4),
        })

    if not results:
        return {"score": 0.0, "label": "neutral", "n_articles": 0, "articles": []}

    total_weight = sum(r["decay_weight"] * r["confidence"] for r in results)
    if total_weight > 0:
        agg_score = sum(r["adjusted_score"] for r in results) / total_weight
    else:
        agg_score = 0.0

    agg_score = float(np.clip(agg_score, -1.0, 1.0))

    if agg_score > 0.1:
        label = "positive"
    elif agg_score < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return {
        "score": round(agg_score, 4),
        "label": label,
        "n_articles": len(results),
        "articles": results,
    }


def _parse_age_hours(pub_raw, now_ts: float) -> float:
    if not pub_raw:
        return 0.0
    if isinstance(pub_raw, (int, float)):
        pub_ts = float(pub_raw)
    else:
        try:
            from dateutil import parser as dateparser
            pub_dt = dateparser.parse(str(pub_raw))
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            pub_ts = pub_dt.timestamp()
        except Exception:
            return 0.0
    return max(0.0, (now_ts - pub_ts) / 3600.0)


def get_portfolio_sentiment(ticker_headlines: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    return {ticker: analyze_headlines(headlines) for ticker, headlines in ticker_headlines.items()}
