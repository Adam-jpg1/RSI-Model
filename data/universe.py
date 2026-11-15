import glob as _glob
import json
import os
import time
from typing import Dict, List, Optional

import pandas as pd

from config import CONFIG


def _load_cache() -> Dict:
    path = CONFIG.universe.cache_path
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        age_hours = (time.time() - data.get("timestamp", 0)) / 3600
        if age_hours < CONFIG.universe.cache_ttl_hours:
            return data
    except Exception:
        pass
    return {}


def _save_cache(data: Dict):
    path = CONFIG.universe.cache_path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data["timestamp"] = time.time()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def _read_html(url: str):
    import requests
    from io import StringIO
    r = requests.get(url, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))


def _fetch_sp500() -> List[str]:
    try:
        tables = _read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return [str(t) for t in tables[0]["Symbol"].tolist()]
    except Exception as e:
        print(f"[universe] Warning: S&P 500 fetch failed: {e}")
        return []


def _fetch_nasdaq100() -> List[str]:
    # Wikipedia restructured the NASDAQ-100 page (~2025) — no longer has a sortable
    # ticker table. S&P 500 already covers all major NASDAQ-100 names, so this is
    # skipped. Re-enable if a reliable data source is added.
    return []


def get_universe(force_refresh: bool = False) -> List[str]:
    """Return combined S&P 500 + NASDAQ-100 ticker list, cached to disk for 24h."""
    if not force_refresh:
        cache = _load_cache()
        if "tickers" in cache:
            age_h = (time.time() - cache.get("timestamp", 0)) / 3600
            print(f"[universe] {len(cache['tickers'])} tickers from cache (age: {age_h:.1f}h)")
            return cache["tickers"]

    symbols: List[str] = []
    if CONFIG.universe.include_sp500:
        sp = _fetch_sp500()
        print(f"[universe] S&P 500: {len(sp)} symbols")
        symbols.extend(sp)
    if CONFIG.universe.include_nasdaq100:
        ndq = _fetch_nasdaq100()
        print(f"[universe] NASDAQ-100: {len(ndq)} symbols")
        symbols.extend(ndq)

    # Deduplicate, clean (dots → dashes for yfinance), drop garbage
    seen = set()
    cleaned = []
    for t in symbols:
        if not isinstance(t, str):
            continue
        t = t.strip().replace(".", "-")
        if not t or len(t) > 6 or t in seen:
            continue
        # Must be letters/dashes only
        if not all(c.isalpha() or c == "-" for c in t):
            continue
        seen.add(t)
        cleaned.append(t)

    cleaned.sort()
    print(f"[universe] Combined universe: {len(cleaned)} unique tickers")
    _save_cache({"tickers": cleaned})
    return cleaned


def get_trained_tickers() -> List[str]:
    """Return tickers that have trained models on disk."""
    paths = _glob.glob("models/xgb_model_*.json")
    return sorted(
        os.path.basename(p).replace("xgb_model_", "").replace(".json", "")
        for p in paths
    )


def get_model_age_hours(ticker: str) -> Optional[float]:
    """Return age in hours of the saved model for ticker, or None if absent."""
    path = f"models/xgb_model_{ticker}.json"
    if not os.path.exists(path):
        return None
    return (time.time() - os.path.getmtime(path)) / 3600
