import os
import pickle
import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from config import CONFIG


_MAX_PERIOD_DAYS = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730, "1h": 730, "90m": 60}

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")
_CACHE_TTL_SECONDS = 3600  # 1-hour cache; fresh enough for research runs


def _cache_path(ticker: str, period: str, interval: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    key = f"{ticker}_{period}_{interval}".replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{key}.pkl")


def _load_cache(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(path):
            age = time.time() - os.path.getmtime(path)
            if age < _CACHE_TTL_SECONDS:
                with open(path, "rb") as f:
                    return pickle.load(f)
    except Exception:
        pass
    return None


def _save_cache(path: str, df: pd.DataFrame):
    try:
        with open(path, "wb") as f:
            pickle.dump(df, f)
    except Exception:
        pass


def _cap_period(period: str, interval: str) -> str:
    max_days = _MAX_PERIOD_DAYS.get(interval)
    if max_days is None:
        return period
    try:
        if period.endswith("d"):
            days = int(period[:-1])
        elif period.endswith("y"):
            days = int(period[:-1]) * 365
        elif period.endswith("mo"):
            days = int(period[:-2]) * 30
        else:
            return period
    except ValueError:
        return period
    if days > max_days:
        print(f"[data] {interval} bars limited to {max_days}d by Yahoo Finance — using {max_days}d instead of {period}")
        return f"{max_days}d"
    return period


def get_price_data(ticker: str, period: str = None, interval: str = None) -> pd.DataFrame:
    period = period or CONFIG.data.default_period
    interval = interval or CONFIG.data.default_interval
    period = _cap_period(period, interval)
    cpath = _cache_path(ticker, period, interval)
    cached = _load_cache(cpath)
    if cached is not None:
        return cached
    df = yf.download(ticker, period=period, interval=interval, multi_level_index=False, progress=False)
    df.dropna(inplace=True)
    if len(df) > 0:
        _save_cache(cpath, df)
    return df


def get_stock_news(ticker: str) -> List[Dict]:
    stock = yf.Ticker(ticker)
    raw_news = stock.news or []
    headlines = []

    for article in raw_news:
        content = article.get("content", {})
        if isinstance(content, dict):
            title = content.get("title", "")
            summary = content.get("summary", "")
            pub_date = content.get("pubDate", "")
            provider_obj = content.get("provider", {})
            provider = provider_obj.get("displayName", "unknown") if isinstance(provider_obj, dict) else "unknown"
        else:
            title = article.get("title", "")
            summary = article.get("summary", "")
            pub_date = article.get("providerPublishTime", "")
            provider = article.get("publisher", "unknown")

        if title:
            headlines.append({
                "title": title,
                "summary": summary,
                "pub_date": pub_date,
                "provider": provider,
                "text": f"{title}. {summary}".strip(". "),
            })

    return headlines


def get_company_info(ticker: str) -> Dict:
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "Unknown"),
        "market_cap": info.get("marketCap", 0),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice", None),
        "52w_high": info.get("fiftyTwoWeekHigh", None),
        "52w_low": info.get("fiftyTwoWeekLow", None),
        "avg_volume": info.get("averageVolume", None),
        "beta": info.get("beta", None),
    }


def get_daily_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch daily OHLCV bars. Used for multi-timeframe trend features."""
    cpath = _cache_path(ticker, period, "1d")
    cached = _load_cache(cpath)
    if cached is not None:
        return cached
    df = yf.download(ticker, period=period, interval="1d", multi_level_index=False, progress=False)
    df.dropna(inplace=True)
    _save_cache(cpath, df)
    return df


def get_earnings_dates(ticker: str) -> List:
    """Return sorted list of historical earnings date objects for ticker.
    Cached for 24h (earnings dates change infrequently)."""
    cpath = _cache_path(ticker, "earnings", "dates")
    cached = _load_cache(cpath)
    if cached is not None:
        return cached
    try:
        stock = yf.Ticker(ticker)
        ed = stock.earnings_dates
        if ed is None or ed.empty:
            return []
        # Extract timezone-naive dates from the DatetimeIndex
        dates = sorted({pd.Timestamp(str(ts)[:10]).date() for ts in ed.index})
        _save_cache(cpath, dates)
        return dates
    except Exception:
        return []


def get_multi_ticker_data(tickers: List[str], period: str = None, interval: str = None) -> Dict[str, pd.DataFrame]:
    return {t: get_price_data(t, period, interval) for t in tickers}
