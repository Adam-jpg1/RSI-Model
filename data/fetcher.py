import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from config import CONFIG


def get_price_data(ticker: str, period: str = None, interval: str = None) -> pd.DataFrame:
    period = period or CONFIG.data.default_period
    interval = interval or CONFIG.data.default_interval
    df = yf.download(ticker, period=period, interval=interval, multi_level_index=False, progress=False)
    df.dropna(inplace=True)
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


def get_multi_ticker_data(tickers: List[str], period: str = None, interval: str = None) -> Dict[str, pd.DataFrame]:
    return {t: get_price_data(t, period, interval) for t in tickers}
