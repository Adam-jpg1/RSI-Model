"""
News sentiment overlay for the RSI(2) strategy.

Fetches recent headlines via yfinance and scores them with FinBERT
(ProsusAI/finbert).  Returns a per-ticker signal modifier in [0, 1] that
scales down (or blocks) position sizing when sentiment is very negative.

Thresholds
----------
score >= -0.25  NORMAL  : modifier = 1.0
score <  -0.25  REDUCE  : modifier scales 0.10 -> 1.0 linearly
score <  -0.55  BLOCK   : modifier = 0.0  (no new entries)
score <  -0.75  PANIC   : modifier = 0.0  (exit signal — handle in caller)

Backtesting note
----------------
Historical news data is not available via free APIs, so the modifier is
always 1.0 during backtests.  This module is a live/forward-use overlay.

Usage
-----
    from rsi2_news import NewsJudgment
    nj = NewsJudgment()
    modifier, reason = nj.get_modifier("SPY")
    nj.report(["SPY", "QQQ", "IWM"])
"""

import time
import warnings
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import yfinance as yf

# ── lazy-loaded FinBERT pipeline ──────────────────────────────────────────────
_pipeline      = None
_finbert_ok    = None   # None = untested, True = loaded, False = unavailable


def _load_finbert() -> bool:
    """Try to load FinBERT pipeline.  Returns True on success, False if torch/model unavailable."""
    global _pipeline, _finbert_ok
    if _finbert_ok is True:
        return True
    if _finbert_ok is False:
        return False
    try:
        import torch  # noqa: F401 — confirm torch is present before loading pipeline
        from transformers import pipeline as hf_pipeline
        warnings.filterwarnings("ignore")
        print("  Loading FinBERT (first call - downloads ~440 MB once) ...",
              end=" ", flush=True)
        _pipeline = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            return_all_scores=True,
            device=-1,
            truncation=True,
            max_length=512,
        )
        print("OK")
        _finbert_ok = True
        return True
    except Exception as e:
        print(f"unavailable - using keyword scorer ({type(e).__name__})")
        _finbert_ok = False
        return False


# ── financial keyword sentiment (fallback when FinBERT/PyTorch unavailable) ───
# Words scored +1 to +3 (bullish) or -1 to -3 (bearish/crisis).
# Domain-specific terms that generic models like TextBlob miss.
_KW_POSITIVE = {
    # strong positive
    "record": 2, "rally": 2, "surge": 2, "soar": 2, "boom": 2,
    "breakout": 2, "beat": 2, "outperform": 2, "upgrade": 2,
    "buyback": 1, "dividend": 1, "profit": 1, "growth": 1, "gain": 1,
    "recovery": 1, "rebound": 1, "bullish": 2, "bull": 1,
    "strong": 1, "robust": 1, "positive": 1, "optimistic": 1,
    "earnings beat": 3, "higher": 1, "rise": 1, "climbs": 1,
}
_KW_NEGATIVE = {
    # scandal / crisis (heavily weighted)
    "scandal": 3, "fraud": 3, "indictment": 3, "arrested": 3,
    "investigation": 2, "probe": 2, "lawsuit": 2, "sec investigation": 3,
    "bankruptcy": 3, "default": 3, "insolvency": 3, "criminal": 3,
    "affair": 2, "misconduct": 2, "bribery": 3,
    # crash / severe loss
    "collapse": 3, "crash": 3, "plunge": 3, "rout": 3, "sell-off": 2,
    "selloff": 2, "tank": 2, "slump": 2, "tumble": 2,
    # mild negative
    "miss": 1, "downgrade": 2, "cut": 1, "lower": 1, "decline": 1,
    "drop": 1, "fall": 1, "loss": 1, "bearish": 2, "bear": 1,
    "risk": 1, "concern": 1, "warn": 1, "uncertainty": 1,
    "layoff": 2, "layoffs": 2, "tariff": 1, "recession": 2,
}

_NEGATIONS = {"not", "no", "never", "without", "halt"}


def _keyword_score(text: str) -> float:
    """
    Simple domain-aware keyword sentiment for a single headline.
    Returns a raw signed score (not normalised to [-1,1] yet).
    """
    lower = text.lower()
    words = lower.split()
    score = 0.0
    for phrase, weight in _KW_POSITIVE.items():
        if phrase in lower:
            score += weight
    for phrase, weight in _KW_NEGATIVE.items():
        if phrase in lower:
            score -= weight
    # Basic negation: if a negation word precedes a positive keyword, flip
    for i, w in enumerate(words):
        if w in _NEGATIONS and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in _KW_POSITIVE:
                score -= 2 * _KW_POSITIVE[next_word]
    # Normalise loosely: typical headline scores -6 to +6
    return max(-1.0, min(1.0, score / 6.0))


# ── news sources ──────────────────────────────────────────────────────────────
# For each index ETF we also pull news from closely correlated tickers so we
# catch sector-level events (e.g. Apple scandal moves QQQ before QQQ articles).
_RELATED: Dict[str, List[str]] = {
    "SPY": ["^GSPC", "^VIX", "GLD", "TLT"],
    "QQQ": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"],
    "IWM": ["^RUT", "IJR"],
}

# RSS feeds used as fallback when yfinance news is unavailable.
# No API key required; parsed with the stdlib xml module.
_RSS_FEEDS: Dict[str, List[str]] = {
    "SPY": [
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",        # WSJ Markets
        "https://www.cnbc.com/id/100003114/device/rss/rss.html", # CNBC Business
        "https://feeds.reuters.com/reuters/businessNews",        # Reuters Business
    ],
    "QQQ": [
        "https://www.cnbc.com/id/19854910/device/rss/rss.html",  # CNBC Technology
        "https://feeds.a.dj.com/rss/RSSWSJD.xml",               # WSJ Tech
        "https://feeds.reuters.com/news/technology",             # Reuters Tech
    ],
    "IWM": [
        "https://feeds.reuters.com/reuters/businessNews",        # Reuters Business
        "https://www.cnbc.com/id/100003114/device/rss/rss.html", # CNBC Business
    ],
}

_RSS_TIMEOUT = 8   # seconds per feed


def _fetch_rss(url: str, max_items: int = 15) -> List[str]:
    """Fetch and parse an RSS feed, returning titles. Returns [] on any error."""
    try:
        resp = requests.get(url, timeout=_RSS_TIMEOUT,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        titles: List[str] = []
        for item in root.iter("item"):
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                titles.append(title_el.text.strip())
                if len(titles) >= max_items:
                    break
        return titles
    except Exception:
        return []


def _fetch_yfinance(tickers: List[str], max_per: int = 15) -> List[str]:
    """Try yfinance news; return [] silently on any error (including outages)."""
    seen:   set       = set()
    titles: List[str] = []
    for src in tickers:
        try:
            news = yf.Ticker(src).news or []
            for item in news[:max_per]:
                title = (item.get("title") or "").strip()
                if title and title not in seen:
                    seen.add(title)
                    titles.append(title)
        except Exception:
            pass
    return titles


def fetch_headlines(ticker: str, max_per_source: int = 15,
                    verbose: bool = False) -> List[str]:
    """
    Return deduplicated recent news titles for *ticker*.

    Strategy:
      1. Try yfinance (ticker + related symbols).
      2. If yfinance returns nothing (API down / rate-limited), fall back to
         public RSS feeds — no API key required.
    """
    seen:   set       = set()
    titles: List[str] = []

    def _add(new_titles: List[str]) -> None:
        for t in new_titles:
            if t and t not in seen:
                seen.add(t)
                titles.append(t)

    # 1. yfinance
    yf_titles = _fetch_yfinance([ticker] + _RELATED.get(ticker, []), max_per_source)
    _add(yf_titles)

    # 2. RSS fallback
    if not titles:
        if verbose:
            print(f"    yfinance news unavailable for {ticker}, trying RSS feeds ...")
        for feed_url in _RSS_FEEDS.get(ticker, []):
            _add(_fetch_rss(feed_url, max_per_source))
            if len(titles) >= max_per_source:
                break

    return titles


# ── FinBERT scoring ───────────────────────────────────────────────────────────
def score_headlines(titles: List[str], use_finbert: bool = True) -> float:
    """
    Aggregate sentiment over *titles*.

    Primary scorer : FinBERT (ProsusAI/finbert) — requires PyTorch.
    Fallback scorer: domain keyword model — no extra dependencies.

    Returns a score in [-1, +1]:
      +1 = uniformly bullish headlines
      -1 = uniformly bearish / scandal headlines
       0 = neutral or mixed

    Applies exponential recency decay (index 0 = newest headline).
    """
    if not titles:
        return 0.0

    raw: List[float] = []

    if use_finbert and _load_finbert():
        # FinBERT path
        for text in titles[:30]:
            try:
                result = _pipeline(text)[0]
                probs  = {r["label"]: r["score"] for r in result}
                raw.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))
            except Exception:
                raw.append(0.0)
    else:
        # Keyword fallback
        for text in titles[:30]:
            raw.append(_keyword_score(text))

    if not raw:
        return 0.0

    decay = np.exp(-0.25 * np.arange(len(raw)))
    decay /= decay.sum()
    return float(np.dot(raw, decay))


# ── threshold mapping ─────────────────────────────────────────────────────────
_T_PANIC  = -0.75   # extreme negative: flag for immediate exit
_T_BLOCK  = -0.55   # strong negative:  block new entries
_T_REDUCE = -0.25   # mild negative:    linear size reduction


def score_to_modifier(score: float) -> Tuple[float, str, str]:
    """
    Map *score* to (modifier, label, detail).

    modifier : float in [0, 1] to multiply final position weight.
    label    : short uppercase status tag.
    detail   : human-readable explanation including score.
    """
    if score < _T_PANIC:
        return 0.0, "PANIC",  f"score={score:+.3f} - extreme negative: exit & block"
    if score < _T_BLOCK:
        return 0.0, "BLOCK",  f"score={score:+.3f} - strong negative: no new entries"
    if score < _T_REDUCE:
        t   = (score - _T_BLOCK) / (_T_REDUCE - _T_BLOCK)  # 0..1
        mod = round(0.10 + 0.90 * t, 3)
        return mod, "REDUCE", f"score={score:+.3f} - mild negative: size x{mod:.0%}"
    return 1.0, "NORMAL", f"score={score:+.3f} - neutral/positive"


# ── main class ────────────────────────────────────────────────────────────────
class NewsJudgment:
    """
    Thin wrapper with an in-process cache (default TTL = 1 hour) so repeated
    calls within a trading session don't re-run FinBERT or hit the API.

    Parameters
    ----------
    use_finbert : bool
        Set False to skip model scoring and always return modifier=1.0
        (useful for quick sanity checks without loading a 440 MB model).
    cache_ttl   : int
        Seconds before a cached score expires.
    """

    def __init__(self, use_finbert: bool = True, cache_ttl: int = 3600):
        self.use_finbert  = use_finbert
        self._ttl         = cache_ttl
        self._cache: Dict[str, Tuple[float, float]] = {}   # ticker -> (score, ts)

    # ------------------------------------------------------------------
    def get_score(self, ticker: str) -> float:
        now = time.time()
        if ticker in self._cache:
            score, ts = self._cache[ticker]
            if now - ts < self._ttl:
                return score
        titles = fetch_headlines(ticker, verbose=True)
        score  = score_headlines(titles, use_finbert=self.use_finbert)
        self._cache[ticker] = (score, now)
        return score

    def get_modifier(self, ticker: str) -> Tuple[float, str, str]:
        """Return (modifier, label, detail) for the ticker."""
        return score_to_modifier(self.get_score(ticker))

    def is_panic(self, ticker: str) -> bool:
        return self.get_score(ticker) < _T_PANIC

    # ------------------------------------------------------------------
    def report(self, tickers: List[str]) -> None:
        """
        Print a formatted sentiment table to stdout.

        Example output:
          Ticker  Sentiment bar         Score  Action
          SPY     [----------##########]  +0.312  NORMAL score=+0.312
          QQQ     [####------ ----------]  -0.611  BLOCK  score=-0.611
        """
        BAR = 20
        mid = BAR // 2
        print(f"\n  {'Ticker':<7} {'Sentiment (<-- neg | pos -->)':<{BAR+2}} "
              f"{'Score':>7}  Status")
        print("  " + "-" * 65)
        for t in tickers:
            score              = self.get_score(t)
            modifier, lbl, dtl = score_to_modifier(score)
            # Build bar: left half = negative, right half = positive
            if score >= 0:
                filled = min(int(score * mid + 0.5), mid)
                bar = " " * mid + "#" * filled + "-" * (mid - filled)
            else:
                filled = min(int(-score * mid + 0.5), mid)
                bar = "-" * (mid - filled) + "#" * filled + " " * mid
            print(f"  {t:<7} [{bar}] {score:>+.3f}  {lbl:<7} {dtl}")
        print()
