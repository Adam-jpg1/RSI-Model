import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from config import CONFIG
from data.fetcher import get_price_data, get_daily_data
from data.indicators import add_all_indicators, add_daily_trend_features
from data.universe import get_universe, get_trained_tickers
from ml.features import get_feature_matrix
from ml.model import RSIModel

logger = logging.getLogger(__name__)


def _quick_signal(ticker: str, model: RSIModel) -> Optional[Dict]:
    """Generate a live signal for one ticker. Returns None on any error."""
    try:
        df = get_price_data(ticker, period=CONFIG.screening.scan_period,
                            interval=CONFIG.screening.scan_interval)
        if len(df) < 30:
            return None
        df = add_all_indicators(df)
        try:
            daily_df = get_daily_data(ticker, period="1y")
            df = add_daily_trend_features(df, daily_df)
        except Exception:
            pass
        features = get_feature_matrix(df, 0.0)
        last_row = features.iloc[-1]
        if last_row.isna().any():
            return None

        pred = model.predict_single(last_row)
        adx = float(df["ADX"].iloc[-1]) if "ADX" in df.columns else 0.0
        rsi = float(df["RSI_14"].iloc[-1]) if "RSI_14" in df.columns else 50.0
        vol_ratio = float(df["volume_ratio"].iloc[-1]) if "volume_ratio" in df.columns else 1.0
        macd_hist = float(df["MACD_hist"].iloc[-1]) if "MACD_hist" in df.columns else 0.0
        d_trend = float(df["d_trend_score"].iloc[-1]) if "d_trend_score" in df.columns else 0.0

        return {
            "ticker": ticker,
            "signal": pred["signal"],
            "confidence": pred["confidence"],
            "probs": pred["probs"],
            "adx": adx,
            "rsi": rsi,
            "vol_ratio": vol_ratio,
            "macd_hist": macd_hist,
            "d_trend": d_trend,
            "price": float(df["Close"].iloc[-1]),
        }
    except Exception:
        logger.debug(f"[scanner] {ticker} failed: {traceback.format_exc()}")
        return None


def scan_universe(
    tickers: List[str] = None,
    max_workers: int = None,
    top_n: int = None,
) -> List[Dict]:
    """
    Scan all trained tickers (or a provided list) and return opportunities ranked
    by ADX quality and model confidence. Only tickers with saved models are scored.
    """
    top_n = top_n or CONFIG.screening.top_n_opportunities
    max_workers = max_workers or CONFIG.screening.max_scan_workers

    if tickers is None:
        universe = get_universe()
        trained = set(get_trained_tickers())
        tickers = [t for t in universe if t in trained]
        if not tickers:
            # Fall back to all trained tickers even if not in universe cache
            tickers = get_trained_tickers()

    print(f"[scanner] Scanning {len(tickers)} tickers...")
    start = time.time()

    # Load all models once (avoids repeated disk I/O inside threads)
    models: Dict[str, RSIModel] = {}
    for ticker in tickers:
        m = RSIModel()
        try:
            m.load(ticker=ticker)
            models[ticker] = m
        except Exception:
            pass

    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_quick_signal, t, m): t for t, m in models.items()}
        for future in as_completed(futures):
            try:
                sig = future.result(timeout=45)
                if sig:
                    results.append(sig)
            except Exception:
                pass

    elapsed = time.time() - start
    print(f"[scanner] {len(results)}/{len(tickers)} scored in {elapsed:.1f}s")

    return _rank_opportunities(results)[:top_n]


def _rank_opportunities(results: List[Dict]) -> List[Dict]:
    """
    Rank signals: directional (UP/DOWN) with strong ADX first, then by confidence.
    """
    def _key(r: Dict):
        is_directional = r["signal"] in ("UP", "DOWN")
        adx_ok = r["adx"] >= CONFIG.screening.min_adx
        conf = r["confidence"]
        # sort ascending: (not directional, not adx_ok) puts best entries first
        return (not (is_directional and adx_ok), -conf)

    return sorted(results, key=_key)


def get_untrained_tickers(universe: List[str] = None) -> List[str]:
    """Return tickers in the universe that do not yet have a trained model."""
    universe = universe or get_universe()
    trained = set(get_trained_tickers())
    return [t for t in universe if t not in trained]
