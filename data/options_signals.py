"""
Live options-derived signals from yfinance.

NOTE: yfinance only provides current-snapshot options chains (no historical).
      These signals are for LIVE TRADING only, not backtesting.

Signals computed:
  atm_iv       — at-the-money implied volatility (30d proxy)
  iv_hv_ratio  — ATM IV / 20-day historical vol
                 > 1.3 = fear premium, mean-revert signals stronger
                 < 0.8 = calm market, momentum signals stronger
  pc_ratio_oi  — put/call ratio by open interest
                 > 1.2 = elevated put buying (bearish sentiment = contrarian BUY)
                 < 0.5 = call frenzy (retail FOMO = contrarian SELL)
  max_pain     — strike that minimizes total options value at expiry
                 price attracted toward max_pain as expiry approaches
  max_pain_dist — (current_price - max_pain) / current_price
                  large positive = price above max pain = drag DOWN expected
  gamma_flip   — approximate gamma neutral price (where call/put gamma balance)
                 above gamma flip = positive gamma env (dampened moves)
                 below gamma flip = negative gamma env (amplified moves)
"""
import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_options_signals(ticker_symbol: str, current_price: float) -> dict:
    """
    Fetch and compute options-derived signals for a ticker.
    Returns dict with IV, P/C ratio, max pain, etc.
    Returns empty dict on any error.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {}

    try:
        t = yf.Ticker(ticker_symbol)
        exps = t.options
        if not exps:
            return {}

        # Find expiry 20-40 days out for ATM IV estimate
        today = pd.Timestamp.now().date()
        target_days = 30
        best_exp = None
        best_diff = 999
        for exp in exps:
            days = (pd.Timestamp(exp).date() - today).days
            if abs(days - target_days) < best_diff and days > 5:
                best_diff = abs(days - target_days)
                best_exp  = exp

        if best_exp is None:
            best_exp = exps[min(2, len(exps) - 1)]  # fallback to 3rd expiry

        chain = t.option_chain(best_exp)
        calls = chain.calls
        puts  = chain.puts

        if calls.empty or puts.empty:
            return {}

        # ATM implied volatility
        atm_call = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]
        atm_put  = puts.iloc[(puts["strike"]  - current_price).abs().argsort()[:1]]

        atm_iv_call = float(atm_call["impliedVolatility"].iloc[0]) if not atm_call.empty else np.nan
        atm_iv_put  = float(atm_put["impliedVolatility"].iloc[0])  if not atm_put.empty  else np.nan
        atm_iv = np.nanmean([atm_iv_call, atm_iv_put])

        # Historical volatility (20-day) for IV/HV ratio
        hist = t.history(period="2mo", interval="1d")
        hv_20 = float(hist["Close"].pct_change().rolling(20).std().iloc[-1]) * np.sqrt(252) if len(hist) >= 21 else np.nan
        iv_hv_ratio = atm_iv / hv_20 if hv_20 and hv_20 > 0 else np.nan

        # Put/call ratio (OI-weighted)
        call_oi = calls["openInterest"].fillna(0).sum()
        put_oi  = puts["openInterest"].fillna(0).sum()
        pc_ratio = put_oi / call_oi if call_oi > 0 else np.nan

        # Volume-weighted P/C ratio
        call_vol = calls["volume"].fillna(0).sum()
        put_vol  = puts["volume"].fillna(0).sum()
        pc_ratio_vol = put_vol / call_vol if call_vol > 0 else np.nan

        # Max pain: strike where total options value is minimized at expiry
        all_calls = calls[["strike", "openInterest"]].rename(columns={"openInterest": "call_oi"})
        all_puts  = puts[["strike", "openInterest"]].rename(columns={"openInterest": "put_oi"})
        pain_df   = pd.merge(all_calls, all_puts, on="strike", how="outer").fillna(0)

        def total_pain(expiry_price: float) -> float:
            call_loss = np.maximum(0, pain_df["strike"] - expiry_price) * pain_df["call_oi"]
            put_loss  = np.maximum(0, expiry_price - pain_df["strike"]) * pain_df["put_oi"]
            return float(call_loss.sum() + put_loss.sum())

        strikes = sorted(pain_df["strike"].unique())
        pain_vals = [total_pain(s) for s in strikes]
        max_pain_strike = float(strikes[int(np.argmin(pain_vals))])
        max_pain_dist = (current_price - max_pain_strike) / current_price

        # Gamma exposure approximation: net call gamma - put gamma by strike
        # Positive gamma zone = market maker hedging dampens moves (above gamma flip)
        # Negative gamma zone = market maker hedging amplifies moves (below gamma flip)
        # Simple proxy: compare call vs put OI at strikes near current price
        atm_range = current_price * 0.03  # +/- 3%
        atm_calls = calls[(calls["strike"] >= current_price - atm_range) &
                          (calls["strike"] <= current_price + atm_range)]
        atm_puts  = puts[(puts["strike"]  >= current_price - atm_range) &
                         (puts["strike"]  <= current_price + atm_range)]
        net_gamma_proxy = float(atm_calls["openInterest"].fillna(0).sum() -
                                atm_puts["openInterest"].fillna(0).sum())
        gamma_regime = "positive" if net_gamma_proxy > 0 else "negative"

        return {
            "atm_iv":       round(atm_iv, 4)       if not np.isnan(atm_iv) else None,
            "hv_20":        round(hv_20, 4)         if not np.isnan(hv_20) else None,
            "iv_hv_ratio":  round(iv_hv_ratio, 3)  if not np.isnan(iv_hv_ratio) else None,
            "pc_ratio_oi":  round(pc_ratio, 3)      if not np.isnan(pc_ratio) else None,
            "pc_ratio_vol": round(pc_ratio_vol, 3)  if not np.isnan(pc_ratio_vol) else None,
            "max_pain":     max_pain_strike,
            "max_pain_dist": round(max_pain_dist, 4),
            "gamma_regime": gamma_regime,
            "net_gamma_proxy": int(net_gamma_proxy),
            "expiry_used":  best_exp,
        }

    except Exception as e:
        logger.debug(f"[options] {ticker_symbol}: {e}")
        return {}


def options_position_multiplier(signals: dict) -> float:
    """
    Returns a position size multiplier (0.5 to 1.5) based on options signals.

    Logic:
      IV/HV ratio:
        > 1.4 = market fearful = good for contrarian signals = 1.2x size
        0.8-1.4 = neutral = 1.0x
        < 0.8 = complacent = reduce exposure = 0.9x

      P/C ratio:
        > 1.2 = put-heavy = contrarian bullish for long signals = 1.1x
        < 0.5 = call-heavy = reduce long exposure = 0.9x

      Max pain:
        > 2% above max pain = price being dragged DOWN = -0.1 for longs
        > 2% below max pain = price being dragged UP = +0.1 for longs
    """
    if not signals:
        return 1.0

    mult = 1.0

    iv_hv = signals.get("iv_hv_ratio")
    if iv_hv:
        if iv_hv > 1.4:
            mult *= 1.2    # high fear premium = good for signals
        elif iv_hv < 0.8:
            mult *= 0.85   # complacency = reduce size

    pc = signals.get("pc_ratio_oi")
    if pc:
        if pc > 1.2:
            mult *= 1.1    # put-heavy = contrarian BUY environment
        elif pc < 0.5:
            mult *= 0.9    # call-heavy = reduce longs

    max_pain_dist = signals.get("max_pain_dist", 0.0)
    if abs(max_pain_dist) > 0.02:
        # Price pulled toward max pain = reduce position in opposite direction
        if max_pain_dist > 0.02:   # price above max pain = gravity DOWN
            mult *= 0.9
        elif max_pain_dist < -0.02: # price below max pain = gravity UP
            mult *= 1.05

    return float(np.clip(mult, 0.5, 1.5))
