import pandas as pd
import numpy as np
from typing import Dict, Optional

from config import CONFIG
from data.fetcher import get_price_data, get_daily_data, get_stock_news
from data.indicators import add_all_indicators, add_daily_trend_features
from sentiment.analyzer import analyze_headlines
from ml.features import get_feature_matrix
from ml.model import RSIModel


def kelly_fraction(win_prob: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 0.0
    odds = avg_win / avg_loss
    f = (odds * win_prob - (1 - win_prob)) / odds
    return float(np.clip(f * CONFIG.bot.kelly_fraction, 0.0, CONFIG.bot.max_position_pct))


def backtest(
    ticker: str,
    model: RSIModel,
    period: str = "2y",
    interval: str = "1h",
    initial_capital: float = 100_000.0,
    sentiment_score: float = 0.0,
    skip_pct: float = 0.85,
    allow_short: bool = False,
) -> Dict:
    df = get_price_data(ticker, period=period, interval=interval)
    df = add_all_indicators(df)
    try:
        daily_df = get_daily_data(ticker, period=period)
        df = add_daily_trend_features(df, daily_df)
    except Exception:
        pass
    features = get_feature_matrix(df, sentiment_score)

    feature_rows = features.dropna()
    skip_n = int(len(feature_rows) * skip_pct)
    feature_rows = feature_rows.iloc[skip_n:]

    return _run_simulation(df, feature_rows, model, initial_capital, ticker,
                           allow_short=allow_short)


def _run_simulation(
    df: pd.DataFrame,
    feature_rows: pd.DataFrame,
    model: RSIModel,
    initial_capital: float = 100_000.0,
    ticker: str = "TICKER",
    adx_filter_min: float = 0.0,
    allow_short: bool = False,
    reentry_after_tp: bool = False,
    daily_signals: Optional[dict] = None,
    d_trend_gate: bool = False,
    d_trend_min_score: int = 2,
    ignore_flat_exit: bool = False,
    trailing_stop_pct: float = 0.0,
    min_conviction_spread: float = 0.0,
    session_hours: Optional[frozenset] = None,
    max_hold_bars: int = 0,
    spy_signals: Optional[dict] = None,
    atr_tp_mult: float = 0.0,
    atr_sl_mult: float = 0.0,
    min_vol_ratio: float = 0.0,
    vwap_align: bool = False,
    min_signal_bars: int = 1,
    regime_gate: bool = False,
    breakeven_pct: float = 0.0,
    partial_tp_pct: float = 0.0,
    partial_tp_ratio: float = 0.5,
    vol_target_atr: float = 0.0,
    daily_min_conf: float = 0.0,
) -> Dict:
    capital = initial_capital
    # Long position state
    long_qty: float = 0.0
    long_entry: float = 0.0
    long_peak: float = 0.0
    long_bars: int = 0
    long_tp: float = 0.0   # per-trade take-profit level (fraction)
    long_sl: float = 0.0   # per-trade stop-loss level (fraction)
    # Short position state
    short_qty: float = 0.0
    short_entry: float = 0.0
    short_trough: float = 0.0
    short_bars: int = 0
    short_tp: float = 0.0
    short_sl: float = 0.0
    # Break-even lock state
    long_be_locked: bool = False
    short_be_locked: bool = False
    # Partial take-profit state
    long_partial_taken: bool = False
    short_partial_taken: bool = False

    trades = []
    equity_curve = []
    # Signal persistence tracking
    _prev_signal: str = "FLAT"
    _signal_run: int = 0

    stop = CONFIG.bot.stop_loss_pct
    take = CONFIG.bot.take_profit_pct

    for idx, row in feature_rows.iterrows():
        if idx not in df.index:
            continue
        close = float(df.loc[idx, "Close"])
        adx_val = float(df.loc[idx, "ADX"]) if "ADX" in df.columns and idx in df.index else 0.0

        # Mark-to-market equity
        long_mv = long_qty * close
        short_mv = short_qty * (2 * short_entry - close) if short_qty > 0 else 0.0
        equity_curve.append({"time": idx, "equity": capital + long_mv + short_mv})

        prediction = model.predict_single(row)
        signal = prediction["signal"]
        confidence = prediction["confidence"]
        probs = prediction.get("probs", {})

        # Conviction spread: require top class to lead second-place by at least min_conviction_spread
        low_conviction = False
        if min_conviction_spread > 0 and probs:
            sorted_p = sorted(probs.values(), reverse=True)
            if len(sorted_p) >= 2 and (sorted_p[0] - sorted_p[1]) < min_conviction_spread:
                low_conviction = True

        if confidence < CONFIG.bot.min_confidence or low_conviction:
            # Still update trailing peaks even when confidence is too low to act
            if long_qty > 0 and close > long_peak:
                long_peak = close
            if short_qty > 0 and close < short_trough:
                short_trough = close
            _prev_signal = "FLAT"
            _signal_run = 0
            continue

        # Signal persistence: track how many consecutive bars show the same signal
        if signal == _prev_signal and signal != "FLAT":
            _signal_run += 1
        else:
            _signal_run = 1
        _prev_signal = signal
        signal_persistent = _signal_run >= min_signal_bars

        # --- Manage open long position ---
        if long_qty > 0:
            long_bars += 1
            if close > long_peak:
                long_peak = close
            pnl_pct = (close - long_entry) / long_entry

            cur_stop = long_sl if long_sl > 0 else stop
            cur_take = long_tp if long_tp > 0 else take

            # Trailing stop: once position gains >= trailing_stop_pct, trail at peak*(1-trail)
            effective_stop = cur_stop
            if trailing_stop_pct > 0 and long_peak > 0:
                trail_floor = (long_peak - long_entry) / long_entry
                if trail_floor >= trailing_stop_pct:
                    trail_stop_price = long_peak * (1 - trailing_stop_pct)
                    # Only tighten stop (never loosen below hard stop)
                    if trail_stop_price > long_entry * (1 - cur_stop):
                        effective_stop = (long_entry - trail_stop_price) / long_entry

            # Break-even lock: once up >= breakeven_pct, never let trade lose from entry
            if breakeven_pct > 0:
                if not long_be_locked and pnl_pct >= breakeven_pct:
                    long_be_locked = True
                if long_be_locked:
                    effective_stop = min(effective_stop, 0.0)

            # Partial take-profit: sell partial_tp_ratio of position at partial_tp_pct gain
            if partial_tp_pct > 0 and not long_partial_taken and pnl_pct >= partial_tp_pct:
                exit_shares = long_qty * partial_tp_ratio
                pnl_partial = exit_shares * (close - long_entry)
                capital += exit_shares * close
                long_qty -= exit_shares
                long_partial_taken = True
                trades.append({
                    "type": "SELL", "side": "long",
                    "price": close, "shares": round(exit_shares, 4),
                    "pnl": round(pnl_partial, 2), "pnl_pct": round(pnl_pct * 100, 2),
                    "exit_reason": "partial_tp", "time": idx,
                })

            signal_exit = signal == "DOWN" if ignore_flat_exit else signal in ("DOWN", "FLAT")
            timed_out = max_hold_bars > 0 and long_bars >= max_hold_bars
            should_close = (
                pnl_pct <= -effective_stop
                or pnl_pct >= cur_take
                or signal_exit
                or timed_out
            )
            if should_close:
                pnl = long_qty * (close - long_entry)
                capital += long_qty * close
                if pnl_pct <= -effective_stop:
                    exit_reason = "stop"
                elif pnl_pct >= cur_take:
                    exit_reason = "take"
                elif timed_out:
                    exit_reason = "timeout"
                else:
                    exit_reason = "signal"
                trades.append({
                    "type": "SELL", "side": "long",
                    "price": close, "shares": long_qty,
                    "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct * 100, 2),
                    "exit_reason": exit_reason,
                    "time": idx,
                })
                long_qty = 0.0
                long_entry = 0.0
                long_peak = 0.0
                long_bars = 0
                long_tp = 0.0
                long_sl = 0.0
                long_be_locked = False
                long_partial_taken = False
                # Immediately re-enter on the same bar if TP hit and signal still UP
                if reentry_after_tp and exit_reason == "take" and signal == "UP" and adx_ok and daily_ok:
                    size_pct = CONFIG.bot.max_position_pct
                    alloc = capital * size_pct
                    shares = alloc / close
                    if shares * close <= capital:
                        long_qty = shares
                        long_entry = close
                        long_peak = close
                        capital -= shares * close
                        trades.append({
                            "type": "BUY", "side": "long", "reentry": True,
                            "price": close, "shares": round(shares, 4),
                            "confidence": round(confidence, 3), "time": idx,
                        })

        # --- Manage open short position ---
        if short_qty > 0:
            short_bars += 1
            if close < short_trough or short_trough == 0.0:
                short_trough = close
            pnl_pct = (short_entry - close) / short_entry  # profit when price falls

            cur_stop_s = short_sl if short_sl > 0 else stop
            cur_take_s = short_tp if short_tp > 0 else take

            effective_stop = cur_stop_s
            if trailing_stop_pct > 0 and short_trough > 0:
                trail_floor = (short_entry - short_trough) / short_entry
                if trail_floor >= trailing_stop_pct:
                    trail_stop_price = short_trough * (1 + trailing_stop_pct)
                    if trail_stop_price < short_entry * (1 + cur_stop_s):
                        effective_stop = (trail_stop_price - short_entry) / short_entry

            if breakeven_pct > 0:
                if not short_be_locked and pnl_pct >= breakeven_pct:
                    short_be_locked = True
                if short_be_locked:
                    effective_stop = min(effective_stop, 0.0)

            if partial_tp_pct > 0 and not short_partial_taken and pnl_pct >= partial_tp_pct:
                exit_shares = short_qty * partial_tp_ratio
                pnl_partial = exit_shares * (short_entry - close)
                capital += pnl_partial + exit_shares * short_entry
                short_qty -= exit_shares
                short_partial_taken = True
                trades.append({
                    "type": "COVER", "side": "short",
                    "price": close, "shares": round(exit_shares, 4),
                    "pnl": round(pnl_partial, 2), "pnl_pct": round(pnl_pct * 100, 2),
                    "exit_reason": "partial_tp", "time": idx,
                })

            signal_exit = signal == "UP" if ignore_flat_exit else signal in ("UP", "FLAT")
            timed_out = max_hold_bars > 0 and short_bars >= max_hold_bars
            should_close = (
                pnl_pct <= -effective_stop
                or pnl_pct >= cur_take_s
                or signal_exit
                or timed_out
            )
            if should_close:
                pnl = short_qty * (short_entry - close)
                capital += pnl + short_qty * short_entry  # return collateral + P&L
                if pnl_pct <= -effective_stop:
                    exit_reason = "stop"
                elif pnl_pct >= cur_take_s:
                    exit_reason = "take"
                elif timed_out:
                    exit_reason = "timeout"
                else:
                    exit_reason = "signal"
                trades.append({
                    "type": "COVER", "side": "short",
                    "price": close, "shares": short_qty,
                    "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct * 100, 2),
                    "exit_reason": exit_reason,
                    "time": idx,
                })
                short_qty = 0.0
                short_entry = 0.0
                short_trough = 0.0
                short_bars = 0
                short_tp = 0.0
                short_sl = 0.0
                short_be_locked = False
                short_partial_taken = False

        # ADX regime gate for new entries
        adx_ok = adx_filter_min <= 0 or adx_val >= adx_filter_min

        # Daily direction gate — only enter when daily signals agree
        daily_ok = True
        if daily_signals is not None:
            date_key = str(idx)[:10]
            daily_sig = daily_signals.get(date_key)
            if daily_sig is not None:
                if isinstance(daily_sig, dict):
                    d_sig = daily_sig.get("signal", "FLAT")
                    d_conf = daily_sig.get("conf", 0.0)
                    if d_sig != "FLAT":
                        daily_ok = (signal == d_sig) and (d_conf >= daily_min_conf)
                elif daily_sig != "FLAT":
                    daily_ok = (signal == daily_sig)

        if d_trend_gate and "d_trend_score" in row.index:
            # Experiment B1: rule-based daily trend score gate (same as live bot)
            d_trend = float(row["d_trend_score"]) if not pd.isna(row["d_trend_score"]) else 0.0
            if signal == "UP" and d_trend < -d_trend_min_score:
                daily_ok = False
            elif signal == "DOWN" and d_trend > d_trend_min_score:
                daily_ok = False

        # Time-of-day gate: only open new entries in allowed hours (bar open hour in ET)
        session_ok = True
        if session_hours is not None:
            try:
                bar_hour = idx.hour
            except AttributeError:
                bar_hour = pd.Timestamp(idx).hour
            session_ok = bar_hour in session_hours

        # SPY alignment gate: skip entries that fight the broader market trend
        spy_ok = True
        if spy_signals is not None:
            ts_str = str(idx)
            spy_sig = spy_signals.get(ts_str, "FLAT")
            if spy_sig != "FLAT":
                spy_ok = (signal == spy_sig)

        # Volume confirmation gate: require elevated volume at entry
        vol_ok = True
        if min_vol_ratio > 0:
            vol_val = 1.0
            if "volume_ratio" in row.index:
                v = row["volume_ratio"]
                vol_val = float(v) if not pd.isna(v) else 1.0
            vol_ok = vol_val >= min_vol_ratio

        # VWAP alignment gate: only enter long above VWAP, short below VWAP
        vwap_ok = True
        if vwap_align:
            vwap_val = 0.0
            if "price_vs_vwap" in row.index:
                v = row["price_vs_vwap"]
                vwap_val = float(v) if not pd.isna(v) else 0.0
            if signal == "UP":
                vwap_ok = vwap_val >= 0  # price above VWAP
            elif signal == "DOWN":
                vwap_ok = vwap_val <= 0  # price below VWAP

        # Market regime gate: block entries when VIX is in high-fear zone (d_vix_regime >= 2.0)
        # or when the daily trend is severely broken (d_trend_align < 1 of 4 EMAs pointing up)
        regime_ok = True
        if regime_gate:
            if "d_vix_regime" in row.index:
                vix_reg = row["d_vix_regime"]
                if not pd.isna(vix_reg) and float(vix_reg) >= 2.0:
                    regime_ok = False
            if regime_ok and signal == "UP" and "d_trend_align" in row.index:
                ta = row["d_trend_align"]
                if not pd.isna(ta) and float(ta) < 1.0:
                    regime_ok = False

        # Compute ATR-scaled TP/SL for this bar (used when atr_tp_mult > 0)
        atr_pct_val = 0.0
        if atr_tp_mult > 0 or atr_sl_mult > 0:
            if "ATR_pct" in row.index:
                v = row["ATR_pct"]
                atr_pct_val = float(v) if not pd.isna(v) else 0.0
            if atr_pct_val <= 0 and "ATR_pct" in df.columns and idx in df.index:
                v = df.loc[idx, "ATR_pct"]
                atr_pct_val = float(v) if not pd.isna(v) else 0.0

        # --- Open long ---
        if long_qty == 0 and short_qty == 0 and signal == "UP" and adx_ok and daily_ok and session_ok and spy_ok and vol_ok and vwap_ok and regime_ok and signal_persistent:
            size_pct = CONFIG.bot.max_position_pct
            # Volatility targeting: scale position inversely with current ATR to equalize dollar risk
            if vol_target_atr > 0 and atr_pct_val > 0:
                vol_scale = min(vol_target_atr / atr_pct_val, 2.0)
                size_pct = min(size_pct * vol_scale, CONFIG.bot.max_position_pct)
            alloc = capital * size_pct
            shares = alloc / close
            if shares * close <= capital:
                long_qty = shares
                long_entry = close
                capital -= shares * close
                # Set per-trade ATR-based levels (fall back to global config if ATR unavailable)
                if atr_tp_mult > 0 and atr_pct_val > 0:
                    long_tp = max(atr_pct_val * atr_tp_mult, 0.005)
                    long_sl = max(atr_pct_val * atr_sl_mult, 0.003) if atr_sl_mult > 0 else stop
                else:
                    long_tp = 0.0
                    long_sl = 0.0
                trades.append({
                    "type": "BUY", "side": "long",
                    "price": close, "shares": round(shares, 4),
                    "confidence": round(confidence, 3), "time": idx,
                })

        # --- Open short ---
        elif allow_short and long_qty == 0 and short_qty == 0 and signal == "DOWN" and adx_ok and daily_ok and session_ok and spy_ok and vol_ok and vwap_ok and regime_ok and signal_persistent:
            size_pct = CONFIG.bot.max_position_pct * CONFIG.bot.short_size_factor
            if vol_target_atr > 0 and atr_pct_val > 0:
                vol_scale = min(vol_target_atr / atr_pct_val, 2.0)
                size_pct = min(size_pct * vol_scale, CONFIG.bot.max_position_pct)
            alloc = capital * size_pct
            shares = alloc / close
            if shares > 0:
                short_qty = shares
                short_entry = close
                # Reserve collateral equal to position notional
                capital -= shares * close
                if atr_tp_mult > 0 and atr_pct_val > 0:
                    short_tp = max(atr_pct_val * atr_tp_mult, 0.005)
                    short_sl = max(atr_pct_val * atr_sl_mult, 0.003) if atr_sl_mult > 0 else stop
                else:
                    short_tp = 0.0
                    short_sl = 0.0
                trades.append({
                    "type": "SHORT", "side": "short",
                    "price": close, "shares": round(shares, 4),
                    "confidence": round(confidence, 3), "time": idx,
                })

    # Force-close any open positions at last bar
    if feature_rows is not None and len(feature_rows) > 0:
        last_idx = feature_rows.index[-1]
        if last_idx in df.index:
            last_close = float(df.loc[last_idx, "Close"])
            if long_qty > 0:
                pnl = long_qty * (last_close - long_entry)
                pnl_pct = (last_close - long_entry) / long_entry
                capital += long_qty * last_close
                trades.append({
                    "type": "SELL", "side": "long",
                    "price": last_close, "shares": long_qty,
                    "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct * 100, 2),
                    "exit_reason": "eod", "time": last_idx,
                })
            if short_qty > 0:
                pnl = short_qty * (short_entry - last_close)
                pnl_pct = (short_entry - last_close) / short_entry
                capital += pnl + short_qty * short_entry
                trades.append({
                    "type": "COVER", "side": "short",
                    "price": last_close, "shares": short_qty,
                    "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct * 100, 2),
                    "exit_reason": "eod", "time": last_idx,
                })

    equity_df = pd.DataFrame(equity_curve).set_index("time") if equity_curve else pd.DataFrame(columns=["equity"])
    final_equity = float(equity_df["equity"].iloc[-1]) if len(equity_df) else initial_capital
    total_return = (final_equity / initial_capital - 1.0) * 100

    trades_df = pd.DataFrame(trades)
    closed = trades_df[trades_df["type"].isin(["SELL", "COVER"])] if len(trades_df) else pd.DataFrame()
    metrics = _compute_metrics(equity_df["equity"] if len(equity_df) else pd.Series([initial_capital]),
                               initial_capital, closed, total_return)

    return {
        "ticker": ticker,
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return, 2),
        "metrics": metrics,
        "trades": trades_df,
        "equity_curve": equity_df,
    }


def _infer_periods_per_year(equity: pd.Series) -> int:
    if len(equity) < 2:
        return 252 * 78
    delta = pd.Series(equity.index).diff().dropna()
    if len(delta) == 0:
        return 252 * 78
    median_secs = delta.median().total_seconds() if hasattr(delta.iloc[0], "total_seconds") else 300
    bars_per_day = (6.5 * 3600) / max(median_secs, 60)
    return int(252 * bars_per_day)


def _compute_metrics(equity: pd.Series, initial: float, closed_trades: pd.DataFrame, total_return: float) -> Dict:
    returns = equity.pct_change().dropna()
    periods_per_year = _infer_periods_per_year(equity)
    sharpe = 0.0
    if returns.std() > 0:
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = float(drawdown.min()) * 100

    win_rate, avg_win, avg_loss = 0.0, 0.0, 0.0
    if len(closed_trades):
        wins = closed_trades[closed_trades["pnl"] > 0]
        losses = closed_trades[closed_trades["pnl"] <= 0]
        win_rate = len(wins) / len(closed_trades) * 100
        avg_win = float(wins["pnl_pct"].mean()) if len(wins) else 0.0
        avg_loss = float(losses["pnl_pct"].mean()) if len(losses) else 0.0

    profit_factor = 0.0
    if len(closed_trades) and avg_loss != 0:
        gross_profit = closed_trades[closed_trades["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(closed_trades[closed_trades["pnl"] <= 0]["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": len(closed_trades),
        "win_rate_pct": round(win_rate, 2),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "profit_factor": round(profit_factor, 3),
    }


def print_backtest_report(result: Dict):
    m = result["metrics"]
    print(f"\n{'='*50}")
    print(f"  BACKTEST: {result['ticker']}")
    print(f"{'='*50}")
    print(f"  Total return  : {result['total_return_pct']:+.2f}%")
    print(f"  Final equity  : ${result['final_equity']:,.2f}")
    print(f"  Sharpe ratio  : {m['sharpe_ratio']:.3f}")
    print(f"  Max drawdown  : {m['max_drawdown_pct']:.2f}%")
    print(f"  Total trades  : {m['total_trades']}")
    print(f"  Win rate      : {m['win_rate_pct']:.1f}%")
    print(f"  Avg win       : {m['avg_win_pct']:+.3f}%")
    print(f"  Avg loss      : {m['avg_loss_pct']:+.3f}%")
    print(f"  Profit factor : {m['profit_factor']:.3f}")
    print(f"{'='*50}")
