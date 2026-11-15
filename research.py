"""
Systematic research framework for improving model performance.
Runs controlled experiments against a baseline with proper validation.

Usage:
  python research.py baseline               # establish baseline across all tickers
  python research.py confidence             # Experiment A: confidence thresholds
  python research.py all                    # run everything

All results saved to research_results.json for comparison.
"""
import sys
import os
import json
import math
import argparse
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime

# Force unbuffered output so progress is visible in log files immediately.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

from config import CONFIG
from ml.model import RSIModel
from ml.optimizer import load_best_params
from data.universe import get_trained_tickers
from backtest.simulate import walk_forward_backtest, bootstrap_confidence
from data.fetcher import get_price_data

RESULTS_FILE = "research_results.json"
N_FOLDS = 4
BARS_PER_YEAR = 252 * 6.5  # ~1638 hourly bars per year


# ------------------------------------------------------------------ #
# Metric utilities                                                    #
# ------------------------------------------------------------------ #

def annualize(total_return_pct: float, test_bars: int) -> float:
    """Convert total walk-forward return to annualized CAGR."""
    years = test_bars / BARS_PER_YEAR
    if years <= 0:
        return 0.0
    r = total_return_pct / 100.0
    # Annualize as CAGR
    cagr = (1 + r) ** (1 / years) - 1
    return round(cagr * 100, 2)


def buy_and_hold_return(ticker: str, bars: int) -> float:
    """Return B&H return over the most recent `bars` hourly bars."""
    try:
        df = get_price_data(ticker, period="2y", interval="1h")
        if len(df) < bars:
            df = df
        else:
            df = df.iloc[-bars:]
        if len(df) < 2:
            return 0.0
        ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0) * 100
        return round(ret, 2)
    except Exception:
        return 0.0


def run_wf(ticker: str, conf_threshold: float = None, adx_min: float = None,
           daily_confirm: bool = False, d_trend_gate: bool = False,
           d_trend_min_score: int = 2, ignore_flat_exit: bool = False,
           trailing_stop_pct: float = 0.0, allow_short: bool = None,
           stop_loss: float = None, take_profit: float = None,
           conf_threshold_param: float = None,
           min_conviction_spread: float = 0.0,
           session_hours: frozenset = None,
           max_hold_bars: int = 0,
           recency_halflife: int = 0,
           spy_align: bool = False,
           atr_tp_mult: float = 0.0,
           atr_sl_mult: float = 0.0,
           min_vol_ratio: float = 0.0,
           vwap_align: bool = False,
           min_signal_bars: int = 1,
           label_pct_low: float = None,
           label_pct_high: float = None,
           target_bars_override: int = None,
           max_position_pct: float = None,
           use_spy_context: bool = False,
           use_earnings_features: bool = False,
           reentry_after_tp: bool = False,
           regime_gate: bool = False,
           breakeven_pct: float = 0.0,
           partial_tp_pct: float = 0.0,
           partial_tp_ratio: float = 0.5,
           vol_target_atr: float = 0.0,
           daily_min_conf: float = 0.0,
           use_sector_features: bool = False,
           daily_n_days_ahead: int = 5,
           n_ensemble: int = 1) -> dict:
    # conf_threshold_param is an alias used by combo dicts (avoid shadowing positional arg)
    if conf_threshold_param is not None:
        conf_threshold = conf_threshold_param
    """Run walk-forward for a single ticker with optional param overrides."""
    orig_conf = CONFIG.bot.min_confidence
    orig_stop = CONFIG.bot.stop_loss_pct
    orig_take = CONFIG.bot.take_profit_pct
    orig_lbl_lo = CONFIG.ml.label_pct_low
    orig_lbl_hi = CONFIG.ml.label_pct_high
    orig_tbars = CONFIG.ml.target_bars_by_interval.get("1h", 6)
    orig_pos_pct = CONFIG.bot.max_position_pct

    try:
        if conf_threshold is not None:
            CONFIG.bot.min_confidence = conf_threshold
        if stop_loss is not None:
            CONFIG.bot.stop_loss_pct = stop_loss
        if take_profit is not None:
            CONFIG.bot.take_profit_pct = take_profit
        if label_pct_low is not None:
            CONFIG.ml.label_pct_low = label_pct_low
        if label_pct_high is not None:
            CONFIG.ml.label_pct_high = label_pct_high
        if target_bars_override is not None:
            CONFIG.ml.target_bars_by_interval["1h"] = target_bars_override
        if max_position_pct is not None:
            CONFIG.bot.max_position_pct = max_position_pct

        result = walk_forward_backtest(
            ticker, n_folds=N_FOLDS,
            adx_filter_min=adx_min,
            daily_confirm=daily_confirm,
            d_trend_gate=d_trend_gate,
            d_trend_min_score=d_trend_min_score,
            ignore_flat_exit=ignore_flat_exit,
            trailing_stop_pct=trailing_stop_pct,
            allow_short=allow_short,
            min_conviction_spread=min_conviction_spread,
            session_hours=session_hours,
            max_hold_bars=max_hold_bars,
            recency_halflife=recency_halflife,
            spy_align=spy_align,
            atr_tp_mult=atr_tp_mult,
            atr_sl_mult=atr_sl_mult,
            min_vol_ratio=min_vol_ratio,
            vwap_align=vwap_align,
            min_signal_bars=min_signal_bars,
            use_spy_context=use_spy_context,
            use_earnings_features=use_earnings_features,
            reentry_after_tp=reentry_after_tp,
            regime_gate=regime_gate,
            breakeven_pct=breakeven_pct,
            partial_tp_pct=partial_tp_pct,
            partial_tp_ratio=partial_tp_ratio,
            vol_target_atr=vol_target_atr,
            daily_min_conf=daily_min_conf,
            use_sector_features=use_sector_features,
            daily_n_days_ahead=daily_n_days_ahead,
            n_ensemble=n_ensemble,
            min_confidence=conf_threshold,
        )

        # Count total test bars across all folds
        test_bars = sum(len(r["trades"]) for r in result["fold_results"]) if result["fold_results"] else 0
        # Better: estimate from fold size
        fold_bars = 682  # typical for 2y/4folds
        test_bars_est = N_FOLDS * fold_bars

        cagr = annualize(result["total_return_pct"], test_bars_est)

        # Compute compound return (not sum)
        compound = 1.0
        for r in result["fold_results"]:
            compound *= (1 + r["total_return_pct"] / 100)
        compound_return = (compound - 1) * 100
        compound_cagr = annualize(compound_return, test_bars_est)

        # Bootstrap
        boot = bootstrap_confidence(result["combined_pnl"])

        # Sharpe and drawdown aggregated across folds
        all_equity = []
        for r in result["fold_results"]:
            eq = r["equity_curve"]["equity"] if "equity" in r["equity_curve"].columns else pd.Series()
            all_equity.append(eq)
        combined_equity = pd.concat(all_equity) if all_equity else pd.Series([100000])
        rolling_max = combined_equity.cummax()
        max_dd = float(((combined_equity - rolling_max) / rolling_max).min() * 100)

        # Win/loss metrics from combined trades
        all_closed = []
        for r in result["fold_results"]:
            trades = r["trades"]
            if len(trades):
                closed = trades[trades["type"].isin(["SELL", "COVER"])]
                if len(closed):
                    all_closed.append(closed)
        if all_closed:
            closed_df = pd.concat(all_closed)
            wins = closed_df[closed_df["pnl"] > 0]
            losses = closed_df[closed_df["pnl"] <= 0]
            win_rate = len(wins) / len(closed_df) * 100
            avg_win = float(wins["pnl_pct"].mean()) if len(wins) else 0.0
            avg_loss = float(losses["pnl_pct"].mean()) if len(losses) else 0.0
            gp = wins["pnl"].sum()
            gl = abs(losses["pnl"].sum())
            pf = gp / gl if gl > 0 else float("inf")
        else:
            win_rate = avg_win = avg_loss = pf = 0.0

        return {
            "ticker": ticker,
            "total_return_pct": result["total_return_pct"],
            "compound_return_pct": round(compound_return, 2),
            "cagr_pct": cagr,
            "compound_cagr_pct": compound_cagr,
            "avg_win_rate": result["avg_win_rate"],
            "total_trades": result["total_trades"],
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate_pct": round(win_rate, 2),
            "avg_win_pct": round(avg_win, 3),
            "avg_loss_pct": round(avg_loss, 3),
            "profit_factor": round(pf, 3),
            "p_profitable": boot.get("p_profitable", 0),
            "confidence": boot.get("confidence_label", "N/A"),
            "adx_from_optuna": result.get("adx_from_optuna", False),
            "error": None,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e), "cagr_pct": 0.0, "total_trades": 0}
    finally:
        CONFIG.bot.min_confidence = orig_conf
        CONFIG.bot.stop_loss_pct = orig_stop
        CONFIG.bot.take_profit_pct = orig_take
        CONFIG.ml.label_pct_low = orig_lbl_lo
        CONFIG.ml.label_pct_high = orig_lbl_hi
        CONFIG.ml.target_bars_by_interval["1h"] = orig_tbars
        CONFIG.bot.max_position_pct = orig_pos_pct


def print_results_table(rows: list, label: str):
    """Print a formatted results table."""
    rows = [r for r in rows if not r.get("error")]
    if not rows:
        print("No results.")
        return

    rows_sorted = sorted(rows, key=lambda x: x.get("cagr_pct", 0), reverse=True)
    sep = "=" * 92
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  {'TICKER':<6} {'CAGR':>7} {'WF_TOT':>8} {'WIN%':>6} {'PF':>6} {'DD%':>6} "
          f"{'TRADES':>7} {'P(+)':>6} {'CONF':>8}")
    print(sep)
    for r in rows_sorted:
        pprof = f"{r.get('p_profitable', 0)*100:.0f}%"
        marker = " <--" if r.get("cagr_pct", 0) >= 15 else ""
        print(f"  {r['ticker']:<6} {r.get('cagr_pct', 0):>+6.1f}%  "
              f"{r.get('total_return_pct', 0):>+7.2f}%  "
              f"{r.get('win_rate_pct', 0):>5.1f}%  "
              f"{r.get('profit_factor', 0):>5.2f}  "
              f"{r.get('max_drawdown_pct', 0):>5.1f}%  "
              f"{r.get('total_trades', 0):>7}  "
              f"{pprof:>6}  "
              f"{r.get('confidence', 'N/A'):>8}{marker}")
    print(sep)

    cagrs = [r.get("cagr_pct", 0) for r in rows_sorted]
    profitable = [r for r in rows_sorted if r.get("cagr_pct", 0) > 0 and r.get("total_trades", 0) >= 10]
    top10 = rows_sorted[:10]
    avg_cagr = sum(cagrs) / len(cagrs) if cagrs else 0
    top10_avg = sum(r.get("cagr_pct", 0) for r in top10) / len(top10) if top10 else 0
    avg_win_rate = sum(r.get("win_rate_pct", 0) for r in rows_sorted) / len(rows_sorted)
    avg_dd = sum(r.get("max_drawdown_pct", 0) for r in rows_sorted) / len(rows_sorted)
    avg_pf = sum(r.get("profit_factor", 0) for r in rows_sorted if r.get("profit_factor", 0) < 100) / max(1, len(rows_sorted))

    print(f"\n  Summary ({len(rows_sorted)} tickers):")
    print(f"    Avg CAGR (all)         : {avg_cagr:+.1f}%")
    print(f"    Avg CAGR (top 10)      : {top10_avg:+.1f}%")
    print(f"    Avg win rate           : {avg_win_rate:.1f}%")
    print(f"    Avg max drawdown       : {avg_dd:.1f}%")
    print(f"    Avg profit factor      : {avg_pf:.2f}")
    print(f"    Profitable tickers     : {len(profitable)}/{len(rows_sorted)}")
    print(f"    Tickers at 15%+ CAGR   : {sum(1 for r in rows_sorted if r.get('cagr_pct', 0) >= 15)}")
    print(f"    Tickers at 10%+ CAGR   : {sum(1 for r in rows_sorted if r.get('cagr_pct', 0) >= 10)}")
    if avg_cagr >= 20:
        print(f"\n    STATUS: TARGET MET - {avg_cagr:+.1f}% average CAGR")
    elif avg_cagr >= 15:
        print(f"\n    STATUS: CLOSE - {avg_cagr:+.1f}% average CAGR (gap: {20-avg_cagr:.1f}%)")
    elif avg_cagr >= 10:
        print(f"\n    STATUS: PROGRESS - {avg_cagr:+.1f}% average CAGR (gap: {20-avg_cagr:.1f}%)")
    else:
        print(f"\n    STATUS: BELOW TARGET - {avg_cagr:+.1f}% average CAGR")


# ------------------------------------------------------------------ #
# Experiments                                                         #
# ------------------------------------------------------------------ #

def run_baseline(tickers: list) -> list:
    print(f"\n{'#'*60}")
    print(f"# BASELINE: {len(tickers)} tickers, conf>{CONFIG.bot.min_confidence}, "
          f"adx>{CONFIG.bot.adx_filter_min}, {N_FOLDS}-fold WF")
    print(f"{'#'*60}")
    rows = []
    for t in sorted(tickers):
        r = run_wf(t)
        if r.get("error"):
            print(f"  {t}: FAILED - {r['error']}")
        else:
            print(f"  {t:<6} CAGR={r['cagr_pct']:>+5.1f}%  win={r['win_rate_pct']:.1f}%  "
                  f"trades={r['total_trades']}  PF={r['profit_factor']:.2f}  "
                  f"DD={r['max_drawdown_pct']:.1f}%")
        rows.append(r)
    print_results_table(rows, f"BASELINE  (min_conf={CONFIG.bot.min_confidence}, adx_min={CONFIG.bot.adx_filter_min})")
    return rows


def experiment_confidence(tickers: list, thresholds: list = None) -> dict:
    """Experiment A: test confidence thresholds 0.52 to 0.80."""
    if thresholds is None:
        thresholds = [0.52, 0.55, 0.60, 0.65, 0.70, 0.75]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT A: Confidence threshold sweep")
    print(f"{'#'*60}")

    results_by_threshold = {}

    for conf in thresholds:
        print(f"\n  Testing conf > {conf}...")
        rows = []
        for t in sorted(tickers):
            r = run_wf(t, conf_threshold=conf)
            rows.append(r)

        valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 5]
        if not valid:
            print(f"    No valid results at conf>{conf}")
            continue

        avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid)
        avg_win = sum(r["win_rate_pct"] for r in valid) / len(valid)
        avg_trades = sum(r["total_trades"] for r in valid) / len(valid)
        avg_pf = sum(r["profit_factor"] for r in valid if r["profit_factor"] < 100) / len(valid)
        above15 = sum(1 for r in valid if r["cagr_pct"] >= 15)

        results_by_threshold[conf] = {
            "avg_cagr": round(avg_cagr, 2),
            "avg_win_rate": round(avg_win, 2),
            "avg_trades": round(avg_trades, 1),
            "avg_pf": round(avg_pf, 3),
            "above_15pct": above15,
            "n_valid": len(valid),
        }

        print(f"    conf>{conf:.2f}: avg_CAGR={avg_cagr:+.1f}%  win={avg_win:.1f}%  "
              f"trades/ticker={avg_trades:.0f}  PF={avg_pf:.2f}  tickers>=15%: {above15}")

    # Summary
    print(f"\n  Confidence sweep summary:")
    print(f"  {'CONF':>6}  {'CAGR':>7}  {'WIN%':>6}  {'TRADES':>8}  {'PF':>6}  {'>=15%':>7}")
    for conf, m in sorted(results_by_threshold.items()):
        marker = " <-- best" if m["avg_cagr"] == max(v["avg_cagr"] for v in results_by_threshold.values()) else ""
        print(f"  {conf:>6.2f}  {m['avg_cagr']:>+6.1f}%  {m['avg_win_rate']:>5.1f}%  "
              f"{m['avg_trades']:>8.0f}  {m['avg_pf']:>6.3f}  {m['above_15pct']:>7}{marker}")

    return results_by_threshold


def experiment_d_trend_gate(tickers: list) -> dict:
    """Experiment B1: rule-based daily trend gate (d_trend_score threshold sweep)."""
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT B1: Daily trend score gate (rule-based)")
    print(f"# d_trend_score = RSI>55/55 + MACD sign + price vs SMA50")
    print(f"# Range -3 to +3. Gate blocks signals that fight daily trend.")
    print(f"{'#'*60}")

    results_by_score = {}

    for min_score in [1, 2, 3]:
        print(f"\n  Testing d_trend_min_score={min_score}...")
        rows = []
        for t in sorted(tickers):
            r = run_wf(t, d_trend_gate=True, d_trend_min_score=min_score)
            rows.append(r)

        valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 5]
        if not valid:
            continue

        avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid)
        avg_win = sum(r["win_rate_pct"] for r in valid) / len(valid)
        avg_trades = sum(r["total_trades"] for r in valid) / len(valid)
        avg_pf = sum(r["profit_factor"] for r in valid if r["profit_factor"] < 100) / len(valid)
        above15 = sum(1 for r in valid if r["cagr_pct"] >= 15)
        above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)

        results_by_score[min_score] = {
            "avg_cagr": round(avg_cagr, 2),
            "avg_win_rate": round(avg_win, 2),
            "avg_trades": round(avg_trades, 1),
            "avg_pf": round(avg_pf, 3),
            "above_15pct": above15,
            "above_10pct": above10,
            "n_valid": len(valid),
        }
        print(f"    min_score={min_score}: avg_CAGR={avg_cagr:+.1f}%  win={avg_win:.1f}%  "
              f"trades/ticker={avg_trades:.0f}  PF={avg_pf:.2f}  tickers>=10%: {above10}  >=15%: {above15}")

    print(f"\n  d_trend gate sweep summary:")
    print(f"  {'SCORE':>7}  {'CAGR':>7}  {'WIN%':>6}  {'TRADES':>8}  {'PF':>6}  {'>=10%':>7}  {'>=15%':>7}")
    for score, m in sorted(results_by_score.items()):
        marker = " <-- best" if m["avg_cagr"] == max(v["avg_cagr"] for v in results_by_score.values()) else ""
        print(f"  {score:>7}  {m['avg_cagr']:>+6.1f}%  {m['avg_win_rate']:>5.1f}%  "
              f"{m['avg_trades']:>8.0f}  {m['avg_pf']:>6.3f}  {m['above_10pct']:>7}  {m['above_15pct']:>7}{marker}")

    return results_by_score


def experiment_risk_reward(tickers: list) -> dict:
    """Experiment C: stop/take ratio sweep — tests whether wider reward:risk improves CAGR."""
    combos = [
        (0.015, 0.03),   # 1.5% stop, 3% take (tighter, 2:1)
        (0.02,  0.04),   # 2% stop, 4% take  (baseline, 2:1)
        (0.02,  0.06),   # 2% stop, 6% take  (3:1)
        (0.02,  0.08),   # 2% stop, 8% take  (4:1)
        (0.015, 0.045),  # 1.5% stop, 4.5% take (3:1)
        (0.01,  0.04),   # 1% stop, 4% take  (4:1 tight)
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT C: Stop/take ratio sweep")
    print(f"# Hypothesis: wider reward:risk ratio improves CAGR")
    print(f"{'#'*60}")

    orig_stop = CONFIG.bot.stop_loss_pct
    orig_take = CONFIG.bot.take_profit_pct
    results_by_combo = {}

    try:
        for stop, take in combos:
            CONFIG.bot.stop_loss_pct = stop
            CONFIG.bot.take_profit_pct = take
            label = f"{stop*100:.1f}%/{take*100:.1f}%"
            print(f"\n  Testing stop={stop*100:.1f}%, take={take*100:.1f}% ...")

            rows = []
            for t in sorted(tickers):
                r = run_wf(t)
                rows.append(r)

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 5]
            if not valid:
                continue

            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid)
            avg_win = sum(r["win_rate_pct"] for r in valid) / len(valid)
            avg_trades = sum(r["total_trades"] for r in valid) / len(valid)
            avg_pf = sum(r["profit_factor"] for r in valid if r["profit_factor"] < 100) / len(valid)
            above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)
            above5 = sum(1 for r in valid if r["cagr_pct"] >= 5)

            results_by_combo[label] = {
                "stop": stop, "take": take,
                "avg_cagr": round(avg_cagr, 2),
                "avg_win_rate": round(avg_win, 2),
                "avg_trades": round(avg_trades, 1),
                "avg_pf": round(avg_pf, 3),
                "above_10pct": above10,
                "above_5pct": above5,
                "n_valid": len(valid),
            }
            print(f"    {label}: avg_CAGR={avg_cagr:+.1f}%  win={avg_win:.1f}%  "
                  f"trades/t={avg_trades:.0f}  PF={avg_pf:.2f}  >=5%: {above5}  >=10%: {above10}")
    finally:
        CONFIG.bot.stop_loss_pct = orig_stop
        CONFIG.bot.take_profit_pct = orig_take

    print(f"\n  Risk/Reward sweep summary:")
    print(f"  {'COMBO':>10}  {'CAGR':>7}  {'WIN%':>6}  {'TRADES':>8}  {'PF':>6}  {'>=5%':>6}  {'>=10%':>7}")
    best_cagr = max(v["avg_cagr"] for v in results_by_combo.values()) if results_by_combo else 0
    for combo, m in results_by_combo.items():
        marker = " <-- best" if m["avg_cagr"] == best_cagr else ""
        print(f"  {combo:>10}  {m['avg_cagr']:>+6.1f}%  {m['avg_win_rate']:>5.1f}%  "
              f"{m['avg_trades']:>8.0f}  {m['avg_pf']:>6.3f}  {m['above_5pct']:>6}  {m['above_10pct']:>7}{marker}")

    return results_by_combo


def experiment_no_flat_exit(tickers: list) -> list:
    """Experiment D: don't close on FLAT signal — only close on opposing signal or stop/take."""
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT D: Ignore FLAT exits")
    print(f"# Hypothesis: FLAT signals cut winners short. Hold until")
    print(f"# opposing signal or stop/take to let winners run more.")
    print(f"{'#'*60}")
    rows = []
    for t in sorted(tickers):
        r = run_wf(t, ignore_flat_exit=True)
        if r.get("error"):
            print(f"  {t}: FAILED - {r['error']}")
        else:
            print(f"  {t:<6} CAGR={r['cagr_pct']:>+5.1f}%  win={r['win_rate_pct']:.1f}%  "
                  f"trades={r['total_trades']}  PF={r['profit_factor']:.2f}  "
                  f"DD={r['max_drawdown_pct']:.1f}%")
        rows.append(r)
    print_results_table(rows, "EXPERIMENT D: No-FLAT exits")
    return rows


def experiment_trailing_stop(tickers: list) -> dict:
    """Experiment E: trailing stop sweep — activate trailing stop at different distances."""
    trail_values = [0.005, 0.01, 0.015, 0.02]  # 0.5%, 1%, 1.5%, 2% trail distance

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT E: Trailing stop sweep")
    print(f"# Trail activates once gain >= trail_pct; stop trails peak by trail_pct")
    print(f"{'#'*60}")

    results_by_trail = {}
    for trail in trail_values:
        print(f"\n  Testing trailing_stop={trail*100:.1f}%...")
        rows = []
        for t in sorted(tickers):
            r = run_wf(t, trailing_stop_pct=trail)
            rows.append(r)

        valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 5]
        if not valid:
            continue

        avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid)
        avg_win = sum(r["win_rate_pct"] for r in valid) / len(valid)
        avg_trades = sum(r["total_trades"] for r in valid) / len(valid)
        avg_pf = sum(r["profit_factor"] for r in valid if r["profit_factor"] < 100) / len(valid)
        above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)

        results_by_trail[trail] = {
            "avg_cagr": round(avg_cagr, 2),
            "avg_win_rate": round(avg_win, 2),
            "avg_trades": round(avg_trades, 1),
            "avg_pf": round(avg_pf, 3),
            "above_10pct": above10,
            "n_valid": len(valid),
        }
        print(f"    trail={trail*100:.1f}%: avg_CAGR={avg_cagr:+.1f}%  win={avg_win:.1f}%  "
              f"trades/t={avg_trades:.0f}  PF={avg_pf:.2f}  >=10%: {above10}")

    print(f"\n  Trailing stop sweep summary:")
    print(f"  {'TRAIL%':>7}  {'CAGR':>7}  {'WIN%':>6}  {'TRADES':>8}  {'PF':>6}  {'>=10%':>7}")
    best_cagr = max(v["avg_cagr"] for v in results_by_trail.values()) if results_by_trail else 0
    for trail, m in sorted(results_by_trail.items()):
        marker = " <-- best" if m["avg_cagr"] == best_cagr else ""
        print(f"  {trail*100:>6.1f}%  {m['avg_cagr']:>+6.1f}%  {m['avg_win_rate']:>5.1f}%  "
              f"{m['avg_trades']:>8.0f}  {m['avg_pf']:>6.3f}  {m['above_10pct']:>7}{marker}")
    return results_by_trail


def experiment_long_only(tickers: list) -> list:
    """Experiment F: long-only — disable shorts. Avoids fighting bull-bias in market."""
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT F: Long-only (no shorts)")
    print(f"# Hypothesis: short trades lose money in a generally bullish")
    print(f"# market; removing them improves average CAGR.")
    print(f"{'#'*60}")
    rows = []
    for t in sorted(tickers):
        r = run_wf(t, allow_short=False)
        if r.get("error"):
            print(f"  {t}: FAILED - {r['error']}")
        else:
            print(f"  {t:<6} CAGR={r['cagr_pct']:>+5.1f}%  win={r['win_rate_pct']:.1f}%  "
                  f"trades={r['total_trades']}  PF={r['profit_factor']:.2f}  "
                  f"DD={r['max_drawdown_pct']:.1f}%")
        rows.append(r)
    print_results_table(rows, "EXPERIMENT F: Long-only")
    return rows


def experiment_combo(tickers: list, combos: list = None) -> dict:
    """Experiment G: test promising combinations of best individual settings."""
    if combos is None:
        combos = [
            {"label": "no_flat+long_only",              "ignore_flat_exit": True,  "allow_short": False},
            {"label": "no_flat+wide_take",               "ignore_flat_exit": True,  "take_profit": 0.06},
            {"label": "no_flat+trail1pct",               "ignore_flat_exit": True,  "trailing_stop_pct": 0.01},
            {"label": "long_only+wide_take",             "allow_short": False,      "take_profit": 0.06},
            {"label": "no_flat+long+wide_take",          "ignore_flat_exit": True,  "allow_short": False, "take_profit": 0.06},
            {"label": "conf60+no_flat+long",             "conf_threshold": 0.60,    "ignore_flat_exit": True, "allow_short": False},
            {"label": "conf65+no_flat+wide_take",        "conf_threshold": 0.65,    "ignore_flat_exit": True, "take_profit": 0.06},
            # Daily confirm combinations
            {"label": "daily_confirm",                   "daily_confirm": True},
            {"label": "daily_confirm+no_flat",           "daily_confirm": True,     "ignore_flat_exit": True},
            {"label": "daily_confirm+long_only",         "daily_confirm": True,     "allow_short": False},
            {"label": "daily_confirm+no_flat+long",      "daily_confirm": True,     "ignore_flat_exit": True, "allow_short": False},
            {"label": "daily_confirm+wide_take",         "daily_confirm": True,     "take_profit": 0.06},
            {"label": "daily_confirm+no_flat+wide_take", "daily_confirm": True,     "ignore_flat_exit": True, "take_profit": 0.06},
        ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT G: Combo sweep")
    print(f"{'#'*60}")

    results = {}
    for combo in combos:
        label = combo.pop("label")
        print(f"\n  Testing {label}...")
        rows = []
        for t in sorted(tickers):
            r = run_wf(t, **combo)
            rows.append(r)

        valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 5]
        if not valid:
            combo["label"] = label
            continue

        avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid)
        avg_win = sum(r["win_rate_pct"] for r in valid) / len(valid)
        avg_trades = sum(r["total_trades"] for r in valid) / len(valid)
        avg_pf = sum(r["profit_factor"] for r in valid if r["profit_factor"] < 100) / len(valid)
        above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)
        above5 = sum(1 for r in valid if r["cagr_pct"] >= 5)

        results[label] = {
            "avg_cagr": round(avg_cagr, 2),
            "avg_win_rate": round(avg_win, 2),
            "avg_trades": round(avg_trades, 1),
            "avg_pf": round(avg_pf, 3),
            "above_10pct": above10,
            "above_5pct": above5,
            "n_valid": len(valid),
        }
        print(f"    {label}: avg_CAGR={avg_cagr:+.1f}%  win={avg_win:.1f}%  "
              f"PF={avg_pf:.2f}  >=5%: {above5}  >=10%: {above10}")
        combo["label"] = label  # restore for safety

    print(f"\n  Combo sweep summary (ranked by CAGR):")
    print(f"  {'COMBO':<35}  {'CAGR':>7}  {'WIN%':>6}  {'TRADES':>8}  {'PF':>6}  {'>=5%':>6}  {'>=10%':>7}")
    best_cagr = max(v["avg_cagr"] for v in results.values()) if results else 0
    for lbl, m in sorted(results.items(), key=lambda x: -x[1]["avg_cagr"]):
        marker = " <--" if m["avg_cagr"] == best_cagr else ""
        print(f"  {lbl:<35}  {m['avg_cagr']:>+6.1f}%  {m['avg_win_rate']:>5.1f}%  "
              f"{m['avg_trades']:>8.0f}  {m['avg_pf']:>6.3f}  {m['above_5pct']:>6}  {m['above_10pct']:>7}{marker}")
    return results


def experiment_daily_confirm(tickers: list) -> list:
    """Experiment B: multi-timeframe confirmation — daily ML model gates hourly signals."""
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT B: Daily model confirmation gate")
    print(f"# Hypothesis: hourly signals confirmed by daily direction")
    print(f"# should have higher win rate and reduce whipsaws.")
    print(f"{'#'*60}")

    rows = []
    for t in sorted(tickers):
        r = run_wf(t, daily_confirm=True)
        if r.get("error"):
            print(f"  {t}: FAILED - {r['error']}")
        else:
            print(f"  {t:<6} CAGR={r['cagr_pct']:>+5.1f}%  win={r['win_rate_pct']:.1f}%  "
                  f"trades={r['total_trades']}  PF={r['profit_factor']:.2f}  "
                  f"DD={r['max_drawdown_pct']:.1f}%")
        rows.append(r)
    print_results_table(rows, "EXPERIMENT B: Daily confirmation gate")
    return rows


def experiment_daily_combos(tickers: list) -> dict:
    """Experiment G2: daily_confirm combinations — the most promising direction."""
    combos = [
        {"label": "daily_confirm",                   "daily_confirm": True},
        {"label": "daily_confirm+no_flat",           "daily_confirm": True, "ignore_flat_exit": True},
        {"label": "daily_confirm+long_only",         "daily_confirm": True, "allow_short": False},
        {"label": "daily_confirm+no_flat+long",      "daily_confirm": True, "ignore_flat_exit": True, "allow_short": False},
        {"label": "daily_confirm+wide_take",         "daily_confirm": True, "take_profit": 0.06},
        {"label": "daily_confirm+no_flat+wide_take", "daily_confirm": True, "ignore_flat_exit": True, "take_profit": 0.06},
        {"label": "daily_confirm+long+wide_take",    "daily_confirm": True, "allow_short": False, "take_profit": 0.06},
        {"label": "daily_confirm+all",               "daily_confirm": True, "ignore_flat_exit": True, "allow_short": False, "take_profit": 0.06},
    ]
    return experiment_combo(tickers, combos)


def experiment_conf_dc(tickers: list) -> dict:
    """Experiment K: confidence threshold sweep WITH daily_confirm gate.
    With daily_confirm already pre-screening setups, find the optimal
    confidence threshold."""
    thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]
    combos = [
        {"label": f"dc_conf{int(c*100)}", "daily_confirm": True, "conf_threshold_param": c}
        for c in thresholds
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT K: Confidence threshold sweep + daily_confirm")
    print(f"{'#'*60}")
    return experiment_combo(tickers, combos)


def experiment_conviction_dc(tickers: list) -> dict:
    """Experiment L: conviction spread sweep WITH daily_confirm gate.
    Only trade when top-class probability leads second-place by at least spread_pct.
    Filters out ambiguous predictions where UP and DOWN are nearly tied."""
    spreads = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    combos = [
        {"label": f"dc_conv{int(s*100):02d}", "daily_confirm": True, "min_conviction_spread": s}
        for s in spreads
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT L: Conviction spread sweep + daily_confirm")
    print(f"{'#'*60}")
    return experiment_combo(tickers, combos)


def experiment_optimize_dc(tickers: list, n_trials: int = 60) -> dict:
    """Experiment J: re-optimize per-ticker hyperparameters WITH daily_confirm gate.
    Finds the best ADX threshold + XGBoost params for the daily_confirm scenario,
    then verifies the improvement with a walk-forward test."""
    from ml.optimizer import optimize_ticker

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT J: Optuna re-optimization with daily_confirm gate")
    print(f"# Finds ADX thresholds + XGBoost params optimized for daily_confirm")
    print(f"{'#'*60}")

    opt_results = {}
    for t in sorted(tickers):
        print(f"\n  Optimizing {t} (daily_confirm=True, {n_trials} trials)...")
        try:
            res = optimize_ticker(t, n_trials=n_trials, n_folds=3, daily_confirm=True, allow_short=False)
            opt_results[t] = {
                "adx": res["adx_filter_min"],
                "score": res["best_score"],
                "n_estimators": res["xgb_params"].get("n_estimators"),
            }
        except Exception as e:
            print(f"  {t} failed: {e}")

    print(f"\n  Optimization complete. Testing with new params...")
    rows = []
    for t in sorted(tickers):
        r = run_wf(t, daily_confirm=True, allow_short=False)
        if r.get("error"):
            print(f"  {t}: FAILED - {r['error']}")
        else:
            print(f"  {t:<6} CAGR={r['cagr_pct']:>+5.1f}%  win={r['win_rate_pct']:.1f}%  "
                  f"trades={r['total_trades']}  PF={r['profit_factor']:.2f}  "
                  f"DD={r['max_drawdown_pct']:.1f}%")
        rows.append(r)

    print_results_table(rows, "EXPERIMENT J: daily_confirm with re-optimized params")
    return {"opt_results": opt_results, "wf_results": rows}


def experiment_daily_risk_reward(tickers: list) -> dict:
    """Experiment I: risk/reward sweep WITH daily_confirm gate.
    With daily_confirm already filtering setups, test whether different
    stop/take ratios further improve CAGR."""
    combos = [
        # Baseline daily_confirm (2%/4%)
        {"label": "dc_2pct_4pct",   "daily_confirm": True, "stop_loss": 0.02, "take_profit": 0.04},
        # Tighter stops → fewer over-sized losses
        {"label": "dc_1.5pct_4pct", "daily_confirm": True, "stop_loss": 0.015, "take_profit": 0.04},
        {"label": "dc_1pct_4pct",   "daily_confirm": True, "stop_loss": 0.01,  "take_profit": 0.04},
        # Wider takes → bigger wins
        {"label": "dc_2pct_6pct",   "daily_confirm": True, "stop_loss": 0.02,  "take_profit": 0.06},
        {"label": "dc_2pct_8pct",   "daily_confirm": True, "stop_loss": 0.02,  "take_profit": 0.08},
        # Tighter stop + wider take → 4:1 ratio
        {"label": "dc_1pct_4pct_long",  "daily_confirm": True, "stop_loss": 0.01, "take_profit": 0.04, "allow_short": False},
        {"label": "dc_1.5pct_6pct",     "daily_confirm": True, "stop_loss": 0.015, "take_profit": 0.06},
        {"label": "dc_1pct_5pct",       "daily_confirm": True, "stop_loss": 0.01,  "take_profit": 0.05},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT I: Risk/reward sweep with daily_confirm gate")
    print(f"{'#'*60}")
    return experiment_combo(tickers, combos)


def experiment_first_touch(tickers: list) -> list:
    """Experiment H: retrain with first-touch target — predict will price hit TP before SL?
    This maps directly to trading outcomes instead of percentile-rank direction."""
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT H: First-touch target retraining")
    print(f"# Label 2(UP)=price hits +{CONFIG.bot.take_profit_pct*100:.0f}% TP first within ~24h")
    print(f"# Label 0(DOWN)=price hits -{CONFIG.bot.stop_loss_pct*100:.0f}% SL first")
    print(f"# Label 1(FLAT)=neither hit within window")
    print(f"{'#'*60}")

    orig_flag = CONFIG.ml.use_first_touch_target
    CONFIG.ml.use_first_touch_target = True
    rows = []
    try:
        for t in sorted(tickers):
            r = run_wf(t)
            if r.get("error"):
                print(f"  {t}: FAILED - {r['error']}")
            else:
                print(f"  {t:<6} CAGR={r['cagr_pct']:>+5.1f}%  win={r['win_rate_pct']:.1f}%  "
                      f"trades={r['total_trades']}  PF={r['profit_factor']:.2f}  "
                      f"DD={r['max_drawdown_pct']:.1f}%")
            rows.append(r)
    finally:
        CONFIG.ml.use_first_touch_target = orig_flag

    print_results_table(rows, "EXPERIMENT H: First-touch target")
    return rows


def experiment_orb_baseline(tickers: list) -> list:
    """Experiment P: baseline with ORB features added (auto-included via add_all_indicators).
    Compare against the original baseline to measure ORB feature lift."""
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT P: ORB-enhanced baseline")
    print(f"# New features: price_vs_orb_high, price_vs_orb_low, intraday_return,")
    print(f"#               orb_width, overnight_gap")
    print(f"{'#'*60}")
    rows = []
    for t in sorted(tickers):
        r = run_wf(t)
        if r.get("error"):
            print(f"  {t}: FAILED - {r['error']}")
        else:
            print(f"  {t:<6} CAGR={r['cagr_pct']:>+5.1f}%  win={r['win_rate_pct']:.1f}%  "
                  f"trades={r['total_trades']}  PF={r['profit_factor']:.2f}  "
                  f"DD={r['max_drawdown_pct']:.1f}%")
        rows.append(r)
    print_results_table(rows, "EXPERIMENT P: ORB-enhanced baseline")
    return rows


def experiment_orb_dc(tickers: list) -> list:
    """Experiment Q: ORB features + daily_confirm — the best combo with new signals."""
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT Q: ORB features + daily_confirm")
    print(f"{'#'*60}")
    rows = []
    for t in sorted(tickers):
        r = run_wf(t, daily_confirm=True)
        if r.get("error"):
            print(f"  {t}: FAILED - {r['error']}")
        else:
            print(f"  {t:<6} CAGR={r['cagr_pct']:>+5.1f}%  win={r['win_rate_pct']:.1f}%  "
                  f"trades={r['total_trades']}  PF={r['profit_factor']:.2f}  "
                  f"DD={r['max_drawdown_pct']:.1f}%")
        rows.append(r)
    print_results_table(rows, "EXPERIMENT Q: ORB + daily_confirm")
    return rows


def experiment_session_filter(tickers: list) -> dict:
    """Experiment M: time-of-day filter — only open entries during allowed market hours.
    US market hours are 9:30–16:00 ET; bar timestamps use bar-open hour.
    Morning bars (hour 9, 10, 11) = 9:30–12:30pm; afternoon (13, 14, 15) = 1:30–4pm."""
    # ET bar-open hours: 9=9:30am, 10=10:30am, 11=11:30am, 12=12:30pm, 13=1:30pm, 14=2:30pm, 15=3:30pm
    sessions = [
        {"label": "all_hours",   "session_hours": None},
        {"label": "morning_2h",  "session_hours": frozenset({9, 10})},
        {"label": "morning_3h",  "session_hours": frozenset({9, 10, 11})},
        {"label": "skip_midday", "session_hours": frozenset({9, 10, 14, 15})},
        {"label": "afternoon",   "session_hours": frozenset({13, 14, 15})},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT M: Time-of-day session filter")
    print(f"# ET bar hours: 9=9:30am  10=10:30am  11=11:30am  12=12:30pm")
    print(f"#               13=1:30pm  14=2:30pm  15=3:30pm")
    print(f"{'#'*60}")

    results = {}
    for sess in sessions:
        label = sess["label"]
        hours = sess["session_hours"]
        hour_str = str(sorted(hours)) if hours else "all"
        print(f"\n  Testing {label} (hours={hour_str})...")
        rows = []
        for t in sorted(tickers):
            r = run_wf(t, session_hours=hours)
            rows.append(r)

        valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
        if not valid:
            print(f"    No valid results for {label}")
            continue

        avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid)
        avg_win = sum(r["win_rate_pct"] for r in valid) / len(valid)
        avg_trades = sum(r["total_trades"] for r in valid) / len(valid)
        avg_pf = sum(r["profit_factor"] for r in valid if r["profit_factor"] < 100) / max(1, len(valid))
        above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)

        results[label] = {
            "avg_cagr": round(avg_cagr, 2),
            "avg_win_rate": round(avg_win, 2),
            "avg_trades": round(avg_trades, 1),
            "avg_pf": round(avg_pf, 3),
            "above_10pct": above10,
            "n_valid": len(valid),
        }
        print(f"    {label}: avg_CAGR={avg_cagr:+.1f}%  win={avg_win:.1f}%  "
              f"trades/ticker={avg_trades:.0f}  PF={avg_pf:.2f}  >=10%: {above10}")

    print(f"\n  Session filter summary:")
    print(f"  {'LABEL':<14} {'CAGR':>7}  {'WIN%':>6}  {'TRADES':>8}  {'PF':>6}  {'>=10%':>6}")
    for label, m in results.items():
        marker = " <-- best" if m["avg_cagr"] == max(v["avg_cagr"] for v in results.values()) else ""
        print(f"  {label:<14} {m['avg_cagr']:>+6.1f}%  {m['avg_win_rate']:>5.1f}%  "
              f"{m['avg_trades']:>8.0f}  {m['avg_pf']:>6.3f}  {m['above_10pct']:>6}{marker}")

    return results


def experiment_recency_weight(tickers: list) -> dict:
    """Experiment R: exponential recency weighting on training samples.
    Recent bars get weight 1.0; bars from halflife bars ago get weight 0.5.
    Tests whether emphasizing recent market conditions improves OOS performance."""
    halflives = [0, 250, 500, 750, 1000]  # 0=disabled; 250≈1.5mo, 500≈3mo, 1000≈6mo in hourly bars

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT R: Recency-weighted training (exponential decay)")
    print(f"# halflife in hourly bars: 250~1.5mo  500~3mo  750~4.5mo  1000~6mo")
    print(f"{'#'*60}")

    results = {}
    for hl in halflives:
        label = "uniform" if hl == 0 else f"hl{hl}"
        print(f"\n  Testing halflife={hl} ({label})...")
        rows = []
        for t in sorted(tickers):
            r = run_wf(t, recency_halflife=hl)
            rows.append(r)

        valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
        if not valid:
            continue

        avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid)
        avg_win = sum(r["win_rate_pct"] for r in valid) / len(valid)
        avg_trades = sum(r["total_trades"] for r in valid) / len(valid)
        avg_pf = sum(r["profit_factor"] for r in valid if r["profit_factor"] < 100) / max(1, len(valid))
        above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)

        results[label] = {
            "halflife": hl,
            "avg_cagr": round(avg_cagr, 2),
            "avg_win_rate": round(avg_win, 2),
            "avg_trades": round(avg_trades, 1),
            "avg_pf": round(avg_pf, 3),
            "above_10pct": above10,
            "n_valid": len(valid),
        }
        print(f"    halflife={hl}: avg_CAGR={avg_cagr:+.1f}%  win={avg_win:.1f}%  "
              f"trades/ticker={avg_trades:.0f}  PF={avg_pf:.2f}  >=10%: {above10}")

    print(f"\n  Recency weight summary:")
    print(f"  {'LABEL':<12} {'HL':>6}  {'CAGR':>7}  {'WIN%':>6}  {'TRADES':>8}  {'PF':>6}")
    for label, m in results.items():
        marker = " <-- best" if m["avg_cagr"] == max(v["avg_cagr"] for v in results.values()) else ""
        print(f"  {label:<12} {m['halflife']:>6}  {m['avg_cagr']:>+6.1f}%  "
              f"{m['avg_win_rate']:>5.1f}%  {m['avg_trades']:>8.0f}  {m['avg_pf']:>6.3f}{marker}")

    return results


def experiment_spy_align(tickers: list) -> dict:
    """Experiment S: SPY alignment filter — only enter long when SPY close > SMA20 and RSI>50."""
    combos = [
        {"label": "baseline",           "spy_align": False, "daily_confirm": False},
        {"label": "spy_only",           "spy_align": True,  "daily_confirm": False},
        {"label": "spy+dc",             "spy_align": True,  "daily_confirm": True},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT S: SPY alignment filter (hourly trend gate)")
    print(f"# Entry allowed only when SPY close>SMA20 and RSI>50 (for longs)")
    print(f"{'#'*60}")

    results = {}
    for combo in combos:
        label = combo["label"]
        print(f"\n  Testing {label}...")
        rows = []
        for t in sorted(tickers):
            r = run_wf(t, spy_align=combo["spy_align"], daily_confirm=combo["daily_confirm"])
            rows.append(r)

        valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
        if not valid:
            continue

        avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid)
        avg_win = sum(r["win_rate_pct"] for r in valid) / len(valid)
        avg_trades = sum(r["total_trades"] for r in valid) / len(valid)
        avg_pf = sum(r["profit_factor"] for r in valid if r["profit_factor"] < 100) / max(1, len(valid))
        above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)

        results[label] = {
            "avg_cagr": round(avg_cagr, 2),
            "avg_win_rate": round(avg_win, 2),
            "avg_trades": round(avg_trades, 1),
            "avg_pf": round(avg_pf, 3),
            "above_10pct": above10,
            "n_valid": len(valid),
        }
        print(f"    {label}: avg_CAGR={avg_cagr:+.1f}%  win={avg_win:.1f}%  "
              f"trades/ticker={avg_trades:.0f}  PF={avg_pf:.2f}  >=10%: {above10}")

    print(f"\n  SPY alignment summary:")
    print(f"  {'LABEL':<14} {'CAGR':>7}  {'WIN%':>6}  {'TRADES':>8}  {'PF':>6}")
    for label, m in results.items():
        marker = " <-- best" if m["avg_cagr"] == max(v["avg_cagr"] for v in results.values()) else ""
        print(f"  {label:<14} {m['avg_cagr']:>+6.1f}%  {m['avg_win_rate']:>5.1f}%  "
              f"{m['avg_trades']:>8.0f}  {m['avg_pf']:>6.3f}{marker}")

    return results


def experiment_dc_trailing(tickers: list) -> dict:
    """Experiment U: daily_confirm + trailing stop sweep.
    Tests whether trailing stops on top of daily_confirm can improve CAGR
    by locking in profits on large winners."""
    combos = [
        {"label": "dc_base",          "daily_confirm": True, "trailing_stop_pct": 0.0},
        {"label": "dc_trail0.5",      "daily_confirm": True, "trailing_stop_pct": 0.005},
        {"label": "dc_trail1.0",      "daily_confirm": True, "trailing_stop_pct": 0.01},
        {"label": "dc_trail1.5",      "daily_confirm": True, "trailing_stop_pct": 0.015},
        {"label": "dc_trail2.0",      "daily_confirm": True, "trailing_stop_pct": 0.02},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT U: daily_confirm + trailing stop sweep")
    print(f"# Trail pct = minimum move from peak before stop triggers")
    print(f"{'#'*60}")
    return experiment_combo(tickers, combos)


def experiment_dc_combos(tickers: list) -> dict:
    """Experiment T: daily_confirm combined with other filters.
    Tests whether adding session/recency/spy/hold filters ON TOP of daily_confirm
    can push beyond the +1.1% CAGR ceiling."""
    combos = [
        {"label": "dc_base",         "daily_confirm": True},
        {"label": "dc_morning_2h",   "daily_confirm": True, "session_hours": frozenset({9, 10})},
        {"label": "dc_morning_3h",   "daily_confirm": True, "session_hours": frozenset({9, 10, 11})},
        {"label": "dc_skip_midday",  "daily_confirm": True, "session_hours": frozenset({9, 10, 14, 15})},
        {"label": "dc_afternoon",    "daily_confirm": True, "session_hours": frozenset({13, 14, 15})},
        {"label": "dc_hl250",        "daily_confirm": True, "recency_halflife": 250},
        {"label": "dc_hl500",        "daily_confirm": True, "recency_halflife": 500},
        {"label": "dc_hl750",        "daily_confirm": True, "recency_halflife": 750},
        {"label": "dc_spy",          "daily_confirm": True, "spy_align": True},
        {"label": "dc_hold6",        "daily_confirm": True, "max_hold_bars": 6},
        {"label": "dc_hold12",       "daily_confirm": True, "max_hold_bars": 12},
        {"label": "dc_hold24",       "daily_confirm": True, "max_hold_bars": 24},
        {"label": "dc_morn_hl250",   "daily_confirm": True, "session_hours": frozenset({9, 10}), "recency_halflife": 250},
        {"label": "dc_morn_spy",     "daily_confirm": True, "session_hours": frozenset({9, 10}), "spy_align": True},
        {"label": "dc_hl250_spy",    "daily_confirm": True, "recency_halflife": 250, "spy_align": True},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT T: daily_confirm + filter combos")
    print(f"# Tests session/recency/spy/hold ON TOP of daily_confirm")
    print(f"{'#'*60}")
    return experiment_combo(tickers, combos)


def experiment_max_hold(tickers: list) -> dict:
    """Experiment N: max hold time — force-exit after N bars to cap drift losses."""
    hold_values = [0, 3, 6, 12, 24, 48]  # 0 = no limit; 6=1 day, 24=4 days, 48=8 days

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT N: Max hold time (force-exit after N bars)")
    print(f"# 1h bars: 6=1 day, 12=2 days, 24=4 days, 48=8 days")
    print(f"{'#'*60}")

    results = {}
    for n_bars in hold_values:
        label = "no_limit" if n_bars == 0 else f"{n_bars}bars"
        print(f"\n  Testing max_hold={n_bars} ({label})...")
        rows = []
        for t in sorted(tickers):
            r = run_wf(t, max_hold_bars=n_bars)
            rows.append(r)

        valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
        if not valid:
            continue

        avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid)
        avg_win = sum(r["win_rate_pct"] for r in valid) / len(valid)
        avg_trades = sum(r["total_trades"] for r in valid) / len(valid)
        avg_pf = sum(r["profit_factor"] for r in valid if r["profit_factor"] < 100) / max(1, len(valid))
        above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)

        results[label] = {
            "max_hold_bars": n_bars,
            "avg_cagr": round(avg_cagr, 2),
            "avg_win_rate": round(avg_win, 2),
            "avg_trades": round(avg_trades, 1),
            "avg_pf": round(avg_pf, 3),
            "above_10pct": above10,
            "n_valid": len(valid),
        }
        print(f"    max_hold={n_bars}: avg_CAGR={avg_cagr:+.1f}%  win={avg_win:.1f}%  "
              f"trades/ticker={avg_trades:.0f}  PF={avg_pf:.2f}  >=10%: {above10}")

    print(f"\n  Max hold time summary:")
    print(f"  {'LABEL':<12} {'BARS':>5}  {'CAGR':>7}  {'WIN%':>6}  {'TRADES':>8}  {'PF':>6}")
    for label, m in results.items():
        marker = " <-- best" if m["avg_cagr"] == max(v["avg_cagr"] for v in results.values()) else ""
        print(f"  {label:<12} {m['max_hold_bars']:>5}  {m['avg_cagr']:>+6.1f}%  "
              f"{m['avg_win_rate']:>5.1f}%  {m['avg_trades']:>8.0f}  {m['avg_pf']:>6.3f}{marker}")

    return results


def experiment_candle_features(tickers: list) -> dict:
    """Experiment AC: daily_confirm + candle-body features.
    Adds candle_body, candle_body_ratio, upper_shadow, lower_shadow to the
    feature matrix — intrabar OHLC structure not captured by close-only indicators."""
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AC: daily_confirm + candle-body features")
    print(f"# candle_body, candle_body_ratio, upper/lower shadow")
    print(f"{'#'*60}")
    rows = []
    for t in sorted(tickers):
        r = run_wf(t, daily_confirm=True)
        rows.append(r)

    valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
    avg_cagr   = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0.0
    avg_win    = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0.0
    avg_pf     = sum(r["profit_factor"] for r in valid if r["profit_factor"] < 100) / max(1, len(valid))
    avg_dd     = sum(r["max_drawdown_pct"] for r in valid) / len(valid) if valid else 0.0
    above10    = sum(1 for r in valid if r["cagr_pct"] >= 10)

    print(f"\n  {'TICKER':<8} {'CAGR':>6}  {'WIN%':>6}  {'PF':>5}  {'TRADES':>6}")
    for r in sorted(valid, key=lambda x: -x["cagr_pct"]):
        print(f"  {r['ticker']:<8} {r['cagr_pct']:>+5.1f}%  {r['win_rate_pct']:>5.1f}%  "
              f"{r['profit_factor']:>5.2f}  {r['total_trades']:>6}")
    print(f"\n  Avg CAGR: {avg_cagr:+.1f}%  Win: {avg_win:.1f}%  PF: {avg_pf:.2f}  DD: {avg_dd:.1f}%")
    target = "ABOVE TARGET (+1.5%)" if avg_cagr >= 1.5 else f"BELOW TARGET (need >{avg_cagr:+.1f}%)"
    print(f"  STATUS: {target}")

    return {
        "rows": rows,
        "summary": {
            "avg_cagr": round(avg_cagr, 2),
            "avg_win_rate": round(avg_win, 2),
            "avg_profit_factor": round(avg_pf, 2),
            "avg_drawdown": round(avg_dd, 2),
            "above10pct": above10,
        }
    }


def experiment_signal_persist(tickers: list) -> dict:
    """Experiment AB: daily_confirm + signal persistence requirement.
    Require the XGBoost signal to point in the same direction for N consecutive
    bars before entering. Reduces whipsaw entries on single-bar noise signals."""
    combos = [
        {"label": "persist_1",  "daily_confirm": True, "min_signal_bars": 1},
        {"label": "persist_2",  "daily_confirm": True, "min_signal_bars": 2},
        {"label": "persist_3",  "daily_confirm": True, "min_signal_bars": 3},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AB: daily_confirm + signal persistence gate")
    print(f"# Only enter after N consecutive bars agree on direction")
    print(f"{'#'*60}")
    return experiment_combo(tickers, combos)


def experiment_vwap_gate(tickers: list) -> dict:
    """Experiment AA: daily_confirm + VWAP alignment gate.
    Only enter long when price > VWAP (intraday bullish) and short when price < VWAP.
    VWAP is the institutional fair-value anchor; trading with VWAP reduces adverse fills."""
    combos = [
        {"label": "dc_base",    "daily_confirm": True, "vwap_align": False},
        {"label": "dc_vwap",    "daily_confirm": True, "vwap_align": True},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AA: daily_confirm + VWAP alignment gate")
    print(f"# long only above VWAP, short only below VWAP")
    print(f"{'#'*60}")
    return experiment_combo(tickers, combos)


def experiment_pred_horizon(tickers: list) -> dict:
    """Experiment Z: daily_confirm + varied prediction horizon.
    Currently trains to predict 6 bars (6h) ahead. Try 3h and 12h horizons
    to see if a different training target improves classification quality."""
    combos = [
        {"label": "horizon_3",  "daily_confirm": True, "target_bars_override": 3},
        {"label": "horizon_6",  "daily_confirm": True, "target_bars_override": 6},
        {"label": "horizon_9",  "daily_confirm": True, "target_bars_override": 9},
        {"label": "horizon_12", "daily_confirm": True, "target_bars_override": 12},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT Z: daily_confirm + prediction horizon variation")
    print(f"# horizon = bars ahead used for label generation (not hold time)")
    print(f"{'#'*60}")
    return experiment_combo(tickers, combos)


def experiment_label_threshold(tickers: list) -> dict:
    """Experiment Y: daily_confirm + tighter label percentile thresholds.
    Current 33/67 means top+bottom thirds are labeled UP/DOWN.
    Tighter splits (25/75, 20/80) make labels represent more decisive moves,
    potentially improving signal quality at the cost of training data."""
    combos = [
        {"label": "lbl_33_67",   "daily_confirm": True, "label_pct_low": 0.33, "label_pct_high": 0.67},
        {"label": "lbl_30_70",   "daily_confirm": True, "label_pct_low": 0.30, "label_pct_high": 0.70},
        {"label": "lbl_25_75",   "daily_confirm": True, "label_pct_low": 0.25, "label_pct_high": 0.75},
        {"label": "lbl_20_80",   "daily_confirm": True, "label_pct_low": 0.20, "label_pct_high": 0.80},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT Y: daily_confirm + label threshold variation")
    print(f"# Tighter splits = rarer but more decisive UP/DOWN labels")
    print(f"{'#'*60}")
    return experiment_combo(tickers, combos)


def experiment_vol_confirm(tickers: list) -> dict:
    """Experiment X: daily_confirm + volume confirmation gate.
    Only enter when the entry bar's volume >= N× the 20-bar average.
    High-volume bars signal conviction; low-volume signals are noise."""
    combos = [
        {"label": "dc_base",        "daily_confirm": True, "min_vol_ratio": 0.0},
        {"label": "vol_1.2x",       "daily_confirm": True, "min_vol_ratio": 1.2},
        {"label": "vol_1.5x",       "daily_confirm": True, "min_vol_ratio": 1.5},
        {"label": "vol_2.0x",       "daily_confirm": True, "min_vol_ratio": 2.0},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT X: daily_confirm + volume confirmation gate")
    print(f"# Only enter on bars with volume >= min_vol_ratio x avg")
    print(f"{'#'*60}")
    return experiment_combo(tickers, combos)


def experiment_atr_stops(tickers: list) -> dict:
    """Experiment W: daily_confirm + ATR-based dynamic stops.
    Instead of fixed 2%/1% TP/SL, scale them by the entry-bar ATR_pct.
    Tests several TP:SL multiplier ratios to find the best R:R for each
    ticker's volatility regime."""
    # Combos: (atr_tp_mult, atr_sl_mult, label)
    # ATR_pct for 1h bars is typically 0.3-0.8%; so mult=3 → ~1-2% TP
    combos = [
        {"label": "dc_base",        "daily_confirm": True, "atr_tp_mult": 0.0, "atr_sl_mult": 0.0},
        {"label": "atr_2x1",        "daily_confirm": True, "atr_tp_mult": 2.0, "atr_sl_mult": 1.0},
        {"label": "atr_3x1",        "daily_confirm": True, "atr_tp_mult": 3.0, "atr_sl_mult": 1.0},
        {"label": "atr_3x1.5",      "daily_confirm": True, "atr_tp_mult": 3.0, "atr_sl_mult": 1.5},
        {"label": "atr_4x1",        "daily_confirm": True, "atr_tp_mult": 4.0, "atr_sl_mult": 1.0},
        {"label": "atr_4x2",        "daily_confirm": True, "atr_tp_mult": 4.0, "atr_sl_mult": 2.0},
        {"label": "atr_5x2",        "daily_confirm": True, "atr_tp_mult": 5.0, "atr_sl_mult": 2.0},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT W: daily_confirm + ATR-scaled dynamic stops")
    print(f"# TP = atr_tp_mult x ATR_pct,  SL = atr_sl_mult x ATR_pct")
    print(f"{'#'*60}")
    return experiment_combo(tickers, combos)


def experiment_weekly_features(tickers: list) -> dict:
    """Experiment V: daily_confirm + weekly trend features (w_rsi_14, w_macd_hist,
    w_price_vs_sma20, w_trend_score, w_mom_4).  Tests whether adding a weekly
    timeframe context layer on top of the existing 14 daily features and 63
    hourly features improves on the +1.1% daily_confirm baseline."""
    combos = [
        {"label": "dc_base",         "daily_confirm": True},
        {"label": "dc_weekly",       "daily_confirm": True},  # weekly features auto-active once in FEATURE_COLS
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT V: daily_confirm + weekly trend features")
    print(f"# w_rsi_14, w_macd_hist, w_price_vs_sma20, w_trend_score, w_mom_4")
    print(f"{'#'*60}")
    # Both combos use daily_confirm and the same run_wf path; since
    # weekly features are always computed when daily data is merged,
    # they are automatically present in the feature matrix.
    # We run the single daily_confirm scenario to measure the net effect.
    rows = []
    for t in sorted(tickers):
        r = run_wf(t, daily_confirm=True)
        rows.append(r)

    valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
    avg_cagr   = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0.0
    avg_win    = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0.0
    avg_pf     = sum(r["profit_factor"] for r in valid if r["profit_factor"] < 100) / max(1, len(valid))
    avg_dd     = sum(r["max_drawdown_pct"] for r in valid) / len(valid) if valid else 0.0
    above10    = sum(1 for r in valid if r["cagr_pct"] >= 10)
    above15    = sum(1 for r in valid if r["cagr_pct"] >= 15)

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT V: daily_confirm + weekly features")
    print(f"{'='*60}")
    print(f"  {'TICKER':<8} {'CAGR':>6}  {'WIN%':>6}  {'PF':>5}  {'DD%':>5}  {'TRADES':>6}")
    for r in sorted(valid, key=lambda x: -x["cagr_pct"]):
        print(f"  {r['ticker']:<8} {r['cagr_pct']:>+5.1f}%  {r['win_rate_pct']:>5.1f}%  "
              f"{r['profit_factor']:>5.2f}  {r['max_drawdown_pct']:>+5.1f}%  {r['total_trades']:>6}")
    print(f"\n  Avg CAGR: {avg_cagr:+.1f}%  Win: {avg_win:.1f}%  PF: {avg_pf:.2f}  DD: {avg_dd:.1f}%")
    print(f"  Tickers >=10% CAGR: {above10}  >=15%: {above15}")
    target = "ABOVE TARGET" if avg_cagr >= 1.5 else "BELOW TARGET"
    print(f"\n    STATUS: {target} - {avg_cagr:+.1f}% average CAGR")

    return {
        "rows": rows,
        "summary": {
            "avg_cagr": round(avg_cagr, 2),
            "avg_win_rate": round(avg_win, 2),
            "avg_profit_factor": round(avg_pf, 2),
            "avg_drawdown": round(avg_dd, 2),
            "above10pct": above10,
            "above15pct": above15,
        }
    }


def experiment_earnings_features(tickers: list) -> dict:
    """Experiment AF: daily_confirm + earnings proximity features.
    Adds days_to_earnings, days_since_earnings, near_earnings as model features.
    Hypothesis: signals near earnings are noisier; the model should learn to
    reduce conviction near high-uncertainty events."""
    combos = [
        {"label": "dc_base",        "daily_confirm": True, "use_earnings_features": False},
        {"label": "dc_earnings",    "daily_confirm": True, "use_earnings_features": True},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AF: daily_confirm + earnings proximity features")
    print(f"# days_to_earnings, days_since_earnings, near_earnings")
    print(f"{'#'*60}")
    return experiment_combo(tickers, combos)


def experiment_spy_context(tickers: list) -> dict:
    """Experiment AE: daily_confirm + SPY cross-market features.
    Adds spy_rsi_14, spy_macd_hist, spy_bb_pct, spy_price_vs_sma20,
    spy_volume_ratio as model features. Gives XGBoost context about whether
    the market backdrop is supportive or hostile for the trade signal."""
    combos = [
        {"label": "dc_base",        "daily_confirm": True, "use_spy_context": False},
        {"label": "dc_spy_context", "daily_confirm": True, "use_spy_context": True},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AE: daily_confirm + SPY cross-market features")
    print(f"# spy_rsi_14, spy_macd_hist, spy_bb_pct, spy_price_vs_sma20, spy_volume_ratio")
    print(f"{'#'*60}")
    return experiment_combo(tickers, combos)


def experiment_position_sizing(tickers: list) -> dict:
    """Experiment AD: daily_confirm + varied position sizing.
    Currently max_position_pct=0.1 means each trade risks only 5-7% of capital
    (confidence × 0.1). Increasing this directly amplifies the edge.
    Tests 0.1 (baseline), 0.25, 0.5, and 1.0."""
    combos = [
        {"label": "pos_10pct",  "daily_confirm": True, "max_position_pct": 0.10},
        {"label": "pos_25pct",  "daily_confirm": True, "max_position_pct": 0.25},
        {"label": "pos_50pct",  "daily_confirm": True, "max_position_pct": 0.50},
        {"label": "pos_100pct", "daily_confirm": True, "max_position_pct": 1.00},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AD: Position sizing sweep (daily_confirm)")
    print(f"# Varies max_position_pct — position = confidence x max_pct")
    print(f"{'#'*60}")
    return experiment_combo(tickers, combos)


def experiment_best_combos(tickers: list) -> dict:
    """Experiment AG: Combining the best confirmed improvements.
    Tests dc + max_hold_6 + long_only with position sizing sweep.
    max_hold=6 (best from Exp N), dc (best filter), long_only (slight improvement),
    position sizing (critical amplifier — confidence × max_pct)."""
    combos = [
        {"label": "dc_base",              "daily_confirm": True},
        {"label": "dc_hold6",             "daily_confirm": True, "max_hold_bars": 6},
        {"label": "dc_hold6_long",        "daily_confirm": True, "max_hold_bars": 6, "allow_short": False},
        {"label": "dc_long",              "daily_confirm": True, "allow_short": False},
        {"label": "dc_hold6_pos25",       "daily_confirm": True, "max_hold_bars": 6, "max_position_pct": 0.25},
        {"label": "dc_hold6_pos50",       "daily_confirm": True, "max_hold_bars": 6, "max_position_pct": 0.50},
        {"label": "dc_hold6_pos100",      "daily_confirm": True, "max_hold_bars": 6, "max_position_pct": 1.00},
        {"label": "dc_hold6_long_pos25",  "daily_confirm": True, "max_hold_bars": 6, "allow_short": False, "max_position_pct": 0.25},
        {"label": "dc_hold6_long_pos50",  "daily_confirm": True, "max_hold_bars": 6, "allow_short": False, "max_position_pct": 0.50},
        {"label": "dc_hold6_long_pos100", "daily_confirm": True, "max_hold_bars": 6, "allow_short": False, "max_position_pct": 1.00},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AG: Best-combo sweep")
    print(f"# dc + max_hold6 + long_only + position sizing (10/25/50/100%)")
    print(f"{'#'*60}")
    return experiment_combo(tickers, combos)


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def load_results() -> dict:
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(key: str, data, results: dict):
    # Reload to pick up any keys written by parallel processes
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            on_disk = json.load(f)
        for k, v in on_disk.items():
            if k not in results:
                results[k] = v
    results[key] = {
        "timestamp": datetime.now().isoformat(),
        "data": data if isinstance(data, (dict, list)) else str(data),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE} (key: {key})")


def experiment_universe_scan(tickers: list = None, combo: dict = None,
                             max_position_pct: float = None) -> list:
    """
    Experiment AI: Run the best combo on every trained ticker and rank by CAGR.
    Finds hidden gems in the full 41-ticker universe.
    """
    all_tickers = tickers or sorted(get_trained_tickers())
    base_combo = combo or {"daily_confirm": True, "allow_short": False, "max_hold_bars": 6}
    label = "dc_hold6_long"

    orig_pos = CONFIG.bot.max_position_pct
    if max_position_pct is not None:
        CONFIG.bot.max_position_pct = max_position_pct
    pos_label = f"{int(CONFIG.bot.max_position_pct * 100)}pct"

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AI: Full universe scan")
    print(f"# Combo: {label}  |  Position: {pos_label}")
    print(f"# Running {len(all_tickers)} tickers — finding the best portfolio")
    print(f"{'#'*60}")

    rows = []
    try:
        for t in sorted(all_tickers):
            r = run_wf(t, **{k: v for k, v in base_combo.items()})
            rows.append(r)
            if not r.get("error"):
                mark = " ***" if r["cagr_pct"] >= 10 else (" **" if r["cagr_pct"] >= 5 else "")
                print(f"  {t:<6} CAGR={r['cagr_pct']:>+6.1f}%  win={r['win_rate_pct']:.1f}%  "
                      f"trades={r['total_trades']:>4}  PF={r['profit_factor']:.2f}{mark}")
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 10]
    valid_sorted = sorted(valid, key=lambda x: -x["cagr_pct"])

    print(f"\n  --- TOP 15 TICKERS ({label} @ {pos_label}) ---")
    for r in valid_sorted[:15]:
        print(f"  {r['ticker']:<6} CAGR={r['cagr_pct']:>+6.1f}%  win={r['win_rate_pct']:.1f}%  "
              f"DD={r['max_drawdown_pct']:.1f}%  trades={r['total_trades']}")

    avg_all = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
    top5 = valid_sorted[:5]
    avg_top5 = sum(r["cagr_pct"] for r in top5) / len(top5) if top5 else 0
    print(f"\n  Avg CAGR all ({len(valid)} tickers): {avg_all:+.1f}%")
    print(f"  Avg CAGR top 5:                  {avg_top5:+.1f}%")
    return rows


def experiment_super_portfolio(tickers: list, pos_pcts: list = None,
                                allow_short: bool = False,
                                max_hold_bars: int = 6,
                                min_conf: float = None) -> dict:
    """
    Experiment AJ: Best-combo sweep at escalating position sizes.
    Runs dc_hold6_long at 10/25/50/100% on the given (pre-selected) tickers.
    Goal: find the position size that hits 20% avg CAGR.
    """
    pos_pcts = pos_pcts or [0.10, 0.25, 0.50, 1.00]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AJ: Super portfolio — escalating position size")
    print(f"# Tickers: {', '.join(sorted(tickers))}")
    print(f"# Combo: dc + hold{max_hold_bars} + {'long_only' if not allow_short else 'long+short'}")
    print(f"# Position sizes: {[int(p*100) for p in pos_pcts]}%")
    print(f"{'#'*60}")

    results = {}
    orig_pos = CONFIG.bot.max_position_pct
    target_hit = False

    try:
        for pct in pos_pcts:
            CONFIG.bot.max_position_pct = pct
            label = f"dc_h{max_hold_bars}_{'long' if not allow_short else 'ls'}_{int(pct*100)}pct"

            print(f"\n  --- Position size: {int(pct*100)}% ---")
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, daily_confirm=True, allow_short=allow_short,
                           max_hold_bars=max_hold_bars,
                           conf_threshold_param=min_conf)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"DD={r['max_drawdown_pct']:.1f}%")

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 5]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)

            results[label] = {
                "avg_cagr": round(avg_cagr, 2),
                "above_20pct": above20,
                "above_10pct": above10,
                "n_tickers": len(valid),
            }
            status = ""
            if avg_cagr >= 20:
                status = "  *** TARGET MET ***"
                target_hit = True
            elif avg_cagr >= 15:
                status = "  ** CLOSE **"
            print(f"\n  RESULT @ {int(pct*100)}%: avg_CAGR={avg_cagr:+.1f}%  "
                  f">=10%: {above10}  >=20%: {above20}{status}")

            if target_hit:
                break
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    print(f"\n  Summary (ranked):")
    for lbl, m in sorted(results.items(), key=lambda x: -x[1]["avg_cagr"]):
        marker = " <-- TARGET MET" if m["avg_cagr"] >= 20 else (" <-- CLOSE" if m["avg_cagr"] >= 15 else "")
        print(f"  {lbl:<35} avg_CAGR={m['avg_cagr']:>+6.1f}%{marker}")

    return results


def experiment_hold_sweep(tickers: list, hold_bars_list: list = None) -> dict:
    """Experiment AK: Sweep max_hold_bars on dc+long+100% to find optimal hold duration."""
    hold_bars_list = hold_bars_list or [6, 12, 24, 48]
    base = {"daily_confirm": True, "allow_short": False}

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AK: Hold-bar sweep (dc+long+100%)")
    print(f"# Hold durations: {hold_bars_list} bars | Tickers: {', '.join(sorted(tickers))}")
    print(f"{'#'*60}")

    orig_pos = CONFIG.bot.max_position_pct
    CONFIG.bot.max_position_pct = 1.0
    results = {}

    try:
        for h in hold_bars_list:
            label = f"hold{h}"
            print(f"\n  --- max_hold_bars={h} ({h}h hold) ---")
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, max_hold_bars=h, **base)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"trades={r['total_trades']:>4}  DD={r['max_drawdown_pct']:.1f}%")

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 5]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            avg_wr = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0
            avg_dd = sum(r["max_drawdown_pct"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)
            results[label] = {
                "avg_cagr": round(avg_cagr, 2),
                "avg_win_rate": round(avg_wr, 2),
                "avg_dd": round(avg_dd, 2),
                "above_10pct": above10,
                "above_20pct": above20,
                "n_valid": len(valid),
                "per_ticker": {r["ticker"]: {"cagr": r["cagr_pct"], "win": r["win_rate_pct"],
                                             "trades": r["total_trades"], "dd": r["max_drawdown_pct"]}
                               for r in valid},
            }
            marker = " *** TARGET" if avg_cagr >= 20 else (" ** CLOSE" if avg_cagr >= 15 else "")
            print(f"\n  RESULT hold{h}: avg_CAGR={avg_cagr:+.1f}%  wr={avg_wr:.1f}%  "
                  f"DD={avg_dd:.1f}%  >=10%:{above10}  >=20%:{above20}{marker}")
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    print(f"\n  Summary (hold-bar sweep, dc+long+100%):")
    for lbl, m in sorted(results.items(), key=lambda x: -x[1]["avg_cagr"]):
        print(f"  {lbl:<10} avg_CAGR={m['avg_cagr']:>+6.1f}%  wr={m['avg_win_rate']:.1f}%  "
              f"DD={m['avg_dd']:.1f}%  >=10%:{m['above_10pct']}  >=20%:{m['above_20pct']}")
    return results


def experiment_rr_sweep(tickers: list) -> dict:
    """Experiment AL: Risk/reward ratio sweep on dc+hold6+long+100%."""
    rr_configs = [
        ("sl2_tp4",   2.0, 4.0),   # baseline 1:2
        ("sl1_tp4",   1.0, 4.0),   # tighter stop, same TP → 1:4
        ("sl1p5_tp6", 1.5, 6.0),   # 1:4
        ("sl2_tp6",   2.0, 6.0),   # 1:3
        ("sl2_tp8",   2.0, 8.0),   # 1:4
        ("sl1_tp5",   1.0, 5.0),   # 1:5
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AL: Risk/Reward sweep (dc+hold6+long+100%)")
    print(f"# Tickers: {', '.join(sorted(tickers))}")
    print(f"{'#'*60}")

    orig_pos = CONFIG.bot.max_position_pct
    CONFIG.bot.max_position_pct = 1.0
    results = {}

    try:
        for label, sl, tp in rr_configs:
            print(f"\n  --- stop={sl}%  take={tp}%  R:R=1:{tp/sl:.1f} ---")
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, daily_confirm=True, allow_short=False,
                           max_hold_bars=6, stop_loss=sl/100.0, take_profit=tp/100.0)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"trades={r['total_trades']:>4}  DD={r['max_drawdown_pct']:.1f}%")

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 5]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            avg_wr = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0
            avg_pf = sum(r["profit_factor"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)
            results[label] = {
                "stop_pct": sl, "take_pct": tp,
                "avg_cagr": round(avg_cagr, 2),
                "avg_win_rate": round(avg_wr, 2),
                "avg_pf": round(avg_pf, 3),
                "above_10pct": above10,
                "above_20pct": above20,
                "n_valid": len(valid),
            }
            print(f"\n  RESULT {label}: avg_CAGR={avg_cagr:+.1f}%  wr={avg_wr:.1f}%  "
                  f"PF={avg_pf:.2f}  >=10%:{above10}  >=20%:{above20}")
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    print(f"\n  Summary (R:R sweep, dc+hold6+long+100%):")
    for lbl, m in sorted(results.items(), key=lambda x: -x[1]["avg_cagr"]):
        print(f"  {lbl:<14} SL={m['stop_pct']}% TP={m['take_pct']}%  "
              f"avg_CAGR={m['avg_cagr']:>+6.1f}%  wr={m['avg_win_rate']:.1f}%  "
              f"PF={m['avg_pf']:.2f}  >=20%:{m['above_20pct']}")
    return results


def experiment_conf_sweep_dc(tickers: list) -> dict:
    """Experiment AM: Confidence threshold sweep on dc+hold6+long+100%."""
    thresholds = [0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AM: Confidence sweep (dc+hold6+long+100%)")
    print(f"# Thresholds: {thresholds}")
    print(f"# Tickers: {', '.join(sorted(tickers))}")
    print(f"{'#'*60}")

    orig_pos = CONFIG.bot.max_position_pct
    CONFIG.bot.max_position_pct = 1.0
    results = {}

    try:
        for thresh in thresholds:
            label = f"conf{int(thresh*100)}"
            print(f"\n  --- min_confidence={thresh} ---")
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, daily_confirm=True, allow_short=False,
                           max_hold_bars=6, conf_threshold=thresh)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"trades={r['total_trades']:>4}  DD={r['max_drawdown_pct']:.1f}%")

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            avg_wr = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0
            avg_trades = sum(r["total_trades"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)
            results[label] = {
                "threshold": thresh,
                "avg_cagr": round(avg_cagr, 2),
                "avg_win_rate": round(avg_wr, 2),
                "avg_trades": round(avg_trades, 1),
                "above_10pct": above10,
                "above_20pct": above20,
                "n_valid": len(valid),
            }
            print(f"\n  RESULT conf={thresh}: avg_CAGR={avg_cagr:+.1f}%  wr={avg_wr:.1f}%  "
                  f"avg_trades={avg_trades:.0f}  >=10%:{above10}  >=20%:{above20}")
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    print(f"\n  Summary (confidence sweep, dc+hold6+long+100%):")
    for lbl, m in sorted(results.items(), key=lambda x: -x[1]["avg_cagr"]):
        print(f"  conf={m['threshold']}  avg_CAGR={m['avg_cagr']:>+6.1f}%  "
              f"wr={m['avg_win_rate']:.1f}%  avg_trades={m['avg_trades']:.0f}  "
              f">=10%:{m['above_10pct']}  >=20%:{m['above_20pct']}")
    return results


def experiment_reentry(tickers: list) -> dict:
    """Experiment AN: Re-entry after TP hit on dc+hold6+long+100%.
    When a trade hits take-profit and the signal is still UP, immediately re-enter.
    Hypothesis: captures continued trend momentum without waiting for the next bar.
    """
    configs = [
        ("dc_hold6_long_no_reentry",   False),
        ("dc_hold6_long_reentry",      True),
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AN: Re-entry after TP (dc+hold6+long+100%)")
    print(f"# Tickers: {', '.join(sorted(tickers))}")
    print(f"{'#'*60}")

    orig_pos = CONFIG.bot.max_position_pct
    CONFIG.bot.max_position_pct = 1.0
    results = {}

    try:
        for label, reentry in configs:
            print(f"\n  --- reentry_after_tp={reentry} ---")
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, daily_confirm=True, allow_short=False,
                           max_hold_bars=6, reentry_after_tp=reentry)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"trades={r['total_trades']:>4}  DD={r['max_drawdown_pct']:.1f}%")

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 5]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            avg_wr = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0
            avg_trades = sum(r["total_trades"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)
            results[label] = {
                "reentry": reentry,
                "avg_cagr": round(avg_cagr, 2),
                "avg_win_rate": round(avg_wr, 2),
                "avg_trades": round(avg_trades, 1),
                "above_10pct": above10,
                "above_20pct": above20,
                "n_valid": len(valid),
                "per_ticker": {r["ticker"]: {"cagr": r["cagr_pct"], "win": r["win_rate_pct"],
                                             "trades": r["total_trades"]}
                               for r in valid},
            }
            print(f"\n  RESULT reentry={reentry}: avg_CAGR={avg_cagr:+.1f}%  "
                  f"wr={avg_wr:.1f}%  avg_trades={avg_trades:.0f}  "
                  f">=10%:{above10}  >=20%:{above20}")
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    no_re = results.get("dc_hold6_long_no_reentry", {})
    with_re = results.get("dc_hold6_long_reentry", {})
    delta = with_re.get("avg_cagr", 0) - no_re.get("avg_cagr", 0)
    print(f"\n  Re-entry delta: {delta:+.1f}% avg CAGR "
          f"({'IMPROVEMENT' if delta > 0 else 'WORSE'})")
    return results


def experiment_regime_gate(tickers: list) -> dict:
    """Experiment AQ: Market regime gate on dc+hold6+long+100%.
    Hard-blocks entries when VIX > 25 (d_vix_regime >= 2.0) or when the daily
    trend has completely broken down (d_trend_align < 1). Tests whether avoiding
    hostile macro environments improves CAGR and reduces drawdown."""
    combos = [
        {"label": "dc_no_regime",    "daily_confirm": True, "allow_short": False, "max_hold_bars": 6, "regime_gate": False},
        {"label": "dc_regime_gate",  "daily_confirm": True, "allow_short": False, "max_hold_bars": 6, "regime_gate": True},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AQ: Market regime gate (VIX + trend filter)")
    print(f"# dc+hold6+long+100%% — blocks entries when VIX>25 or trend broken")
    print(f"{'#'*60}")

    orig_pos = CONFIG.bot.max_position_pct
    CONFIG.bot.max_position_pct = 1.0
    results = {}

    try:
        for combo in combos:
            label = combo.pop("label")
            print(f"\n  Testing {label}...")
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, **combo)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"trades={r['total_trades']:>4}  DD={r['max_drawdown_pct']:.1f}%")

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            avg_wr = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0
            avg_dd = sum(r["max_drawdown_pct"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)
            results[label] = {
                "avg_cagr": round(avg_cagr, 2), "avg_win_rate": round(avg_wr, 2),
                "avg_dd": round(avg_dd, 2), "above_10pct": above10, "above_20pct": above20,
                "n_valid": len(valid),
            }
            print(f"\n  RESULT {label}: avg_CAGR={avg_cagr:+.1f}%  wr={avg_wr:.1f}%  "
                  f"DD={avg_dd:.1f}%  >=10%:{above10}  >=20%:{above20}")
            combo["label"] = label
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    if len(results) == 2:
        base = results.get("dc_no_regime", {}).get("avg_cagr", 0)
        gated = results.get("dc_regime_gate", {}).get("avg_cagr", 0)
        print(f"\n  Regime gate delta: {gated - base:+.1f}% avg CAGR  "
              f"DD: {results.get('dc_regime_gate',{}).get('avg_dd',0):.1f}% vs "
              f"{results.get('dc_no_regime',{}).get('avg_dd',0):.1f}%")
    return results


def experiment_breakeven_stop(tickers: list) -> dict:
    """Experiment AR: Break-even ratchet stop on dc+hold6+long+100%.
    Once a trade gains >= breakeven_pct from entry, the stop-loss moves to entry price.
    This converts a potential loser back into a risk-free trade once it's profitable.
    Tests values: 0% (disabled), 0.5%, 1.0%, 1.5%."""
    be_values = [0.0, 0.005, 0.01, 0.015]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AR: Break-even ratchet stop (dc+hold6+long+100%)")
    print(f"# breakeven_pct: gain needed to lock SL at entry price")
    print(f"{'#'*60}")

    orig_pos = CONFIG.bot.max_position_pct
    CONFIG.bot.max_position_pct = 1.0
    results = {}

    try:
        for be in be_values:
            label = "be_off" if be == 0.0 else f"be_{int(be*1000)}bp"
            print(f"\n  --- breakeven_pct={be:.3f} ({label}) ---")
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, daily_confirm=True, allow_short=False,
                           max_hold_bars=6, breakeven_pct=be)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"trades={r['total_trades']:>4}  DD={r['max_drawdown_pct']:.1f}%")

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            avg_wr = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0
            avg_dd = sum(r["max_drawdown_pct"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)
            results[label] = {
                "breakeven_pct": be,
                "avg_cagr": round(avg_cagr, 2), "avg_win_rate": round(avg_wr, 2),
                "avg_dd": round(avg_dd, 2), "above_10pct": above10, "above_20pct": above20,
                "n_valid": len(valid),
            }
            print(f"\n  RESULT {label}: avg_CAGR={avg_cagr:+.1f}%  wr={avg_wr:.1f}%  "
                  f"DD={avg_dd:.1f}%  >=10%:{above10}  >=20%:{above20}")
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    print(f"\n  Break-even stop summary (dc+hold6+long+100%):")
    for lbl, m in sorted(results.items(), key=lambda x: -x[1]["avg_cagr"]):
        marker = " <-- best" if m["avg_cagr"] == max(v["avg_cagr"] for v in results.values()) else ""
        print(f"  {lbl:<12} be={m['breakeven_pct']*100:.1f}%  avg_CAGR={m['avg_cagr']:>+6.1f}%  "
              f"wr={m['avg_win_rate']:.1f}%  DD={m['avg_dd']:.1f}%{marker}")
    return results


def experiment_first_touch_dc(tickers: list) -> dict:
    """Experiment AS: First-touch target + daily_confirm + hold6 + long + 100%.
    Combines the best training signal (first-touch: predict TP hit before SL)
    with the best trading setup. First-touch labels map directly to our trade exits,
    unlike percentile-rank labels which only measure relative return."""
    combos = [
        {"label": "percentile_dc",   "daily_confirm": True, "allow_short": False, "max_hold_bars": 6},
        # first_touch is set via CONFIG.ml.use_first_touch_target inside run_wf
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AS: First-touch target + dc+hold6+long+100%")
    print(f"# Compares percentile-rank label vs first-touch TP/SL label")
    print(f"{'#'*60}")

    orig_pos = CONFIG.bot.max_position_pct
    orig_ft = CONFIG.ml.use_first_touch_target
    CONFIG.bot.max_position_pct = 1.0
    results = {}

    try:
        for use_ft in [False, True]:
            CONFIG.ml.use_first_touch_target = use_ft
            label = "first_touch_dc" if use_ft else "percentile_dc"
            print(f"\n  --- use_first_touch_target={use_ft} ---")
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, daily_confirm=True, allow_short=False, max_hold_bars=6)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"trades={r['total_trades']:>4}  DD={r['max_drawdown_pct']:.1f}%")

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            avg_wr = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0
            avg_dd = sum(r["max_drawdown_pct"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)
            results[label] = {
                "first_touch": use_ft,
                "avg_cagr": round(avg_cagr, 2), "avg_win_rate": round(avg_wr, 2),
                "avg_dd": round(avg_dd, 2), "above_10pct": above10, "above_20pct": above20,
                "n_valid": len(valid),
            }
            print(f"\n  RESULT {label}: avg_CAGR={avg_cagr:+.1f}%  wr={avg_wr:.1f}%  "
                  f"DD={avg_dd:.1f}%  >=10%:{above10}  >=20%:{above20}")
    finally:
        CONFIG.bot.max_position_pct = orig_pos
        CONFIG.ml.use_first_touch_target = orig_ft

    base = results.get("percentile_dc", {}).get("avg_cagr", 0)
    ft = results.get("first_touch_dc", {}).get("avg_cagr", 0)
    print(f"\n  First-touch delta: {ft - base:+.1f}% avg CAGR  "
          f"({'IMPROVEMENT' if ft > base else 'WORSE'})")
    return results


def experiment_partial_tp(tickers: list) -> dict:
    """Experiment AT: Partial take-profit sweep on dc+hold6+long+100%.
    Takes partial_tp_ratio (default 50%) off at partial_tp_pct gain,
    then lets the remainder run to full TP. Tests 1%, 1.5%, 2% partial TP levels
    combined with a break-even ratchet after partial exit."""
    configs = [
        ("no_partial",        0.0,   0.5),
        ("partial_1pct",      0.01,  0.5),
        ("partial_1p5pct",    0.015, 0.5),
        ("partial_2pct",      0.02,  0.5),
        ("partial_1pct_be",   0.01,  0.5),   # partial + breakeven (combined below)
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AT: Partial take-profit sweep (dc+hold6+long+100%)")
    print(f"# Takes 50%% of position at partial_tp_pct, holds rest to full TP")
    print(f"{'#'*60}")

    orig_pos = CONFIG.bot.max_position_pct
    CONFIG.bot.max_position_pct = 1.0
    results = {}

    try:
        for label, ptp, ratio in configs:
            # For the combined partial+be version, also set breakeven
            be = 0.01 if label == "partial_1pct_be" else 0.0
            print(f"\n  --- partial_tp={ptp*100:.1f}% ratio={ratio} be={be*100:.1f}% ---")
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, daily_confirm=True, allow_short=False, max_hold_bars=6,
                           partial_tp_pct=ptp, partial_tp_ratio=ratio, breakeven_pct=be)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"trades={r['total_trades']:>4}  DD={r['max_drawdown_pct']:.1f}%")

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            avg_wr = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0
            avg_dd = sum(r["max_drawdown_pct"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)
            results[label] = {
                "partial_tp_pct": ptp, "breakeven_pct": be,
                "avg_cagr": round(avg_cagr, 2), "avg_win_rate": round(avg_wr, 2),
                "avg_dd": round(avg_dd, 2), "above_10pct": above10, "above_20pct": above20,
                "n_valid": len(valid),
            }
            print(f"\n  RESULT {label}: avg_CAGR={avg_cagr:+.1f}%  wr={avg_wr:.1f}%  "
                  f"DD={avg_dd:.1f}%  >=10%:{above10}  >=20%:{above20}")
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    print(f"\n  Partial TP summary (dc+hold6+long+100%):")
    for lbl, m in sorted(results.items(), key=lambda x: -x[1]["avg_cagr"]):
        marker = " <-- best" if m["avg_cagr"] == max(v["avg_cagr"] for v in results.values()) else ""
        print(f"  {lbl:<20} ptp={m['partial_tp_pct']*100:.1f}%  avg_CAGR={m['avg_cagr']:>+6.1f}%  "
              f"wr={m['avg_win_rate']:.1f}%  DD={m['avg_dd']:.1f}%{marker}")
    return results


def experiment_vol_targeting(tickers: list) -> dict:
    """Experiment AU: Volatility-targeted position sizing on dc+hold6+long+100%.
    Scales position inversely with current ATR_pct to maintain constant dollar
    risk per trade (risk parity). Lower ATR = bigger position (more capital
    deployed when market is calm); higher ATR = smaller position (protect against
    whipsaws in volatile conditions). Tests target ATR values: 0 (disabled),
    0.003, 0.005, 0.007, 0.01."""
    targets = [0.0, 0.003, 0.005, 0.007, 0.010]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AU: Volatility-targeted position sizing")
    print(f"# dc+hold6+long+100%% base | scale by target_atr/current_atr")
    print(f"# Typical 1h ATR_pct ~0.3-0.8%; target_atr ~0.5% = neutral")
    print(f"{'#'*60}")

    orig_pos = CONFIG.bot.max_position_pct
    CONFIG.bot.max_position_pct = 1.0
    results = {}

    try:
        for tgt in targets:
            label = "no_vol_target" if tgt == 0.0 else f"vt_{int(tgt*1000)}bp"
            print(f"\n  --- vol_target_atr={tgt:.3f} ({label}) ---")
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, daily_confirm=True, allow_short=False,
                           max_hold_bars=6, vol_target_atr=tgt)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"trades={r['total_trades']:>4}  DD={r['max_drawdown_pct']:.1f}%")

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            avg_wr = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0
            avg_dd = sum(r["max_drawdown_pct"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)
            results[label] = {
                "vol_target_atr": tgt,
                "avg_cagr": round(avg_cagr, 2), "avg_win_rate": round(avg_wr, 2),
                "avg_dd": round(avg_dd, 2), "above_10pct": above10, "above_20pct": above20,
                "n_valid": len(valid),
            }
            print(f"\n  RESULT {label}: avg_CAGR={avg_cagr:+.1f}%  wr={avg_wr:.1f}%  "
                  f"DD={avg_dd:.1f}%  >=10%:{above10}  >=20%:{above20}")
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    print(f"\n  Volatility targeting summary (dc+hold6+long+100%):")
    for lbl, m in sorted(results.items(), key=lambda x: -x[1]["avg_cagr"]):
        marker = " <-- best" if m["avg_cagr"] == max(v["avg_cagr"] for v in results.values()) else ""
        print(f"  {lbl:<18} vt={m['vol_target_atr']*100:.2f}%%  avg_CAGR={m['avg_cagr']:>+6.1f}%  "
              f"wr={m['avg_win_rate']:.1f}%  DD={m['avg_dd']:.1f}%{marker}")
    return results


def experiment_best_stack(tickers: list) -> dict:
    """Experiment AV: Stacked best-of-everything — combines all confirmed improvements.
    Tests the cumulative effect of: regime gate + break-even ratchet + partial TP +
    vol targeting. Each is added incrementally to see which combinations stack well."""
    combos = [
        # Baseline
        {"label": "baseline_dc_h6_long",
         "daily_confirm": True, "allow_short": False, "max_hold_bars": 6},
        # Individual best improvements
        {"label": "+ regime_gate",
         "daily_confirm": True, "allow_short": False, "max_hold_bars": 6,
         "regime_gate": True},
        {"label": "+ breakeven_1pct",
         "daily_confirm": True, "allow_short": False, "max_hold_bars": 6,
         "breakeven_pct": 0.01},
        {"label": "+ partial_tp_2pct",
         "daily_confirm": True, "allow_short": False, "max_hold_bars": 6,
         "partial_tp_pct": 0.02, "partial_tp_ratio": 0.5},
        {"label": "+ vol_target_5bp",
         "daily_confirm": True, "allow_short": False, "max_hold_bars": 6,
         "vol_target_atr": 0.005},
        # Pairs
        {"label": "regime+breakeven",
         "daily_confirm": True, "allow_short": False, "max_hold_bars": 6,
         "regime_gate": True, "breakeven_pct": 0.01},
        {"label": "breakeven+partial",
         "daily_confirm": True, "allow_short": False, "max_hold_bars": 6,
         "breakeven_pct": 0.01, "partial_tp_pct": 0.02, "partial_tp_ratio": 0.5},
        {"label": "regime+partial",
         "daily_confirm": True, "allow_short": False, "max_hold_bars": 6,
         "regime_gate": True, "partial_tp_pct": 0.02, "partial_tp_ratio": 0.5},
        # Triple stack
        {"label": "regime+be+partial",
         "daily_confirm": True, "allow_short": False, "max_hold_bars": 6,
         "regime_gate": True, "breakeven_pct": 0.01,
         "partial_tp_pct": 0.02, "partial_tp_ratio": 0.5},
        # Full stack
        {"label": "FULL_STACK",
         "daily_confirm": True, "allow_short": False, "max_hold_bars": 6,
         "regime_gate": True, "breakeven_pct": 0.01,
         "partial_tp_pct": 0.02, "partial_tp_ratio": 0.5,
         "vol_target_atr": 0.005},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AV: Best-stack combination test")
    print(f"# dc+hold6+long+100%% + incremental improvements")
    print(f"# {', '.join(sorted(tickers))}")
    print(f"{'#'*60}")

    orig_pos = CONFIG.bot.max_position_pct
    CONFIG.bot.max_position_pct = 1.0
    results = {}

    try:
        for combo in combos:
            label = combo.pop("label")
            print(f"\n  Testing [{label}]...")
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, **combo)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"DD={r['max_drawdown_pct']:.1f}%")

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            avg_wr = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0
            avg_dd = sum(r["max_drawdown_pct"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)
            results[label] = {
                "avg_cagr": round(avg_cagr, 2), "avg_win_rate": round(avg_wr, 2),
                "avg_dd": round(avg_dd, 2), "above_10pct": above10, "above_20pct": above20,
                "n_valid": len(valid), "params": {**combo},
            }
            marker = " *** TARGET" if avg_cagr >= 25 else (" ** CLOSE" if avg_cagr >= 20 else "")
            print(f"\n  RESULT [{label}]: avg_CAGR={avg_cagr:+.1f}%  wr={avg_wr:.1f}%  "
                  f"DD={avg_dd:.1f}%  >=20%:{above20}{marker}")
            combo["label"] = label
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    print(f"\n{'='*70}")
    print(f"  BEST STACK SUMMARY (ranked by avg CAGR):")
    print(f"  {'COMBO':<30} {'CAGR':>7}  {'WIN%':>6}  {'DD%':>7}  {'>=20%':>6}")
    print(f"{'='*70}")
    best_cagr = max(v["avg_cagr"] for v in results.values()) if results else 0
    for lbl, m in sorted(results.items(), key=lambda x: -x[1]["avg_cagr"]):
        marker = " <-- BEST" if m["avg_cagr"] == best_cagr else ""
        print(f"  {lbl:<30} {m['avg_cagr']:>+6.1f}%  {m['avg_win_rate']:>5.1f}%  "
              f"{m['avg_dd']:>6.1f}%  {m['above_20pct']:>6}{marker}")

    baseline = results.get("baseline_dc_h6_long", {}).get("avg_cagr", 0)
    champion = max(results.values(), key=lambda x: x["avg_cagr"]) if results else {}
    lift = champion.get("avg_cagr", 0) - baseline
    print(f"\n  Baseline:  {baseline:+.1f}%  -->  Best stack: {champion.get('avg_cagr', 0):+.1f}%  (lift: {lift:+.1f}%)")
    return results


def experiment_daily_conf_gate(tickers: list) -> dict:
    """Experiment AW: Daily model confidence gate on dc+hold6+long+100%.
    Requires the daily confirmation model to have confidence >= threshold.
    A higher threshold means only high-conviction daily signals pass through,
    trading fewer but higher-quality setups.
    Sweeps: 0 (all signals), 0.38, 0.42, 0.46, 0.50, 0.55.
    """
    thresholds = [0.0, 0.38, 0.42, 0.46, 0.50, 0.55]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AW: Daily model confidence gate sweep")
    print(f"# Base: dc+hold6+long+100%")
    print(f"# Thresholds: {thresholds}")
    print(f"# {', '.join(sorted(tickers))}")
    print(f"{'#'*60}")

    orig_pos = CONFIG.bot.max_position_pct
    CONFIG.bot.max_position_pct = 1.0
    results = {}

    try:
        for thr in thresholds:
            label = f"daily_conf>={thr:.2f}" if thr > 0 else "baseline (no conf gate)"
            print(f"\n  Testing [{label}]...")
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, daily_confirm=True, allow_short=False,
                           max_hold_bars=6, daily_min_conf=thr)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"trades={r['total_trades']}  DD={r['max_drawdown_pct']:.1f}%")

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            avg_wr = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0
            avg_dd = sum(r["max_drawdown_pct"] for r in valid) / len(valid) if valid else 0
            avg_trades = sum(r["total_trades"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            results[label] = {
                "threshold": thr,
                "avg_cagr": round(avg_cagr, 2), "avg_win_rate": round(avg_wr, 2),
                "avg_dd": round(avg_dd, 2), "avg_trades": round(avg_trades, 1),
                "above_20pct": above20, "n_valid": len(valid),
            }
            marker = " *** TARGET" if avg_cagr >= 25 else (" ** CLOSE" if avg_cagr >= 20 else "")
            print(f"\n  RESULT [{label}]: avg_CAGR={avg_cagr:+.1f}%  wr={avg_wr:.1f}%  "
                  f"trades={avg_trades:.0f}  DD={avg_dd:.1f}%{marker}")
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    print(f"\n{'='*70}")
    print(f"  DAILY CONF GATE SUMMARY (ranked by avg CAGR):")
    print(f"  {'THRESHOLD':<30} {'CAGR':>7}  {'WIN%':>6}  {'TRADES':>7}  {'DD%':>7}")
    print(f"{'='*70}")
    best_cagr = max(v["avg_cagr"] for v in results.values()) if results else 0
    for lbl, m in sorted(results.items(), key=lambda x: -x[1]["avg_cagr"]):
        marker = " <-- BEST" if m["avg_cagr"] == best_cagr else ""
        print(f"  {lbl:<30} {m['avg_cagr']:>+6.1f}%  {m['avg_win_rate']:>5.1f}%  "
              f"{m['avg_trades']:>6.0f}  {m['avg_dd']:>6.1f}%{marker}")

    baseline_cagr = results.get("baseline (no conf gate)", {}).get("avg_cagr", 0)
    best_m = max(results.values(), key=lambda x: x["avg_cagr"]) if results else {}
    lift = best_m.get("avg_cagr", 0) - baseline_cagr
    print(f"\n  Baseline: {baseline_cagr:+.1f}%  -->  Best: {best_m.get('avg_cagr', 0):+.1f}%  (lift: {lift:+.1f}%)")
    return results


def experiment_sector_features(tickers: list) -> dict:
    """Experiment AX: Sector relative strength features on dc+hold6+long+100%.
    Adds sector_rs_5, sector_rs_20, sector_rs_slope by comparing each stock
    against its sector ETF (XLK for tech, XLI for industrials, etc.).
    Stocks outperforming their sector historically show momentum continuation.
    Compares: baseline dc vs dc+sector_features.
    """
    combos = [
        {"label": "baseline_dc",
         "daily_confirm": True, "allow_short": False, "max_hold_bars": 6,
         "use_sector_features": False},
        {"label": "dc+sector_features",
         "daily_confirm": True, "allow_short": False, "max_hold_bars": 6,
         "use_sector_features": True},
    ]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AX: Sector relative strength features")
    print(f"# Base: dc+hold6+long+100%")
    print(f"# {', '.join(sorted(tickers))}")
    print(f"{'#'*60}")

    orig_pos = CONFIG.bot.max_position_pct
    CONFIG.bot.max_position_pct = 1.0
    results = {}

    try:
        for combo in combos:
            label = combo.pop("label")
            print(f"\n  Testing [{label}]...")
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, **combo)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"trades={r['total_trades']}  DD={r['max_drawdown_pct']:.1f}%")

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            avg_wr = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0
            avg_dd = sum(r["max_drawdown_pct"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            results[label] = {
                "avg_cagr": round(avg_cagr, 2), "avg_win_rate": round(avg_wr, 2),
                "avg_dd": round(avg_dd, 2), "above_20pct": above20, "n_valid": len(valid),
            }
            marker = " *** TARGET" if avg_cagr >= 25 else (" ** CLOSE" if avg_cagr >= 20 else "")
            print(f"\n  RESULT [{label}]: avg_CAGR={avg_cagr:+.1f}%  wr={avg_wr:.1f}%  "
                  f"DD={avg_dd:.1f}%  >=20%:{above20}{marker}")
            combo["label"] = label
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    baseline = results.get("baseline_dc", {}).get("avg_cagr", 0)
    sector_cagr = results.get("dc+sector_features", {}).get("avg_cagr", 0)
    lift = sector_cagr - baseline
    print(f"\n  Baseline: {baseline:+.1f}%  -->  With sector RS: {sector_cagr:+.1f}%  (lift: {lift:+.1f}%)")
    return results


def experiment_ensemble(tickers: list) -> dict:
    """Experiment AZ: Ensemble of N XGBoost models with different seeds.
    Training multiple models and averaging their probabilities reduces prediction
    variance, filters out noise signals, and improves confidence calibration.
    Tests n_ensemble=1 (baseline), 3, 5.
    """
    ensemble_sizes = [1, 3, 5]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AZ: XGBoost ensemble sweep")
    print(f"# Base: dc+hold6+long+100%")
    print(f"# N ensemble: {ensemble_sizes}")
    print(f"# {', '.join(sorted(tickers))}")
    print(f"{'#'*60}")

    orig_pos = CONFIG.bot.max_position_pct
    CONFIG.bot.max_position_pct = 1.0
    results = {}

    try:
        for n in ensemble_sizes:
            label = f"ensemble_{n}x"
            print(f"\n  Testing [{label}]...")
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, daily_confirm=True, allow_short=False,
                           max_hold_bars=6, n_ensemble=n)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"trades={r['total_trades']}  DD={r['max_drawdown_pct']:.1f}%")

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            avg_wr = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0
            avg_dd = sum(r["max_drawdown_pct"] for r in valid) / len(valid) if valid else 0
            avg_trades = sum(r["total_trades"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            results[label] = {
                "n_ensemble": n,
                "avg_cagr": round(avg_cagr, 2), "avg_win_rate": round(avg_wr, 2),
                "avg_dd": round(avg_dd, 2), "avg_trades": round(avg_trades, 1),
                "above_20pct": above20, "n_valid": len(valid),
            }
            marker = " *** TARGET" if avg_cagr >= 25 else (" ** CLOSE" if avg_cagr >= 20 else "")
            print(f"\n  RESULT [{label}]: avg_CAGR={avg_cagr:+.1f}%  wr={avg_wr:.1f}%  "
                  f"trades={avg_trades:.0f}  DD={avg_dd:.1f}%{marker}")
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    print(f"\n{'='*70}")
    print(f"  ENSEMBLE SUMMARY (ranked by avg CAGR):")
    print(f"  {'CONFIG':<20} {'CAGR':>7}  {'WIN%':>6}  {'TRADES':>7}  {'DD%':>7}")
    print(f"{'='*70}")
    best_cagr = max(v["avg_cagr"] for v in results.values()) if results else 0
    for lbl, m in sorted(results.items(), key=lambda x: -x[1]["avg_cagr"]):
        marker = " <-- BEST" if m["avg_cagr"] == best_cagr else ""
        print(f"  {lbl:<20} {m['avg_cagr']:>+6.1f}%  {m['avg_win_rate']:>5.1f}%  "
              f"{m['avg_trades']:>6.0f}  {m['avg_dd']:>6.1f}%{marker}")

    baseline_cagr = results.get("ensemble_1x", {}).get("avg_cagr", 0)
    best_m = max(results.values(), key=lambda x: x["avg_cagr"]) if results else {}
    lift = best_m.get("avg_cagr", 0) - baseline_cagr
    print(f"\n  Baseline: {baseline_cagr:+.1f}%  -->  Best: {best_m.get('avg_cagr', 0):+.1f}%  (lift: {lift:+.1f}%)")
    return results


def experiment_daily_horizon(tickers: list) -> dict:
    """Experiment AY: Daily model prediction horizon sweep on dc+hold6+long+100%.
    Tests whether predicting 1-day vs 3-day vs 5-day vs 10-day ahead for the daily
    confirmation model changes signal quality. Longer horizons smooth out noise and
    better identify multi-day regimes that support intraday momentum entries.
    """
    horizons = [1, 3, 5, 10]

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AY: Daily model prediction horizon sweep")
    print(f"# Base: dc+hold6+long+100%")
    print(f"# Horizons: {horizons} days")
    print(f"# {', '.join(sorted(tickers))}")
    print(f"{'#'*60}")

    orig_pos = CONFIG.bot.max_position_pct
    CONFIG.bot.max_position_pct = 1.0
    results = {}

    try:
        for n in horizons:
            label = f"daily_{n}d_ahead"
            print(f"\n  Testing [{label}]...")
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, daily_confirm=True, allow_short=False,
                           max_hold_bars=6, daily_n_days_ahead=n)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"trades={r['total_trades']}  DD={r['max_drawdown_pct']:.1f}%")

            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 3]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            avg_wr = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0
            avg_dd = sum(r["max_drawdown_pct"] for r in valid) / len(valid) if valid else 0
            avg_trades = sum(r["total_trades"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            results[label] = {
                "n_days": n,
                "avg_cagr": round(avg_cagr, 2), "avg_win_rate": round(avg_wr, 2),
                "avg_dd": round(avg_dd, 2), "avg_trades": round(avg_trades, 1),
                "above_20pct": above20, "n_valid": len(valid),
            }
            marker = " *** TARGET" if avg_cagr >= 25 else (" ** CLOSE" if avg_cagr >= 20 else "")
            print(f"\n  RESULT [{label}]: avg_CAGR={avg_cagr:+.1f}%  wr={avg_wr:.1f}%  "
                  f"trades={avg_trades:.0f}  DD={avg_dd:.1f}%{marker}")
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    print(f"\n{'='*70}")
    print(f"  DAILY HORIZON SUMMARY (ranked by avg CAGR):")
    print(f"  {'HORIZON':<25} {'CAGR':>7}  {'WIN%':>6}  {'TRADES':>7}  {'DD%':>7}")
    print(f"{'='*70}")
    best_cagr = max(v["avg_cagr"] for v in results.values()) if results else 0
    for lbl, m in sorted(results.items(), key=lambda x: -x[1]["avg_cagr"]):
        marker = " <-- BEST" if m["avg_cagr"] == best_cagr else ""
        print(f"  {lbl:<25} {m['avg_cagr']:>+6.1f}%  {m['avg_win_rate']:>5.1f}%  "
              f"{m['avg_trades']:>6.0f}  {m['avg_dd']:>6.1f}%{marker}")
    return results


def experiment_yearly_backtest(tickers: list, daily_confirm: bool = True,
                               max_position_pct: float = None,
                               allow_short: bool = None,
                               max_hold_bars: int = 0) -> dict:
    """Experiment AH: Calendar-year paper backtest — train on prior years, test each year."""
    from backtest.simulate import yearly_backtest

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT AH: Calendar-year paper backtest")
    print(f"# Trains on all data before each year; tests that year out-of-sample")
    print(f"{'#'*60}")

    orig_pos_pct = CONFIG.bot.max_position_pct
    if max_position_pct is not None:
        CONFIG.bot.max_position_pct = max_position_pct

    all_results = {}
    all_years = set()

    try:
        for ticker in sorted(tickers):
            print(f"\n  [{ticker}] fetching & testing...")
            yr = yearly_backtest(ticker, daily_confirm=daily_confirm,
                                 allow_short=allow_short, max_hold_bars=max_hold_bars)
            all_results[ticker] = yr
            for y, v in yr.items():
                if not v.get("skip"):
                    all_years.add(y)
                    print(f"    {y}: {v['return_pct']:>+6.2f}%  "
                          f"{v['trades']} trades  {v['win_rate']:.1f}% win  "
                          f"({v['train_bars']} train bars)")
                else:
                    print(f"    {y}: skipped — {v['reason']}")
    finally:
        CONFIG.bot.max_position_pct = orig_pos_pct

    if not all_years:
        print("  No valid year results.")
        return all_results

    years = sorted(all_years)
    col_w = 11

    sep = "-" * (8 + col_w * len(years) + 10)
    print(f"\n  {sep}")
    hdr = f"  {'TICKER':<6}"
    for y in years:
        hdr += f"  {y:>{col_w-2}}"
    hdr += f"  {'TOTAL':>8}"
    print(hdr)
    print(f"  {sep}")

    for ticker in sorted(tickers):
        yr = all_results.get(ticker, {})
        row = f"  {ticker:<6}"
        compound = 1.0
        all_valid = True
        for y in years:
            v = yr.get(y)
            if v is None or v.get("skip"):
                row += f"  {'N/A':>{col_w-2}}"
                all_valid = False
            else:
                row += f"  {v['return_pct']:>+{col_w-2}.2f}%"
                compound *= (1 + v["return_pct"] / 100)
        total = (compound - 1) * 100
        row += f"  {total:>+8.2f}%" if all_valid else f"  {'N/A':>8}"
        print(row)

    # Avg row
    print(f"  {sep}")
    avg_row = f"  {'AVG':<6}"
    year_avgs = {}
    for y in years:
        vals = [all_results[t][y]["return_pct"] for t in tickers
                if y in all_results.get(t, {}) and not all_results[t][y].get("skip")]
        avg = sum(vals) / len(vals) if vals else None
        year_avgs[y] = avg
        avg_row += f"  {avg:>+{col_w-2}.2f}%" if avg is not None else f"  {'N/A':>{col_w-2}}"
    compound_avg = 1.0
    for y in years:
        if year_avgs[y] is not None:
            compound_avg *= (1 + year_avgs[y] / 100)
    avg_row += f"  {(compound_avg - 1) * 100:>+8.2f}%"
    print(avg_row)

    return all_results


def experiment_post_optuna_eval(tickers: list, hold_bars: int = 0, min_conf: float = None) -> dict:
    """Experiment BAA: Post-Optuna full portfolio evaluation.
    Runs all tickers with freshly optimized DC params (params_TICKER_dc.json).
    hold_bars=0 means use the per-ticker value from the saved params file.
    min_conf: override confidence threshold (None=use saved params or global default;
              pass 0.0 to disable filtering and see unfiltered max potential).
    Baseline comparison: AM experiment avg_CAGR=21.6% (conf=0.52, 8 tickers).
    Target: avg_CAGR >= 25%.
    """
    conf_label = f"conf={min_conf:.2f}" if min_conf is not None else "conf=saved"
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT BAA: Post-Optuna Full Portfolio Eval")
    print(f"# dc+hold(opt)+long+100%  {conf_label}  |  tickers: {', '.join(sorted(tickers))}")
    print(f"# AM baseline: avg_CAGR=21.6%  |  TARGET: >= 25%")
    print(f"{'#'*60}\n")

    orig_pos = CONFIG.bot.max_position_pct
    CONFIG.bot.max_position_pct = 1.0
    rows = []

    try:
        for t in sorted(tickers):
            # Pass min_conf as conf_threshold so run_wf -> walk_forward_backtest
            # knows this is an explicit override (not the global default), which
            # prevents walk_forward_backtest from then overriding it with saved params.
            r = run_wf(t, daily_confirm=True, allow_short=False, max_hold_bars=hold_bars,
                       conf_threshold=min_conf)
            rows.append(r)
            if r.get("error"):
                print(f"  {t:<6} ERROR: {r['error']}")
            else:
                print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                      f"trades={r['total_trades']:>4}  DD={r['max_drawdown_pct']:.1f}%")
    finally:
        CONFIG.bot.max_position_pct = orig_pos

    valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 5]
    avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
    avg_wr = sum(r["win_rate_pct"] for r in valid) / len(valid) if valid else 0
    avg_dd = sum(r["max_drawdown_pct"] for r in valid) / len(valid) if valid else 0
    above25 = sum(1 for r in valid if r["cagr_pct"] >= 25)
    above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
    above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)

    am_baseline = 21.6
    delta = avg_cagr - am_baseline
    target_met = "*** TARGET MET ***" if avg_cagr >= 25 else f"({25 - avg_cagr:.1f}% short of 25%)"
    print(f"\n{'='*60}")
    print(f"  POST-OPTUNA RESULT: avg_CAGR={avg_cagr:+.1f}%  {target_met}")
    print(f"  vs AM baseline:    avg_CAGR=+21.6%   delta={delta:+.1f}%")
    print(f"  win={avg_wr:.1f}%  DD={avg_dd:.1f}%  >=25%:{above25}  >=20%:{above20}  >=10%:{above10}")
    print(f"{'='*60}")

    return {
        "avg_cagr": round(avg_cagr, 2),
        "avg_win_rate": round(avg_wr, 2),
        "avg_dd": round(avg_dd, 2),
        "above_25pct": above25,
        "above_20pct": above20,
        "above_10pct": above10,
        "am_baseline": am_baseline,
        "delta_vs_baseline": round(delta, 2),
        "n_valid": len(valid),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Research framework for RSI-Model improvement")
    parser.add_argument("experiment", nargs="?", default="baseline",
                        choices=["baseline", "confidence", "d_trend_gate", "daily_confirm",
                                 "risk_reward", "no_flat", "trailing", "long_only", "combo",
                                 "daily_combos", "daily_risk_reward", "conf_dc",
                                 "conviction_dc", "optimize_dc", "first_touch",
                                 "session_filter", "max_hold", "recency_weight", "spy_align",
                                 "orb_baseline", "orb_dc", "dc_combos", "dc_trailing",
                                 "weekly_features", "atr_stops", "vol_confirm",
                                 "label_threshold", "pred_horizon", "vwap_gate",
                                 "signal_persist", "candle_features",
                                 "position_sizing", "spy_context",
                                 "earnings_features", "best_combos", "yearly_backtest",
                                 "universe_scan", "super_portfolio",
                                 "hold_sweep", "rr_sweep", "conf_sweep_dc",
                                 "reentry", "new_features",
                                 "regime_gate", "breakeven_stop", "first_touch_dc",
                                 "partial_tp", "vol_targeting", "best_stack",
                                 "daily_conf_gate", "sector_features", "daily_horizon",
                                 "ensemble", "post_optuna_eval",
                                 "all"],
                        help="Which experiment to run")
    parser.add_argument("--tickers", nargs="+", default=None, help="Subset of tickers")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--trials", type=int, default=60, help="Optuna trials for optimize_dc")
    parser.add_argument("--pos-pcts", nargs="+", type=int, default=None,
                        help="Position sizes in %% for super_portfolio (e.g. 50 100)")
    parser.add_argument("--max-hold-bars", type=int, default=6,
                        help="Max hold bars for super_portfolio")
    parser.add_argument("--max-position-pct", type=float, default=None,
                        help="Position size override (0.1-1.0) for universe_scan")
    parser.add_argument("--min-conf", type=float, default=None,
                        help="Min confidence threshold override (default 0.52)")
    args = parser.parse_args()

    global N_FOLDS
    N_FOLDS = args.folds

    tickers = args.tickers or sorted(get_trained_tickers())
    print(f"\nResearch framework — {len(tickers)} tickers, {N_FOLDS}-fold walk-forward")
    print(f"Config: min_conf={CONFIG.bot.min_confidence}, adx_min={CONFIG.bot.adx_filter_min}, "
          f"train_period={CONFIG.data.train_period}\n")

    results = load_results()

    if args.experiment in ("baseline", "all"):
        rows = run_baseline(tickers)
        save_results("baseline", rows, results)

    if args.experiment in ("confidence", "all"):
        conf_results = experiment_confidence(tickers)
        save_results("experiment_A_confidence", conf_results, results)

    if args.experiment in ("d_trend_gate", "all"):
        trend_results = experiment_d_trend_gate(tickers)
        save_results("experiment_B1_d_trend_gate", trend_results, results)

    if args.experiment in ("daily_confirm", "all"):
        daily_rows = experiment_daily_confirm(tickers)
        save_results("experiment_B2_daily_confirm", daily_rows, results)

    if args.experiment in ("risk_reward", "all"):
        rr_results = experiment_risk_reward(tickers)
        save_results("experiment_C_risk_reward", rr_results, results)

    if args.experiment in ("no_flat", "all"):
        no_flat_rows = experiment_no_flat_exit(tickers)
        save_results("experiment_D_no_flat_exit", no_flat_rows, results)

    if args.experiment in ("trailing", "all"):
        trail_results = experiment_trailing_stop(tickers)
        save_results("experiment_E_trailing_stop", trail_results, results)

    if args.experiment in ("long_only", "all"):
        lo_rows = experiment_long_only(tickers)
        save_results("experiment_F_long_only", lo_rows, results)

    if args.experiment in ("combo", "all"):
        combo_results = experiment_combo(tickers)
        save_results("experiment_G_combo", combo_results, results)

    if args.experiment in ("daily_combos", "all"):
        dc_results = experiment_daily_combos(tickers)
        save_results("experiment_G2_daily_combos", dc_results, results)

    if args.experiment in ("daily_risk_reward", "all"):
        drr_results = experiment_daily_risk_reward(tickers)
        save_results("experiment_I_daily_risk_reward", drr_results, results)

    if args.experiment in ("conf_dc",):
        conf_results = experiment_conf_dc(tickers)
        save_results("experiment_K_conf_dc", conf_results, results)

    if args.experiment in ("conviction_dc",):
        conv_results = experiment_conviction_dc(tickers)
        save_results("experiment_L_conviction_dc", conv_results, results)

    if args.experiment in ("optimize_dc",):
        n_trials = getattr(args, "trials", 60)
        opt_results = experiment_optimize_dc(tickers, n_trials=n_trials)
        save_results("experiment_J_optimize_dc", opt_results, results)

    if args.experiment in ("first_touch", "all"):
        ft_rows = experiment_first_touch(tickers)
        save_results("experiment_H_first_touch", ft_rows, results)

    if args.experiment in ("session_filter",):
        sess_results = experiment_session_filter(tickers)
        save_results("experiment_M_session_filter", sess_results, results)

    if args.experiment in ("max_hold",):
        mh_results = experiment_max_hold(tickers)
        save_results("experiment_N_max_hold", mh_results, results)

    if args.experiment in ("recency_weight",):
        rw_results = experiment_recency_weight(tickers)
        save_results("experiment_R_recency_weight", rw_results, results)

    if args.experiment in ("spy_align",):
        spy_results = experiment_spy_align(tickers)
        save_results("experiment_S_spy_align", spy_results, results)

    if args.experiment in ("orb_baseline",):
        orb_rows = experiment_orb_baseline(tickers)
        save_results("experiment_P_orb_baseline", orb_rows, results)

    if args.experiment in ("orb_dc",):
        orb_dc_rows = experiment_orb_dc(tickers)
        save_results("experiment_Q_orb_dc", orb_dc_rows, results)

    if args.experiment in ("dc_combos",):
        dc_combo_results = experiment_dc_combos(tickers)
        save_results("experiment_T_dc_combos", dc_combo_results, results)

    if args.experiment in ("dc_trailing",):
        dc_trail_results = experiment_dc_trailing(tickers)
        save_results("experiment_U_dc_trailing", dc_trail_results, results)

    if args.experiment in ("weekly_features",):
        wf_results = experiment_weekly_features(tickers)
        save_results("experiment_V_weekly_features", wf_results, results)

    if args.experiment in ("atr_stops",):
        atr_results = experiment_atr_stops(tickers)
        save_results("experiment_W_atr_stops", atr_results, results)

    if args.experiment in ("vol_confirm",):
        vol_results = experiment_vol_confirm(tickers)
        save_results("experiment_X_vol_confirm", vol_results, results)

    if args.experiment in ("label_threshold",):
        lbl_results = experiment_label_threshold(tickers)
        save_results("experiment_Y_label_threshold", lbl_results, results)

    if args.experiment in ("pred_horizon",):
        ph_results = experiment_pred_horizon(tickers)
        save_results("experiment_Z_pred_horizon", ph_results, results)

    if args.experiment in ("vwap_gate",):
        vwap_results = experiment_vwap_gate(tickers)
        save_results("experiment_AA_vwap_gate", vwap_results, results)

    if args.experiment in ("signal_persist",):
        sp_results = experiment_signal_persist(tickers)
        save_results("experiment_AB_signal_persist", sp_results, results)

    if args.experiment in ("candle_features",):
        cf_results = experiment_candle_features(tickers)
        save_results("experiment_AC_candle_features", cf_results, results)

    if args.experiment in ("position_sizing",):
        ps_results = experiment_position_sizing(tickers)
        save_results("experiment_AD_position_sizing", ps_results, results)

    if args.experiment in ("spy_context",):
        sc_results = experiment_spy_context(tickers)
        save_results("experiment_AE_spy_context", sc_results, results)

    if args.experiment in ("earnings_features",):
        ef_results = experiment_earnings_features(tickers)
        save_results("experiment_AF_earnings_features", ef_results, results)

    if args.experiment in ("best_combos",):
        bc_results = experiment_best_combos(tickers)
        save_results("experiment_AG_best_combos", bc_results, results)

    if args.experiment in ("yearly_backtest",):
        yb_results = experiment_yearly_backtest(tickers)
        save_results("experiment_AH_yearly_backtest", yb_results, results)

    if args.experiment in ("universe_scan",):
        us_rows = experiment_universe_scan(tickers, max_position_pct=args.max_position_pct)
        save_results("experiment_AI_universe_scan", us_rows, results)

    if args.experiment in ("super_portfolio",):
        pos_pcts = [p / 100.0 for p in args.pos_pcts] if args.pos_pcts else None
        sp_results = experiment_super_portfolio(tickers, pos_pcts=pos_pcts,
                                                max_hold_bars=args.max_hold_bars,
                                                min_conf=args.min_conf)
        save_results("experiment_AJ_super_portfolio", sp_results, results)

    if args.experiment in ("hold_sweep",):
        hs_results = experiment_hold_sweep(tickers)
        save_results("experiment_AK_hold_sweep", hs_results, results)

    if args.experiment in ("rr_sweep",):
        rr_results = experiment_rr_sweep(tickers)
        save_results("experiment_AL_rr_sweep", rr_results, results)

    if args.experiment in ("conf_sweep_dc",):
        cs_results = experiment_conf_sweep_dc(tickers)
        save_results("experiment_AM_conf_sweep_dc", cs_results, results)

    if args.experiment in ("reentry",):
        re_results = experiment_reentry(tickers)
        save_results("experiment_AN_reentry", re_results, results)

    if args.experiment in ("new_features",):
        print(f"\n{'#'*60}")
        print(f"# EXPERIMENT AO: New features (d_trend_align + d_rs_vs_spy)")
        print(f"# dc+hold6+long+100%% on {', '.join(sorted(tickers))}")
        print(f"{'#'*60}")
        orig_pos = CONFIG.bot.max_position_pct
        CONFIG.bot.max_position_pct = 1.0
        try:
            rows = []
            for t in sorted(tickers):
                r = run_wf(t, daily_confirm=True, allow_short=False, max_hold_bars=6)
                rows.append(r)
                if not r.get("error"):
                    print(f"  {t:<6} CAGR={r['cagr_pct']:>+7.1f}%  win={r['win_rate_pct']:.1f}%  "
                          f"trades={r['total_trades']:>4}  DD={r['max_drawdown_pct']:.1f}%  conf={r['confidence']}")
            valid = [r for r in rows if not r.get("error") and r.get("total_trades", 0) >= 5]
            avg_cagr = sum(r["cagr_pct"] for r in valid) / len(valid) if valid else 0
            above20 = sum(1 for r in valid if r["cagr_pct"] >= 20)
            above10 = sum(1 for r in valid if r["cagr_pct"] >= 10)
            print(f"\n  RESULT: avg_CAGR={avg_cagr:+.1f}%  >=10%:{above10}  >=20%:{above20}")
        finally:
            CONFIG.bot.max_position_pct = orig_pos
        save_results("experiment_AO_new_features", rows, results)

    if args.experiment in ("regime_gate",):
        rg_results = experiment_regime_gate(tickers)
        save_results("experiment_AQ_regime_gate", rg_results, results)

    if args.experiment in ("breakeven_stop",):
        be_results = experiment_breakeven_stop(tickers)
        save_results("experiment_AR_breakeven_stop", be_results, results)

    if args.experiment in ("first_touch_dc",):
        ft_results = experiment_first_touch_dc(tickers)
        save_results("experiment_AS_first_touch_dc", ft_results, results)

    if args.experiment in ("partial_tp",):
        pt_results = experiment_partial_tp(tickers)
        save_results("experiment_AT_partial_tp", pt_results, results)

    if args.experiment in ("vol_targeting",):
        vt_results = experiment_vol_targeting(tickers)
        save_results("experiment_AU_vol_targeting", vt_results, results)

    if args.experiment in ("best_stack",):
        bs_results = experiment_best_stack(tickers)
        save_results("experiment_AV_best_stack", bs_results, results)

    if args.experiment in ("daily_conf_gate",):
        dc_conf_results = experiment_daily_conf_gate(tickers)
        save_results("experiment_AW_daily_conf_gate", dc_conf_results, results)

    if args.experiment in ("sector_features",):
        sx_results = experiment_sector_features(tickers)
        save_results("experiment_AX_sector_features", sx_results, results)

    if args.experiment in ("daily_horizon",):
        dh_results = experiment_daily_horizon(tickers)
        save_results("experiment_AY_daily_horizon", dh_results, results)

    if args.experiment in ("ensemble",):
        en_results = experiment_ensemble(tickers)
        save_results("experiment_AZ_ensemble", en_results, results)

    if args.experiment in ("post_optuna_eval",):
        poe_results = experiment_post_optuna_eval(tickers, hold_bars=args.max_hold_bars,
                                                  min_conf=args.min_conf)
        save_results("experiment_BAA_post_optuna_eval", poe_results, results)

    print(f"\nDone. Results in {RESULTS_FILE}")


if __name__ == "__main__":
    main()
