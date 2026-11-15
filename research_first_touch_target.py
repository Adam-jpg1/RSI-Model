"""
First-Touch Target vs Percentile Return Target

Current target (build_target): labels top/bottom 33% of returns as UP/DOWN.
  - Percentile-based: always produces 33/33/33 class balance
  - But UP label can be tiny positive return (e.g., +0.3%) in a low-vol day
  - Trains the model to predict RELATIVE performance, not ABSOLUTE profitability

First-touch target (build_target_first_touch): labels based on TP/SL hit.
  - UP = price hits take_profit (e.g., +2.0%) before stop_loss (e.g., -1.3%)
  - DOWN = SL hit first
  - FLAT = neither within N bars
  - Directly maps to trading outcomes
  - Class imbalance: depends on market conditions (may need weighting)

Hypothesis: First-touch labels train a model that is directly optimized for
the actual trade entry/exit mechanics. The percentile model is optimized for
prediction accuracy but the threshold between "good enough to trade" vs.
"too small to trade" isn't captured.

Methodology:
  1. Build both targets for all 6 tickers
  2. Train models with both targets using saved params
  3. Run walk-forward backtest: compare CAGR, win rate, # trades
  4. Check class distribution for first-touch labels
  5. If first-touch wins: add it as default via CONFIG.ml.use_first_touch_target = True
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

from data.fetcher import get_price_data, get_daily_data
from data.indicators import (add_all_indicators, add_daily_trend_features,
                              add_peer_features, add_macro_event_features,
                              add_market_features, add_cross_asset_features,
                              add_pead_features, add_calendar_features,
                              add_cross_sectional_features,
                              add_microstructure_features, add_regime_features)
from ml.features import (get_feature_matrix, build_target,
                          build_target_first_touch, target_bars_for_interval)
from ml.model import RSIModel
from ml.optimizer import load_best_params
from backtest.engine import _run_simulation
from config import CONFIG

TICKERS  = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD   = "2y"
INTERVAL = "1h"
N_FOLDS  = 4


def build_full_df(ticker: str, all_close: dict) -> tuple:
    df = get_price_data(ticker, period=PERIOD, interval=INTERVAL)
    df = add_all_indicators(df)
    try:
        daily_df = get_daily_data(ticker, period=PERIOD)
        daily_df = add_all_indicators(daily_df)
        spy_daily = None
        try:
            spy_daily = get_daily_data("SPY", period=PERIOD)
        except Exception:
            pass
        vix_daily = None
        try:
            vix_daily = get_daily_data("^VIX", period=PERIOD)
        except Exception:
            pass
        df = add_daily_trend_features(df, daily_df, spy_daily_df=spy_daily,
                                      vix_daily_df=vix_daily)
        try:
            ca = {}
            for sym in ["TLT", "HYG", "GLD", "UUP"]:
                try:
                    ca[sym] = get_daily_data(sym, period=PERIOD)
                except Exception:
                    ca[sym] = None
            df = add_cross_asset_features(df, tlt_daily_df=ca.get("TLT"),
                                          hyg_daily_df=ca.get("HYG"),
                                          gld_daily_df=ca.get("GLD"),
                                          uup_daily_df=ca.get("UUP"))
        except Exception:
            pass
        try:
            df = add_pead_features(df, daily_df)
        except Exception:
            pass
    except Exception:
        pass
    try:
        peer_assets = {}
        for sym in ["NVDA", "XLK", "^VIX"]:
            if sym != ticker:
                try:
                    peer_assets[sym] = get_price_data(sym, period=PERIOD, interval=INTERVAL)
                except Exception:
                    peer_assets[sym] = None
        df = add_peer_features(df, nvda_hourly_df=peer_assets.get("NVDA"),
                               xlk_hourly_df=peer_assets.get("XLK"),
                               vix_hourly_df=peer_assets.get("^VIX"))
    except Exception:
        pass
    try:
        spy_h = get_price_data("SPY", period=PERIOD, interval=INTERVAL)
        spy_h = add_all_indicators(spy_h)
        df = add_market_features(df, spy_h)
    except Exception:
        pass
    try:
        df = add_macro_event_features(df)
    except Exception:
        pass
    try:
        df = add_calendar_features(df)
    except Exception:
        pass
    peer_close = {t: s for t, s in all_close.items()
                  if t != ticker and s is not None}
    if peer_close:
        try:
            df = add_cross_sectional_features(df, ticker, peer_close)
        except Exception:
            pass
    try:
        df = add_microstructure_features(df)
    except Exception:
        pass
    try:
        df = add_regime_features(df)
    except Exception:
        pass
    return df


def run_wf_with_target(ticker: str, df: pd.DataFrame, use_first_touch: bool,
                       saved: dict) -> dict:
    """Walk-forward backtest using percentile or first-touch target."""
    n_bars = target_bars_for_interval(INTERVAL)
    features = get_feature_matrix(df, 0.0).dropna()
    df_aligned = df.loc[features.index]

    adx_min = saved.get("adx_filter_min", 15.0) if saved else 15.0
    tp = saved.get("take_profit", CONFIG.bot.take_profit_pct) if saved else CONFIG.bot.take_profit_pct
    sl = saved.get("stop_loss",   CONFIG.bot.stop_loss_pct)   if saved else CONFIG.bot.stop_loss_pct
    max_bars = saved.get("max_hold_bars", 6)                  if saved else 6

    if use_first_touch:
        # First-touch lookahead: use max_bars * 8 (matches prepare_dataset)
        ft_bars = n_bars * 8
        target = build_target_first_touch(
            df_aligned, take_profit=tp, stop_loss=sl, max_bars=ft_bars
        )
        mask_tail = np.arange(len(features)) < len(features) - ft_bars
    else:
        target = build_target(df_aligned, n_bars=n_bars)
        mask_tail = np.arange(len(features)) < len(features) - n_bars

    mask = features.notna().all(axis=1) & target.notna() & mask_tail
    features = features[mask]
    target   = target[mask]
    df_al    = df_aligned[mask]

    n = len(features)
    fold_size = n // (N_FOLDS + 1)

    orig_tp = CONFIG.bot.take_profit_pct
    orig_sl = CONFIG.bot.stop_loss_pct
    orig_xgb = {}
    if saved and "xgb_params" in saved:
        orig_xgb = {k: CONFIG.ml.xgb_params[k] for k in saved["xgb_params"]
                    if k in CONFIG.ml.xgb_params}
        CONFIG.ml.xgb_params.update(saved["xgb_params"])
    CONFIG.bot.take_profit_pct = tp
    CONFIG.bot.stop_loss_pct   = sl

    fold_results = []
    try:
        for fold in range(1, N_FOLDS + 1):
            train_end  = fold * fold_size
            test_start = train_end
            test_end   = min((fold + 1) * fold_size, n)

            X_tr = features.iloc[:train_end]
            y_tr = target.iloc[:train_end]
            X_te = features.iloc[test_start:test_end]
            df_te = df_al.iloc[test_start:test_end]

            if len(X_tr) < 100:
                continue

            val_cut = int(len(X_tr) * 0.85)
            model = RSIModel()
            model.build(early_stopping_rounds=20)
            try:
                model.fit(X_tr.iloc[:val_cut], y_tr.iloc[:val_cut],
                          X_val=X_tr.iloc[val_cut:], y_val=y_tr.iloc[val_cut:])
            except Exception:
                continue

            r = _run_simulation(df_te, X_te, model, 100_000.0,
                                ticker, adx_min, allow_short=False,
                                max_hold_bars=max_bars)
            fold_results.append(r["metrics"])
    finally:
        CONFIG.bot.take_profit_pct = orig_tp
        CONFIG.bot.stop_loss_pct   = orig_sl
        if orig_xgb:
            CONFIG.ml.xgb_params.update(orig_xgb)

    if not fold_results:
        return {"cagr": 0.0, "total_trades": 0, "win_rate": 0.0}
    total_trades = sum(m.get("total_trades", 0) for m in fold_results)
    avg_cagr = np.mean([m.get("cagr", 0.0) for m in fold_results])
    valid_wr = [m.get("win_rate", 0.0) for m in fold_results if m.get("total_trades", 0) > 0]
    avg_win  = np.mean(valid_wr) if valid_wr else 0.0
    return {"cagr": avg_cagr, "total_trades": total_trades, "win_rate": avg_win}


def run():
    print("=" * 70)
    print("  First-Touch vs Percentile Target Comparison")
    print(f"  Tickers: {TICKERS}  |  {N_FOLDS}-fold walk-forward")
    print(f"  TP={CONFIG.bot.take_profit_pct:.1%}  SL={CONFIG.bot.stop_loss_pct:.1%}")
    print("=" * 70)

    # Pre-fetch Close series
    print("\nPre-fetching Close series...")
    all_close = {}
    for sym in TICKERS:
        try:
            df_sym = get_price_data(sym, period=PERIOD, interval=INTERVAL)
            all_close[sym] = df_sym["Close"].rename(sym)
            print(f"  {sym}: {len(df_sym)} bars")
        except Exception as e:
            print(f"  {sym}: ERROR {e}")
            all_close[sym] = None

    results_table = []

    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")
        try:
            df = build_full_df(ticker, all_close)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        saved = load_best_params(ticker)
        tp = saved.get("take_profit", CONFIG.bot.take_profit_pct) if saved else CONFIG.bot.take_profit_pct
        sl = saved.get("stop_loss",   CONFIG.bot.stop_loss_pct)   if saved else CONFIG.bot.stop_loss_pct

        # Check class distribution for first-touch labels
        n_bars = target_bars_for_interval(INTERVAL)
        ft_target = build_target_first_touch(df, take_profit=tp, stop_loss=sl,
                                              max_bars=n_bars * 8)
        counts = ft_target.value_counts().sort_index()
        total_ft = len(ft_target)
        print(f"  First-touch label distribution (TP={tp:.1%}, SL={sl:.1%}):")
        for label, cnt in counts.items():
            names = {0: "DOWN", 1: "FLAT", 2: "UP"}
            print(f"    {names.get(label, label)}: {cnt} ({cnt/total_ft:.0%})")

        pct_target = build_target(df, n_bars=n_bars)
        counts_pct = pct_target.value_counts().sort_index()
        print(f"  Percentile label distribution:")
        for label, cnt in counts_pct.items():
            names = {0: "DOWN", 1: "FLAT", 2: "UP"}
            print(f"    {names.get(label, label)}: {cnt} ({cnt/len(pct_target):.0%})")

        print(f"\n  Running walk-forward with percentile target...")
        pct_result = run_wf_with_target(ticker, df, use_first_touch=False, saved=saved)
        print(f"  Percentile: CAGR={pct_result['cagr']:+.1f}%  "
              f"trades={pct_result['total_trades']}  WR={pct_result['win_rate']:.0%}")

        print(f"  Running walk-forward with first-touch target...")
        ft_result = run_wf_with_target(ticker, df, use_first_touch=True, saved=saved)
        print(f"  First-touch: CAGR={ft_result['cagr']:+.1f}%  "
              f"trades={ft_result['total_trades']}  WR={ft_result['win_rate']:.0%}")

        winner = "FIRST-TOUCH" if ft_result["cagr"] > pct_result["cagr"] else "PERCENTILE"
        results_table.append({
            "ticker":    ticker,
            "pct_cagr":  pct_result["cagr"],
            "ft_cagr":   ft_result["cagr"],
            "pct_trades": pct_result["total_trades"],
            "ft_trades":  ft_result["total_trades"],
            "winner":    winner,
        })

    print("\n\n" + "=" * 70)
    print("  SUMMARY: Percentile vs First-Touch Target")
    print("=" * 70)
    print(f"\n  {'Ticker':<8} {'Pct CAGR':>10} {'FT CAGR':>10} {'Delta':>8} {'Winner'}")
    print("  " + "-" * 55)
    ft_wins = 0
    for r in results_table:
        delta = r["ft_cagr"] - r["pct_cagr"]
        if r["winner"] == "FIRST-TOUCH":
            ft_wins += 1
        print(f"  {r['ticker']:<8} {r['pct_cagr']:>+10.1f}% {r['ft_cagr']:>+10.1f}%"
              f" {delta:>+8.1f}% {r['winner']}")

    if results_table:
        avg_pct = np.mean([r["pct_cagr"] for r in results_table])
        avg_ft  = np.mean([r["ft_cagr"]  for r in results_table])
        print(f"\n  {'AVG':<8} {avg_pct:>+10.1f}% {avg_ft:>+10.1f}% {avg_ft-avg_pct:>+8.1f}%")
        print(f"\n  First-touch wins: {ft_wins}/{len(results_table)} tickers")

        if ft_wins >= 4:
            print("\n  RECOMMENDATION: Switch to first-touch target")
            print("  Set CONFIG.ml.use_first_touch_target = True and re-run Wave 4")
        elif ft_wins >= 2:
            print("\n  RECOMMENDATION: Ticker-specific — use first-touch for winners")
        else:
            print("\n  RECOMMENDATION: Keep percentile target (more consistent)")


if __name__ == "__main__":
    run()
