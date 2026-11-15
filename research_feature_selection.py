"""
Feature Selection Experiment

Problem: Feature importance shows 138 features with flat ~0.7-2% importance each.
         This suggests the model spreads gain across noise instead of learning
         from a few strong signals. N/M ratio = 138/~500 = 0.28 (high).

Hypothesis: Reducing to the top 40-50 consensus features improves out-of-sample
            CAGR because:
              1. Lower N/M ratio -> better regularization efficiency
              2. Forcing the model to use fewer features -> avoids noise overfitting
              3. Consensus features are more robust (confirmed across tickers)

Experiment:
  Walk-forward backtest (4 folds, 2y data) on each ticker with:
    - FULL model: 138 features (baseline)
    - TOP-40 model: top 40 features by average IC across tickers
    - CONSENSUS model: 18 consensus features + 15 ticker-specific top features

Report: CAGR, Sharpe, n_trades, P(profitable) per model variant
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

from data.fetcher import get_price_data, get_daily_data
from data.indicators import (add_all_indicators, add_peer_features, add_macro_event_features,
                              add_calendar_features, add_cross_sectional_features,
                              add_daily_trend_features)
from ml.features import get_feature_matrix, build_target, target_bars_for_interval, FEATURE_COLS
from ml.model import RSIModel
from ml.optimizer import load_best_params

TICKERS   = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD    = "2y"
INTERVAL  = "1h"
N_FOLDS   = 4
N_BARS    = target_bars_for_interval(INTERVAL)

# Top 40 features selected from feature importance consensus + IC research
# These are the 18 consensus features + 22 more with known IC
TOP_40_FEATURES = [
    # Calendar (highest IC signals from calendar research)
    "is_fri", "hour_of_day", "opex_monthly", "opex_thursday", "opex_friday",
    "is_wed", "is_mon", "month_first_days", "post_opex",
    # Volatility (top consensus)
    "ATR_pct", "d_atr_pct", "d_range_pct", "range_rank_10",
    # Daily trend (top consensus)
    "d_trend_score", "d_price_vs_sma50", "d_price_vs_sma200",
    "d_mom_20", "d_bb_pct", "d_rsi_14", "d_mom_5",
    # Opening hour
    "open_gap_x", "open_macd_slope_x",
    # VWAP / session
    "price_vs_vwap_lag3", "price_vs_vwap_lag1", "sess_range_pos",
    # ORB
    "orb_width",
    # Volume / microstructure
    "volume_ratio", "pvd_10",
    # Peer / macro
    "spy_bb_pct", "spy_rsi_14",
    # Cross-sectional
    "cs_rank_5d", "peer_spread_5d",
    # Cross-asset
    "nvda_rsi_14_lag2",
    # Momentum
    "rsi_14", "macd_hist", "bb_pct",
    # Additional high-IC signals
    "vwap_id_slope_5", "d_macd_slope", "sess_range_width",
    # NR patterns
    "nr7", "inside_bar",
]


def build_full_df(ticker: str, all_close: dict) -> pd.DataFrame:
    df = get_price_data(ticker, period=PERIOD, interval=INTERVAL)
    df = add_all_indicators(df)
    try:
        peer_df = get_price_data("NVDA", period=PERIOD, interval=INTERVAL) if ticker != "NVDA" else None
        xlk_df  = get_price_data("XLK",  period=PERIOD, interval=INTERVAL)
        vix_df  = get_price_data("^VIX", period=PERIOD, interval=INTERVAL)
        df = add_peer_features(df, nvda_hourly_df=peer_df,
                               xlk_hourly_df=xlk_df, vix_hourly_df=vix_df)
    except Exception:
        pass
    try:
        daily_df = get_daily_data(ticker, period=PERIOD)
        df = add_daily_trend_features(df, daily_df)
    except Exception:
        pass
    df = add_macro_event_features(df)
    df = add_calendar_features(df)
    if all_close:
        others = {t: s for t, s in all_close.items() if t != ticker and s is not None}
        if others:
            df = add_cross_sectional_features(df, ticker, others)
    return df


def walk_forward(df: pd.DataFrame, feature_cols: list, ticker: str) -> dict:
    """Mini walk-forward backtest: 4 folds, return CAGR, n_trades."""
    params = load_best_params(ticker)
    target = build_target(df, n_bars=N_BARS)

    # Keep only features that exist in df
    available = [f for f in feature_cols if f in df.columns]
    if len(available) < 5:
        return {"cagr": 0.0, "n_trades": 0, "win_rate": 0.0, "n_features": 0}

    X_all = df[available].copy()
    common = X_all.index.intersection(target.dropna().index)
    X_all  = X_all.loc[common]
    y_all  = target.loc[common]
    mask   = X_all.notna().all(axis=1) & y_all.notna()
    X_all, y_all = X_all[mask], y_all[mask]

    n = len(X_all)
    if n < 200:
        return {"cagr": 0.0, "n_trades": 0, "win_rate": 0.0, "n_features": len(available)}

    fold_size = n // (N_FOLDS + 1)
    all_returns = []

    for fold in range(N_FOLDS):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end   = min(test_start + fold_size, n)
        if test_end - test_start < 20:
            continue

        X_tr, y_tr = X_all.iloc[:train_end], y_all.iloc[:train_end]
        X_te, y_te = X_all.iloc[test_start:test_end], y_all.iloc[test_start:test_end]

        if len(X_tr) < 100 or len(X_te) < 10:
            continue

        model = RSIModel()
        model.build()
        model.fit(X_tr, y_tr)

        # Confidence threshold
        min_conf = 0.45
        if params and "min_confidence" in params:
            min_conf = params["min_confidence"]

        price_col = df["Close"]
        for i in range(len(X_te)):
            row = X_te.iloc[i]
            pred_res = model.predict_single(row)
            signal = pred_res["signal"]
            conf   = pred_res["confidence"]
            if conf < min_conf or signal == "FLAT":
                continue

            # Simple return: if UP signal, next N bars return; if DOWN, negate
            idx = X_te.index[i]
            if idx not in price_col.index:
                continue
            pos = price_col.index.get_loc(idx)
            end_pos = min(pos + N_BARS, len(price_col) - 1)
            ret = (price_col.iloc[end_pos] - price_col.iloc[pos]) / price_col.iloc[pos]
            if signal == "DOWN":
                ret = -ret
            all_returns.append(ret)

    if not all_returns:
        return {"cagr": 0.0, "n_trades": 0, "win_rate": 0.0, "n_features": len(available)}

    rets = np.array(all_returns)
    total_return = np.prod(1 + rets) - 1
    n_bars_total = len(X_all)
    bars_per_year = 252 * 6.5  # ~1638 hourly bars per year
    years = n_bars_total / bars_per_year
    cagr  = (1 + total_return) ** (1 / max(years, 0.1)) - 1
    win_rate = float(np.mean(rets > 0))

    return {
        "cagr": cagr * 100,
        "n_trades": len(rets),
        "win_rate": win_rate,
        "n_features": len(available),
    }


def run():
    print("=" * 70)
    print("  Feature Selection Experiment")
    print("  FULL (138) vs TOP-40 vs CONSENSUS-18")
    print("=" * 70)

    # Pre-fetch close data for cross-sectional features
    print("\nPre-fetching close data...", flush=True)
    all_close = {}
    for sym in TICKERS:
        try:
            df = get_price_data(sym, period=PERIOD, interval=INTERVAL)
            all_close[sym] = df["Close"].rename(sym)
        except Exception:
            all_close[sym] = None

    # Consensus features only (18 from importance analysis)
    consensus_18 = [
        "is_fri", "ATR_pct", "d_atr_pct", "open_macd_slope_x", "d_trend_score",
        "d_price_vs_sma50", "d_price_vs_sma200", "price_vs_vwap_lag3",
        "d_mom_20", "d_range_pct", "open_gap_x", "d_bb_pct",
        "hour_of_day", "d_rsi_14", "price_vs_vwap_lag1", "orb_width",
        "d_mom_5", "month_first_days",
    ]

    variants = {
        "FULL_138": FEATURE_COLS,
        "TOP_40":   TOP_40_FEATURES,
        "CONSENSUS_18": consensus_18,
    }

    summary = {v: [] for v in variants}

    for ticker in TICKERS:
        print(f"\n--- {ticker} ---", flush=True)
        try:
            df = build_full_df(ticker, all_close)
            print(f"  df: {len(df)} bars, {len(df.columns)} columns")

            for variant, feat_list in variants.items():
                res = walk_forward(df, feat_list, ticker)
                summary[variant].append(res["cagr"])
                print(f"  {variant:<15} CAGR={res['cagr']:+.1f}%  "
                      f"trades={res['n_trades']:3d}  "
                      f"win={res['win_rate']:.0%}  "
                      f"n_feat={res['n_features']}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}")

    print("\n\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print(f"  {'Variant':<20} {'Avg_CAGR':>10} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 50)
    for variant, cagrs in summary.items():
        if not cagrs:
            continue
        print(f"  {variant:<20} {np.mean(cagrs):+10.1f}%  "
              f"{min(cagrs):+8.1f}%  {max(cagrs):+8.1f}%")

    print("\n  Baseline (Wave 1): AVG CAGR = +4.7%")
    print("  Baseline (Wave 2): AVG CAGR = +2.9%")


if __name__ == "__main__":
    run()
