"""
Multi-Ticker Pooled Model Research

Current approach: 6 separate XGBoost models, one per ticker.
  Pros: each model specializes for that ticker's regime
  Cons: each model only has ~3400 hourly bars (2 years of data)
        splits into 4 folds -> each train has ~2000 bars -> overfitting risk

Pooled approach: 1 model trained on all 6 tickers simultaneously.
  - Ticker identity = {is_app, is_amd, is_googl, is_snow, is_mrvl, is_tsla}
  - Total training rows: 6 × 3400 = 20,400 bars
  - 10x more data per model → better regularization, less overfitting
  - Cross-ticker patterns learned naturally (sector rotation, etc.)
  - Lead-lag features become even more powerful (model can learn NVDA→AMD within same batch)

Hypothesis:
  A pooled model trained on 20k samples will generalize MUCH better than
  6 separate models on 3k samples each. The ticker one-hot encoding allows
  the model to learn both shared patterns (macro, bonds) and ticker-specific
  behavior (TSLA is more volatile, SNOW has fewer trades).

Methodology:
  1. Pool all 6 tickers into one DataFrame (stack vertically)
  2. Add ticker one-hot encoding (is_app, is_amd, etc.)
  3. Run same walk-forward (fold by time, not by ticker)
  4. Compare: pooled vs. individual models on each ticker
"""
import sys, warnings
sys.path.insert(0, r'C:\Users\adamk\Documents\RSI-Model')
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from scipy import stats

from data.fetcher import get_price_data, get_daily_data
from data.indicators import (add_all_indicators, add_daily_trend_features,
                              add_peer_features, add_macro_event_features,
                              add_market_features, add_cross_asset_features,
                              add_pead_features, add_calendar_features,
                              add_cross_sectional_features,
                              add_microstructure_features, add_regime_features,
                              add_earnings_features)
from data.fetcher import get_earnings_dates
from ml.features import get_feature_matrix, build_target, target_bars_for_interval
from ml.model import RSIModel
from ml.optimizer import load_best_params
from backtest.engine import _run_simulation
from config import CONFIG

TICKERS  = ["APP", "AMD", "GOOGL", "SNOW", "MRVL", "TSLA"]
PERIOD   = "2y"
INTERVAL = "1h"
N_FOLDS  = 4


def build_full_features(ticker: str, all_close: dict) -> tuple:
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
    try:
        edates = get_earnings_dates(ticker)
        if edates:
            df = add_earnings_features(df, edates)
    except Exception:
        pass

    features = get_feature_matrix(df, 0.0).dropna()
    n_bars = target_bars_for_interval(INTERVAL)
    target = build_target(df.loc[features.index], n_bars=n_bars)
    mask = target.notna() & (np.arange(len(features)) < len(features) - n_bars)
    return features[mask], target[mask], df.loc[features[mask].index]


def add_ticker_dummies(X: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add one-hot encoding for ticker identity."""
    for t in TICKERS:
        X[f"is_{t.lower()}"] = 1.0 if t == ticker else 0.0
    return X


def run_individual_model(ticker: str, X: pd.DataFrame, y: pd.Series,
                         df: pd.DataFrame, saved: dict) -> dict:
    """Walk-forward with per-ticker model (baseline)."""
    n = len(X)
    fold_size = n // (N_FOLDS + 1)

    adx_min = saved.get("adx_filter_min", 15.0) if saved else 15.0
    tp = saved.get("take_profit", CONFIG.bot.take_profit_pct) if saved else CONFIG.bot.take_profit_pct
    sl = saved.get("stop_loss",   CONFIG.bot.stop_loss_pct)   if saved else CONFIG.bot.stop_loss_pct
    mc = saved.get("min_confidence", 0.0) if saved else 0.0

    orig_xgb = {}
    if saved and "xgb_params" in saved:
        orig_xgb = {k: CONFIG.ml.xgb_params[k] for k in saved["xgb_params"]
                    if k in CONFIG.ml.xgb_params}
        CONFIG.ml.xgb_params.update(saved["xgb_params"])
    orig_tp   = CONFIG.bot.take_profit_pct
    orig_sl   = CONFIG.bot.stop_loss_pct
    orig_conf = CONFIG.bot.min_confidence
    CONFIG.bot.take_profit_pct = tp
    CONFIG.bot.stop_loss_pct   = sl
    CONFIG.bot.min_confidence  = mc

    _BARS_PER_YEAR = 252 * 6.5  # ~1638 hourly bars per trading year

    fold_results = []
    try:
        for fold in range(1, N_FOLDS + 1):
            train_end  = fold * fold_size
            test_start = train_end
            test_end   = min((fold + 1) * fold_size, n)

            X_tr = X.iloc[:train_end]
            y_tr = y.iloc[:train_end]
            X_te = X.iloc[test_start:test_end]
            df_te = df.iloc[test_start:test_end]

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
            r = _run_simulation(df_te, X_te, model, 100_000.0, ticker, adx_min,
                                 allow_short=False)
            m = r["metrics"]
            # Compute CAGR from total_return_pct and fold duration
            years = len(df_te) / _BARS_PER_YEAR
            tr = r.get("total_return_pct", 0.0) / 100.0
            cagr_fold = ((1.0 + tr) ** (1.0 / max(years, 0.01)) - 1.0) * 100.0
            fold_results.append({
                "cagr": cagr_fold,
                "total_trades": m.get("total_trades", 0),
                "win_rate": m.get("win_rate_pct", 0.0),
            })
    finally:
        CONFIG.bot.take_profit_pct = orig_tp
        CONFIG.bot.stop_loss_pct   = orig_sl
        CONFIG.bot.min_confidence  = orig_conf
        if orig_xgb:
            CONFIG.ml.xgb_params.update(orig_xgb)

    if not fold_results:
        return {"cagr": 0.0, "total_trades": 0, "win_rate": 0.0}
    return {
        "cagr": np.mean([m["cagr"] for m in fold_results]),
        "total_trades": sum(m["total_trades"] for m in fold_results),
        "win_rate": np.mean([m["win_rate"] for m in fold_results
                             if m["total_trades"] > 0]) if any(
                             m["total_trades"] > 0 for m in fold_results) else 0.0,
    }


def run():
    print("=" * 70)
    print("  Multi-Ticker Pooled Model Research")
    print(f"  Tickers: {TICKERS}")
    print(f"  Hypothesis: pooled 20k-row model generalizes better than 6×3k models")
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

    # Build features for each ticker
    print("\nBuilding features...")
    ticker_features = {}
    for ticker in TICKERS:
        print(f"  {ticker}...", end=" ")
        try:
            X, y, df = build_full_features(ticker, all_close)
            # Add ticker dummies to features
            X = add_ticker_dummies(X.copy(), ticker)
            ticker_features[ticker] = {"X": X, "y": y, "df": df}
            print(f"{len(X)} rows x {len(X.columns)} cols")
        except Exception as e:
            print(f"ERROR: {e}")
            ticker_features[ticker] = None

    # === Part 1: Individual models (baseline) ===
    print("\n\n--- Individual models (baseline) ---")
    individual_results = {}
    for ticker in TICKERS:
        td = ticker_features.get(ticker)
        if td is None:
            continue
        saved = load_best_params(ticker)
        result = run_individual_model(ticker, td["X"], td["y"], td["df"], saved)
        individual_results[ticker] = result
        print(f"  {ticker}: CAGR={result['cagr']:+.1f}%  trades={result['total_trades']:>4}  WR={result['win_rate']:.1f}%")

    # === Part 2: Pooled model ===
    print("\n--- Pooled model (all 6 tickers combined) ---")

    # Find common features across all tickers
    all_feature_sets = [set(td["X"].columns) for td in ticker_features.values() if td]
    common_features = sorted(all_feature_sets[0].intersection(*all_feature_sets[1:]))
    print(f"  Common features across tickers: {len(common_features)}")

    # Pool data (time-aligned: interleave by sorted timestamp)
    pool_dfs = []
    for ticker, td in ticker_features.items():
        if td is None:
            continue
        X_sub = td["X"][common_features].copy()
        X_sub["_ticker"] = ticker
        X_sub["_target"] = td["y"].values
        # Assign df row data for simulation
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in td["df"].columns:
                X_sub[f"_df_{col}"] = td["df"][col].values
        pool_dfs.append(X_sub)

    if not pool_dfs:
        print("  No data to pool.")
        return

    # NOTE: Pooling approach uses a simplified walk-forward by ticker
    # (since timestamps across tickers overlap — fold splits must be per-ticker)
    # We test: train on 5 tickers, predict 6th (leave-one-out)

    print("\n  LEAVE-ONE-TICKER-OUT cross-validation:")
    print("  (Train on 5 tickers, evaluate on the held-out ticker)")
    print()

    loo_results = {}
    for target_ticker in TICKERS:
        td_target = ticker_features.get(target_ticker)
        if td_target is None:
            continue

        # Train on all OTHER tickers
        train_X_parts = []
        train_y_parts = []
        for t, td in ticker_features.items():
            if t == target_ticker or td is None:
                continue
            train_X_parts.append(td["X"][common_features])
            train_y_parts.append(td["y"])

        if not train_X_parts:
            continue

        X_train = pd.concat(train_X_parts)
        y_train = pd.concat(train_y_parts)
        X_test  = td_target["X"][common_features]
        y_test  = td_target["y"]
        df_test = td_target["df"]

        # Shuffle training data to avoid ordering effects
        perm = np.random.default_rng(42).permutation(len(X_train))
        X_train = X_train.iloc[perm]
        y_train = y_train.iloc[perm]

        saved = load_best_params(target_ticker)
        orig_xgb = {}
        if saved and "xgb_params" in saved:
            orig_xgb = {k: CONFIG.ml.xgb_params[k] for k in saved["xgb_params"]
                        if k in CONFIG.ml.xgb_params}
            CONFIG.ml.xgb_params.update(saved["xgb_params"])
        adx_min = saved.get("adx_filter_min", 15.0) if saved else 15.0
        orig_tp   = CONFIG.bot.take_profit_pct
        orig_sl   = CONFIG.bot.stop_loss_pct
        orig_conf = CONFIG.bot.min_confidence
        CONFIG.bot.take_profit_pct = saved.get("take_profit",      orig_tp)   if saved else orig_tp
        CONFIG.bot.stop_loss_pct   = saved.get("stop_loss",        orig_sl)   if saved else orig_sl
        CONFIG.bot.min_confidence  = saved.get("min_confidence",   0.0)       if saved else 0.0

        try:
            val_cut = int(len(X_train) * 0.85)
            model = RSIModel()
            model.build(early_stopping_rounds=20)
            model.fit(X_train.iloc[:val_cut], y_train.iloc[:val_cut],
                      X_val=X_train.iloc[val_cut:], y_val=y_train.iloc[val_cut:])
            r = _run_simulation(df_test, X_test, model, 100_000.0,
                                target_ticker, adx_min, allow_short=False)
            m = r["metrics"]
            _BARS_PER_YEAR = 252 * 6.5
            years = len(df_test) / _BARS_PER_YEAR
            tr = r.get("total_return_pct", 0.0) / 100.0
            cagr = ((1.0 + tr) ** (1.0 / max(years, 0.01)) - 1.0) * 100.0
            trades = m.get("total_trades", 0)
            wr = m.get("win_rate_pct", 0.0)
        except Exception as e:
            cagr, trades, wr = 0.0, 0, 0.0
            print(f"    {target_ticker} ERROR: {e}")
        finally:
            CONFIG.bot.take_profit_pct = orig_tp
            CONFIG.bot.stop_loss_pct   = orig_sl
            CONFIG.bot.min_confidence  = orig_conf
            if orig_xgb:
                CONFIG.ml.xgb_params.update(orig_xgb)

        loo_results[target_ticker] = {"cagr": cagr, "total_trades": trades, "win_rate": wr}
        print(f"  LOO {target_ticker:>6}: CAGR={cagr:+.1f}%  trades={trades:>4}  WR={wr:.1f}%")

    # === Summary ===
    print("\n\n" + "=" * 70)
    print("  SUMMARY: Individual vs. Pooled (Leave-One-Out)")
    print("=" * 70)
    print(f"\n  {'Ticker':<8} {'Individual CAGR':>16} {'LOO-Pooled CAGR':>16} {'Delta':>8}")
    print("  " + "-" * 55)
    deltas = []
    for ticker in TICKERS:
        ind = individual_results.get(ticker, {}).get("cagr", 0.0)
        loo = loo_results.get(ticker,         {}).get("cagr", 0.0)
        delta = loo - ind
        deltas.append(delta)
        better = "POOLED" if delta > 0 else "INDIV "
        print(f"  {ticker:<8} {ind:>+15.1f}%  {loo:>+15.1f}%  {delta:>+8.1f}%  [{better}]")

    if deltas:
        print(f"\n  Average delta: {np.mean(deltas):>+.1f}%")
        pooled_wins = sum(1 for d in deltas if d > 0)
        print(f"  Pooled wins: {pooled_wins}/{len(deltas)} tickers")

        if pooled_wins >= 4:
            print("\n  RECOMMENDATION: Implement pooled model in production")
            print("  Next step: add pooled optimizer to reoptimize_wave6.py")
        elif pooled_wins >= 2:
            print("\n  RECOMMENDATION: Mixed — use pooled for some tickers")
        else:
            print("\n  RECOMMENDATION: Stick with individual models (specialization wins)")


if __name__ == "__main__":
    run()
