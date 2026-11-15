"""
Systematic edge research: test novel signal hypotheses against baseline.

Experiments:
  E1  Baseline (dc+hold6+long)
  E2  + cross-asset regime features (TLT/HYG/GLD/UUP/VIX term structure)
  E3  + PEAD proxy (post-earnings gap momentum)
  E4  + VIX term structure gating (suppress longs when ca_vts_regime==2 rising fast)
  E5  First-touch target labeling vs percentile
  E6  Ensemble (3 seeds per fold)
  E7  IC analysis — which features actually predict returns?

Usage:
  python edge_research.py                        # all experiments on super portfolio
  python edge_research.py --tickers AMD MU       # subset
  python edge_research.py --exp E1 E2 E7         # specific experiments
  python edge_research.py --ic AMD               # IC analysis only on AMD
"""
import sys, os, json, argparse, warnings
import numpy as np
import pandas as pd
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

from config import CONFIG
from backtest.simulate import walk_forward_backtest, bootstrap_confidence
from data.fetcher import get_price_data, get_daily_data
from data.indicators import (add_all_indicators, add_daily_trend_features,
                              add_cross_asset_features, add_pead_features)
from ml.features import get_feature_matrix, build_target, target_bars_for_interval
from ml.model import RSIModel

RESULTS_FILE = "edge_research_results.json"
BARS_PER_YEAR = 252 * 6.5
N_FOLDS = 4

SUPER_PORTFOLIO = ["AMD", "APP", "AVGO", "GE", "T", "LRCX", "MU", "MSFT"]


# ------------------------------------------------------------------ #
# Utilities                                                           #
# ------------------------------------------------------------------ #

def annualize(total_return_pct: float, test_bars: int = 2728) -> float:
    years = test_bars / BARS_PER_YEAR
    if years <= 0:
        return 0.0
    r = total_return_pct / 100.0
    cagr = (1 + r) ** (1 / years) - 1
    return round(cagr * 100, 2)


def run_wf(ticker: str, **kwargs) -> dict:
    """Thin wrapper: run walk-forward and return a summary dict."""
    orig_tp   = CONFIG.bot.take_profit_pct
    orig_sl   = CONFIG.bot.stop_loss_pct
    orig_conf = CONFIG.bot.min_confidence

    result = walk_forward_backtest(
        ticker, n_folds=N_FOLDS,
        daily_confirm=kwargs.get("daily_confirm", False),
        allow_short=kwargs.get("allow_short", False),
        max_hold_bars=kwargs.get("max_hold_bars", 6),
        n_ensemble=kwargs.get("n_ensemble", 1),
        min_confidence=kwargs.get("min_confidence", None),
        adx_filter_min=kwargs.get("adx_filter_min", None),  # None = use Optuna value
    )

    CONFIG.bot.take_profit_pct = orig_tp
    CONFIG.bot.stop_loss_pct   = orig_sl
    CONFIG.bot.min_confidence  = orig_conf

    compound = 1.0
    for r in result["fold_results"]:
        compound *= (1 + r["total_return_pct"] / 100)
    compound_ret = (compound - 1) * 100

    boot = bootstrap_confidence(result["combined_pnl"])
    cagr = annualize(compound_ret)

    all_dd = []
    for r in result["fold_results"]:
        eq = r["equity_curve"]["equity"] if "equity" in r["equity_curve"].columns else pd.Series([100000])
        rolling_max = eq.cummax()
        dd = float(((eq - rolling_max) / rolling_max).min() * 100)
        all_dd.append(dd)
    max_dd = min(all_dd) if all_dd else 0.0

    return {
        "ticker": ticker,
        "total_trades": result["total_trades"],
        "avg_win_rate": result["avg_win_rate"],
        "total_return_pct": round(compound_ret, 2),
        "cagr": cagr,
        "max_dd": round(max_dd, 1),
        "p_profitable": boot.get("p_profitable", 0.0) if "error" not in boot else 0.0,
        "mean_trade_pct": boot.get("mean_return_pct", 0.0) if "error" not in boot else 0.0,
        "confidence": boot.get("confidence_label", "N/A") if "error" not in boot else "N/A",
    }


def print_table(rows: list, title: str):
    sep = "=" * 76
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  {'Ticker':<8} {'CAGR':>7} {'Return':>8} {'MaxDD':>7} {'WinRate':>8} "
          f"{'Trades':>7} {'P(+)':>6} {'Signal':>8}")
    print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*6} {'-'*8}")
    cagrs = []
    for r in rows:
        if "error" in r:
            print(f"  {r['ticker']:<8}  ERROR: {r['error']}")
            continue
        cagrs.append(r["cagr"])
        print(f"  {r['ticker']:<8} {r['cagr']:>+6.1f}% {r['total_return_pct']:>+7.1f}% "
              f"{r['max_dd']:>+6.1f}% {r['avg_win_rate']:>7.1f}% "
              f"{r['total_trades']:>7} {r['p_profitable']:>5.0%}  {r['confidence']:>8}")
    if cagrs:
        print(f"  {'-'*8} {'-'*7}")
        print(f"  {'AVG':<8} {np.mean(cagrs):>+6.1f}%")
        print(f"  {'TOP5':<8} {np.mean(sorted(cagrs, reverse=True)[:5]):>+6.1f}%")
    print(sep)


# ------------------------------------------------------------------ #
# Information Coefficient Analysis                                    #
# ------------------------------------------------------------------ #

def compute_ic(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Compute Information Coefficient for each feature:
    IC = Spearman rank correlation between feature value and future return.
    IC > 0.03 = useful signal;  IC > 0.06 = strong signal.
    """
    from scipy.stats import spearmanr

    print(f"\n[IC] Loading {ticker}...")
    df = get_price_data(ticker, period=period, interval="1h")
    df = add_all_indicators(df)

    try:
        daily_df = get_daily_data(ticker, period=period)
        spy_df   = get_daily_data("SPY", period=period)
        vix_df   = get_daily_data("^VIX", period=period)
        df = add_daily_trend_features(df, daily_df, compute_weekly=True,
                                      spy_daily_df=spy_df, vix_daily_df=vix_df)
        ca = {}
        for sym in ["TLT", "HYG", "GLD", "UUP", "^VIX9D", "^VIX3M"]:
            try:
                ca[sym] = get_daily_data(sym, period=period)
            except Exception:
                ca[sym] = None
        df = add_cross_asset_features(df, tlt_daily_df=ca.get("TLT"),
                                       hyg_daily_df=ca.get("HYG"),
                                       gld_daily_df=ca.get("GLD"),
                                       uup_daily_df=ca.get("UUP"),
                                       vix9d_daily_df=ca.get("^VIX9D"),
                                       vix3m_daily_df=ca.get("^VIX3M"))
        df = add_pead_features(df, daily_df)
    except Exception as e:
        print(f"[IC] Warning: {e}")

    features = get_feature_matrix(df, sentiment_score=0.0).dropna()
    n_bars = target_bars_for_interval("1h")
    future_ret = df["Close"].shift(-n_bars) / df["Close"] - 1.0
    future_ret = future_ret.reindex(features.index).dropna()
    features = features.loc[future_ret.index]

    ics = {}
    pvals = {}
    for col in features.columns:
        try:
            vals = features[col].values.astype(float)
            if np.std(vals) < 1e-10:  # constant feature — skip
                continue
            ic, pval = spearmanr(vals, future_ret.values, nan_policy="omit")
            ics[col] = round(float(ic), 5)
            pvals[col] = round(float(pval), 5)
        except Exception:
            pass

    df_ic = pd.DataFrame({"IC": ics, "p_value": pvals})
    df_ic["abs_IC"] = df_ic["IC"].abs()
    df_ic = df_ic.sort_values("abs_IC", ascending=False)
    return df_ic


def run_ic_analysis(tickers: list):
    all_ics = []
    for t in tickers:
        try:
            ic_df = compute_ic(t)
            all_ics.append(ic_df["IC"].rename(t))
        except Exception as e:
            print(f"[IC] {t} failed: {e}")

    if not all_ics:
        print("[IC] No results")
        return

    combined = pd.concat(all_ics, axis=1)
    combined["mean_IC"] = combined.mean(axis=1)
    combined["abs_mean_IC"] = combined["mean_IC"].abs()
    combined = combined.sort_values("abs_mean_IC", ascending=False)

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  INFORMATION COEFFICIENT ANALYSIS  ({', '.join(tickers)})")
    print(f"  IC = Spearman correlation(feature, next-6h return)")
    print(f"  |IC| > 0.03 = useful  |IC| > 0.06 = strong")
    print(sep)
    print(f"  {'Feature':<30} {'Mean IC':>8}  {'Values'}")
    print(f"  {'-'*30} {'-'*8}  {'-'*30}")

    for feat, row in combined.head(40).iterrows():
        mean_ic = row["mean_IC"]
        ticker_ics = [f"{row[t]:+.3f}" for t in tickers if t in row.index and not pd.isna(row[t])]
        marker = "  *** STRONG" if abs(mean_ic) > 0.06 else ("  *  useful" if abs(mean_ic) > 0.03 else "")
        print(f"  {str(feat):<30} {mean_ic:>+8.4f}  {' '.join(ticker_ics)}{marker}")

    print(sep)

    # Flag near-zero features (IC ≈ 0 across all tickers = noise)
    noisy = combined[combined["abs_mean_IC"] < 0.005]
    if len(noisy) > 0:
        print(f"\n  Low-IC features (|IC| < 0.005, consider pruning): {len(noisy)}")
        print(f"  {list(noisy.index[:20])}")

    return combined


# ------------------------------------------------------------------ #
# Experiment runners                                                  #
# ------------------------------------------------------------------ #

def run_experiment(label: str, tickers: list, **kwargs) -> list:
    print(f"\n>>> {label}")
    rows = []
    for t in tickers:
        try:
            r = run_wf(t, **kwargs)
            rows.append(r)
            print(f"    {t}: CAGR={r['cagr']:+.1f}%  ret={r['total_return_pct']:+.1f}%  "
                  f"win={r['avg_win_rate']:.0f}%  trades={r['total_trades']}  dd={r['max_dd']:.1f}%")
        except Exception as e:
            rows.append({"ticker": t, "error": str(e), "cagr": 0.0})
            print(f"    {t}: ERROR {e}")
    return rows


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=SUPER_PORTFOLIO)
    parser.add_argument("--exp", nargs="+", default=["E1", "E2", "E3", "E4", "E5", "E6", "E7"])
    parser.add_argument("--ic", nargs="*", default=None,
                        help="Run IC analysis on these tickers (default: super portfolio)")
    args = parser.parse_args()

    tickers = args.tickers
    exps    = set(args.exp)
    results = {}

    print(f"\n{'#'*68}")
    print(f"  EDGE RESEARCH — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Tickers: {tickers}")
    print(f"  Experiments: {sorted(exps)}")
    print(f"{'#'*68}")

    # ---- E7: IC analysis first (pure research, no trading) --------
    if "E7" in exps:
        ic_tickers = args.ic if args.ic is not None else tickers[:4]  # first 4 to stay fast
        ic_results = run_ic_analysis(ic_tickers)
        if ic_results is not None:
            results["E7_ic"] = ic_results.head(30).to_dict()

    # ---- E1: Baseline — no ADX filter, let model decide via new features
    if "E1" in exps:
        rows = run_experiment(
            "E1 — No ADX filter (new features handle regime internally)",
            tickers,
            daily_confirm=False, allow_short=False, max_hold_bars=6, adx_filter_min=0,
        )
        print_table(rows, "E1 — No ADX Filter (baseline)")
        results["E1"] = rows

    # ---- E2: ADX=15 moderate filter (old-style gate)
    if "E2" in exps:
        rows = run_experiment(
            "E2 — ADX>=15 + hold=6 (moderate filter)",
            tickers,
            daily_confirm=False, allow_short=False, max_hold_bars=6, adx_filter_min=15,
        )
        print_table(rows, "E2 — ADX >= 15")
        results["E2"] = rows

    # ---- E3: daily_confirm=True, no ADX filter
    if "E3" in exps:
        rows = run_experiment(
            "E3 — daily_confirm=True, no ADX filter",
            tickers,
            daily_confirm=True, allow_short=False, max_hold_bars=6, adx_filter_min=0,
        )
        print_table(rows, "E3 — Daily Confirm, No ADX")
        results["E3"] = rows

    # ---- E4: Ensemble x3 + daily_confirm, no ADX
    if "E4" in exps:
        rows = run_experiment(
            "E4 — Ensemble x3 + daily_confirm, no ADX",
            tickers,
            daily_confirm=True, allow_short=False, max_hold_bars=6,
            n_ensemble=3, adx_filter_min=0,
        )
        print_table(rows, "E4 — Ensemble x3 + Daily Confirm")
        results["E4"] = rows

    # ---- E5: Higher confidence threshold (0.60), no ADX
    if "E5" in exps:
        rows = run_experiment(
            "E5 — Confidence >= 0.60, no ADX",
            tickers,
            daily_confirm=False, allow_short=False, max_hold_bars=6,
            min_confidence=0.60, adx_filter_min=0,
        )
        print_table(rows, "E5 — Confidence >= 0.60 (no ADX)")
        results["E5"] = rows

    # ---- E6: Extended hold 12 bars + daily_confirm, no ADX
    if "E6" in exps:
        rows = run_experiment(
            "E6 — Hold 12 bars + daily_confirm, no ADX",
            tickers,
            daily_confirm=True, allow_short=False, max_hold_bars=12, adx_filter_min=0,
        )
        print_table(rows, "E6 — Hold 12 Bars (no ADX)")
        results["E6"] = rows

    # ---- E8: VIX regime gate — only trade when ca_vts_regime != 2 (not panic)
    #          Implemented as: allow_short=True when backwardation, long-only when calm
    if "E8" in exps:
        rows = run_experiment(
            "E8 — ADX=20 gate (clean trending markets only)",
            tickers,
            daily_confirm=True, allow_short=False, max_hold_bars=6, adx_filter_min=20,
        )
        print_table(rows, "E8 — ADX >= 20 Clean Trending Markets")
        results["E8"] = rows

    # ---- Final summary: compare E1 vs best experiment ----------------
    if "E1" in results and len(results) > 1:
        print(f"\n{'='*68}")
        print("  DELTA vs BASELINE (E1)")
        print(f"{'='*68}")
        e1_by_ticker = {r["ticker"]: r["cagr"] for r in results["E1"] if "cagr" in r}
        for exp_key in [k for k in results if k not in ("E1", "E7_ic")]:
            exp_rows = results[exp_key]
            if not isinstance(exp_rows, list):
                continue
            deltas = []
            for r in exp_rows:
                if "cagr" in r and r["ticker"] in e1_by_ticker:
                    delta = r["cagr"] - e1_by_ticker[r["ticker"]]
                    deltas.append(delta)
            if deltas:
                print(f"  {exp_key}: avg CAGR delta = {np.mean(deltas):+.1f}pp  "
                      f"(min {min(deltas):+.1f}  max {max(deltas):+.1f})")
        print(f"{'='*68}")

    # Save results
    try:
        serializable = {}
        for k, v in results.items():
            if isinstance(v, list):
                serializable[k] = v
            # Skip DataFrame (E7)
        with open(RESULTS_FILE, "w") as f:
            json.dump({"timestamp": str(datetime.now()), "results": serializable}, f, indent=2)
        print(f"\nResults saved to {RESULTS_FILE}")
    except Exception as e:
        print(f"[warn] Could not save results: {e}")


if __name__ == "__main__":
    main()
