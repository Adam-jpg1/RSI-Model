"""
Read all saved Optuna params_*.json files and update per_ticker_adx in config.py.
Run after optimization completes: python apply_optuna_params.py
"""
import json
import os
import re
import sys

MODELS_DIR = "models"
CONFIG_PATH = "config.py"


def load_all_params():
    results = {}
    for fname in sorted(os.listdir(MODELS_DIR)):
        if fname.startswith("params_") and fname.endswith(".json"):
            ticker = fname[len("params_"):-len(".json")]
            with open(os.path.join(MODELS_DIR, fname)) as f:
                data = json.load(f)
            results[ticker] = {
                "adx": data.get("adx_filter_min", 0.0),
                "score": data.get("best_score", 0.0),
            }
    return results


def update_config(params: dict):
    with open(CONFIG_PATH) as f:
        src = f.read()

    # Build the new per_ticker_adx dict block
    lines = []
    for ticker, p in sorted(params.items()):
        lines.append(f'        "{ticker}": {p["adx"]:.1f},')
    new_block = "field(default_factory=lambda: {\n" + "\n".join(lines) + "\n    })"

    # Replace the per_ticker_adx field_factory
    pattern = r'per_ticker_adx: Dict\[str, float\] = field\(default_factory=lambda: \{[^}]*\}\)'
    new_field = "per_ticker_adx: Dict[str, float] = " + new_block
    updated = re.sub(pattern, new_field, src, flags=re.DOTALL)

    if updated == src:
        print("[apply_optuna] WARNING: regex didn't match per_ticker_adx — config unchanged")
        return False

    with open(CONFIG_PATH, "w") as f:
        f.write(updated)
    return True


def main():
    params = load_all_params()
    if not params:
        print("No params_*.json files found in models/")
        sys.exit(1)

    print(f"\nFound Optuna params for {len(params)} tickers:\n")
    print(f"  {'TICKER':<8} {'ADX_MIN':>8}  {'SCORE':>8}")
    print(f"  {'-'*30}")
    for ticker, p in sorted(params.items(), key=lambda x: -x[1]["score"]):
        print(f"  {ticker:<8} {p['adx']:>8.1f}  {p['score']:>8.4f}")

    ok = update_config(params)
    if ok:
        print(f"\n[apply_optuna] config.py per_ticker_adx updated with {len(params)} tickers.")
    else:
        print("\n[apply_optuna] Manual update required.")


if __name__ == "__main__":
    main()
