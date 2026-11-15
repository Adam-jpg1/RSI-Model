import json, os
params_dir = "models"
for fname in sorted(os.listdir(params_dir)):
    if fname.startswith("params_") and fname.endswith(".json"):
        ticker = fname[7:-5]
        with open(os.path.join(params_dir, fname)) as f:
            d = json.load(f)
        adx = d["adx_filter_min"]
        score = d["best_score"]
        print(f"{ticker}: ADX={adx:.1f}  score={score:.4f}")
