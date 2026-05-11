"""
Dry-run: validate threshold calibration logic end-to-end.
Run from rag_router/ directory.
"""
import sys, json
import numpy as np
sys.path.insert(0, ".")

print("=== DRY-RUN: Threshold Calibration Logic ===")
from experiments.train_router import load_labeled_data, calibrate_threshold
from router.pre_router import PreRouter

X, y, records = load_labeled_data(dataset_filter="pubmedqa")
print(f"Loaded {len(records)} pubmedqa samples")
print(f"Positive rate: {y.mean():.3f} ({y.mean()*100:.1f}%)")

# Train a quick logistic router in-memory
router = PreRouter("logistic")
report = router.train(X, y)
print(f"Train AUC: {report['train_auc']:.4f}")

# Run calibration
best_t = calibrate_threshold(router, X, y)
print(f"Calibrated threshold: {best_t:.4f}")

# Verify threshold is in expected range
assert 0.80 <= best_t <= 0.99, f"Threshold {best_t} outside [0.80, 0.99]"
print("[PASS] Threshold in expected range [0.80, 0.99]")

# Verify JSON was written
from pathlib import Path
from config import MODELS_DIR
p = Path(MODELS_DIR) / "pre_router_threshold.json"
assert p.exists(), f"{p} not found"
with open(p) as f:
    data = json.load(f)
print(f"[PASS] pre_router_threshold.json: {data}")

# Verify load_calibrated_threshold reads it correctly
from experiments.run_ablation import load_calibrated_threshold
t = load_calibrated_threshold()
assert t == data["threshold"], f"Mismatch: {t} vs {data['threshold']}"
print(f"[PASS] load_calibrated_threshold() returned: {t:.4f}")

# Verify router fires at calibrated threshold
router.threshold = best_t
proba_batch = router.predict_proba_batch(X)
full_fraction = float(np.mean(proba_batch < best_t))
print(f"Full LLM fraction at threshold={best_t:.4f}: {full_fraction:.1%}")
assert full_fraction > 0.0, "Router never fired even with calibrated threshold!"
print("[PASS] Router fires > 0% of queries")

# Verify validate module imports cleanly
from experiments.validate import validate_results
import pandas as pd
_test_df = pd.DataFrame([
    {"system": "always_cheap", "bertscore_f1": 0.72, "full_llm_fraction": 0.0},
    {"system": "always_full",  "bertscore_f1": 0.78, "full_llm_fraction": 1.0},
    {"system": "rag_router",   "bertscore_f1": 0.75, "full_llm_fraction": full_fraction},
])
validate_results(_test_df, positive_rate=float(y.mean()))

print()
print("=== DRY-RUN COMPLETE: All checks passed ===")
