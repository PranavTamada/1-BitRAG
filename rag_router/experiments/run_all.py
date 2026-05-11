"""
Master Pipeline: Run All Experiments After Label Collection
============================================================
This script runs Steps 2-6 sequentially:
  1. Re-train routers (using pubmedqa data from labeled_routing_data.jsonl)
  2. Ablation study on pubmedqa
  3. Feature ablation on pubmedqa
  4. Pareto curve on pubmedqa
  5. Cross-domain evaluation (pubmedqa only — healthcare_qa removed)

Data source: data/labeled_routing_data.jsonl
    Only pubmedqa samples are used; healthcare_qa entries are ignored.

Prerequisites:
  - labeled_routing_data.jsonl must contain pubmedqa samples (~300)
  - Ollama must be running with llama3.2:1b
  - GROQ_API_KEY must be set

Usage:
    python experiments/run_all.py
    python experiments/run_all.py --skip-training   # if routers already trained
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from config import DATA_DIR


def check_prerequisites():
    """Verify labeled data exists and count pubmedqa samples."""
    path = DATA_DIR / "labeled_routing_data.jsonl"
    if not path.exists():
        print("[ERROR] No labeled data found. Run collect_labels.py first.")
        sys.exit(1)

    import json
    dataset_counts = {}
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rec = json.loads(line.strip())
                    ds = rec.get("dataset", "unknown")
                    dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
                    total += 1
                except json.JSONDecodeError:
                    pass

    print(f"  Found {total} labeled samples across datasets: {dataset_counts}")

    # Only use pubmedqa for all experiments
    pubmedqa_count = dataset_counts.get("pubmedqa", 0)
    if pubmedqa_count == 0:
        print("[ERROR] No pubmedqa samples found in labeled_routing_data.jsonl.")
        print("        Run: python experiments/collect_labels.py --dataset pubmedqa")
        sys.exit(1)

    print(f"  Using pubmedqa samples: {pubmedqa_count} (healthcare_qa excluded)")
    return {"pubmedqa": pubmedqa_count}, pubmedqa_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all experiments after label collection.")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip router re-training (use existing models)")
    args = parser.parse_args()

    print("=" * 70)
    print("RAG-Router: Master Experiment Pipeline")
    print("  Dataset: pubmedqa (from labeled_routing_data.jsonl)")
    print("=" * 70)

    dataset_counts, total_samples = check_prerequisites()
    # Always only pubmedqa
    datasets_found = ["pubmedqa"]

    # ── Step 2: Train Routers ─────────────────────────────────────────
    if not args.skip_training:
        print(f"\n{'='*70}")
        print("STEP 2: Training Routers (pubmedqa data only)")
        print(f"{'='*70}")
        from experiments.train_router import train_all_routers
        train_all_routers(dataset_filter="pubmedqa")
    else:
        print("\n  [SKIP] Router training (using existing models)")

    # ── Step 3: Feature Ablation ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 3: Feature Ablation Study (pubmedqa)")
    print(f"{'='*70}")
    from experiments.feature_ablation import run_feature_ablation
    run_feature_ablation(dataset_name="pubmedqa")

    # ── Step 4: Ablation Study (Mode A vs B vs C) ─────────────────────
    print(f"\n{'='*70}")
    print("STEP 4: Ablation Study (Mode A vs B vs C) — pubmedqa")
    print(f"{'='*70}")
    from experiments.run_ablation import run_ablation
    try:
        run_ablation("pubmedqa", max_samples=dataset_counts["pubmedqa"])
    except Exception as e:
        print(f"  [ERROR] Ablation failed for pubmedqa: {e}")

    # ── Step 5: Pareto Curves ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 5: Pareto Curves (pubmedqa)")
    print(f"{'='*70}")
    from experiments.pareto_curve import run_pareto
    try:
        run_pareto("pubmedqa")
    except Exception as e:
        print(f"  [ERROR] Pareto failed for pubmedqa: {e}")

    # ── Step 6: Cross-Domain Evaluation ───────────────────────────────
    # Note: cross-domain step skipped — only one active dataset (pubmedqa)
    print(f"\n{'='*70}")
    print("STEP 6: Cross-Domain Evaluation")
    print("  [SKIP] Only pubmedqa is active. Add natural_questions labels")
    print("         to labeled_routing_data.jsonl for cross-domain analysis.")
    print(f"{'='*70}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"  Total pubmedqa samples used: {total_samples}")
    print(f"  Check results/ for all outputs.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

