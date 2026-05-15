"""
Step 5: Cross-Domain Evaluation
=================================
Evaluates all baselines across datasets to test whether
retrieval geometry features generalise beyond the training domain.
Data source: data/labeled_routing_data.jsonl  (pubmedqa samples only)

Expected runtime: 30-60 min (depends on dataset size + LLM caching)
Expected output:
    - results/tables/cross_domain_results.csv
    - results/tables/cross_domain_results.tex
    - Console: the final multi-dataset result table (Table 1 in the paper)

Key research question:
    Does the pre-router trained on PubMedQA labeled data generalise to
    Natural Questions without retraining?
    If yes: retrieval geometry features are domain-general.
    If no: features capture domain-specific patterns (still publishable
    as a finding, but weaker claim).

Usage:
    python experiments/cross_domain.py
    python experiments/cross_domain.py --max-samples 50
    python experiments/cross_domain.py --datasets pubmedqa
"""

import sys
import os
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASETS, TABLES_DIR
from evaluation.evaluate import evaluate_baselines
from utils.logger import log_training_event


CROSS_DOMAIN_BASELINES = [
    "always_cheap",
    "always_full",
    "random_routing",
    "frugal_gpt",
    "pre_only",
    "rag_router",
]


def run_cross_domain(
    datasets: list[str] | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Run all baselines across all specified datasets.

    Returns:
        Combined DataFrame with results for all datasets.
    """
    if datasets is None:
        datasets = DATASETS

    print("=" * 80)
    print("Cross-Domain Evaluation")
    print("=" * 80)
    print(f"  Datasets:  {datasets}")
    print(f"  Baselines: {CROSS_DOMAIN_BASELINES}")
    if max_samples:
        print(f"  Max samples per dataset: {max_samples}")

    all_dfs = []

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*60}")

        try:
            df = evaluate_baselines(
                dataset_name=dataset_name,
                baseline_names=CROSS_DOMAIN_BASELINES,
                max_samples=max_samples,
            )
            all_dfs.append(df)
        except Exception as e:
            print(f"\n  [ERROR] Failed on {dataset_name}: {e}")
            print(f"  Skipping this dataset.\n")
            continue

    if not all_dfs:
        print("\n[ERROR] No datasets completed successfully.")
        return pd.DataFrame()

    # Combine all results
    combined = pd.concat(all_dfs, ignore_index=True)

    # Display the final table
    display_cols = [
        "dataset", "system", "bertscore_f1", "rouge_l_f1",
        "full_llm_fraction", "cost_savings_fraction", "mean_latency_ms",
    ]
    display_cols = [c for c in display_cols if c in combined.columns]
    df_display = combined[display_cols]

    print(f"\n{'='*80}")
    print("CROSS-DOMAIN RESULTS (Table 1)")
    print(f"{'='*80}")
    print(df_display.to_string(index=False))

    # Save combined results
    csv_path = TABLES_DIR / "cross_domain_results.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    tex_path = TABLES_DIR / "cross_domain_results.tex"
    df_display.to_latex(tex_path, index=False, float_format="%.4f")
    print(f"Saved to {tex_path}")

    # Log
    log_training_event({
        "event": "cross_domain_evaluation_complete",
        "datasets": datasets,
        "n_total_rows": len(combined),
    })

    # Summary: does RAG-Router beat FrugalGPT on each dataset?
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    for ds in datasets:
        ds_data = combined[combined["dataset"] == ds]
        rr = ds_data[ds_data["system"] == "rag_router"]
        fg = ds_data[ds_data["system"] == "frugal_gpt"]
        af = ds_data[ds_data["system"] == "always_full"]

        if rr.empty or fg.empty or af.empty:
            continue

        rr_f1 = rr["bertscore_f1"].values[0]
        fg_f1 = fg["bertscore_f1"].values[0]
        af_f1 = af["bertscore_f1"].values[0]
        rr_cost = rr["full_llm_fraction"].values[0]

        retention = rr_f1 / af_f1 if af_f1 > 0 else 0
        beats_frugal = rr_f1 > fg_f1

        print(f"\n  {ds}:")
        print(f"    RAG-Router BERTScore F1:  {rr_f1:.4f}")
        print(f"    Always-Full BERTScore F1: {af_f1:.4f}")
        print(f"    Accuracy retention:       {retention:.1%} of always-full")
        print(f"    Full LLM usage:           {rr_cost:.1%}")
        print(f"    Beats FrugalGPT:          {'YES' if beats_frugal else 'NO'} "
              f"({rr_f1:.4f} vs {fg_f1:.4f})")

    print(f"\n{'='*80}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-domain evaluation across all datasets."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        choices=["natural_questions", "pubmedqa"],
        help="Datasets to evaluate. Defaults to all non-healthcare datasets.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_cross_domain(datasets=args.datasets, max_samples=args.max_samples)
