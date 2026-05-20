"""
Error Analysis: Per-Query Routing Decisions
=============================================
Produces qualitative analysis of routing decisions for the paper.

Outputs:
    1. Scatter plot: score_gap (x) vs BERTScore gap (y), colored by routing decision
    2. CSV of misrouted queries (router sent to cheap, but full was much better)
    3. Feature distribution comparison: correctly vs incorrectly routed queries
    4. Summary statistics for paper narrative

Research purpose:
    Reviewers will ask: "What kinds of queries does the router get wrong?"
    This script answers that question with both visuals and statistics.

Usage:
    python experiments/error_analysis.py
    python experiments/error_analysis.py --dataset pubmedqa --threshold 0.85
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, RANDOM_STATE
from router.pre_router import PreRouter
from features.retrieval_features import feature_vector, RETRIEVAL_FEATURE_NAMES
from features.query_features import query_feature_vector, QUERY_FEATURE_NAMES

# Canonical feature names in training order
ALL_FEATURE_NAMES = [
    # Retrieval geometry features (10)
    "score_gap", "score_mean", "score_variance", "score_entropy",
    "top_score", "score_ratio", "low_score_fraction", "retrieval_hit",
    "bm25_dense_agreement", "context_density",
    # Query complexity features (8)
    "query_length", "query_entropy", "has_negation", "has_conditional",
    "question_count", "has_comparison", "avg_word_length",
    "named_entity_count",
]


def load_analysis_data(dataset_filter: str = "pubmedqa"):
    """Load labeled data and build analysis dataframe.

    Returns:
        DataFrame with columns: query, cheap_bertscore, full_bertscore,
        bertscore_gap, cheap_succeeds, + all 18 features + routing prediction.
    """
    path = DATA_DIR / "labeled_routing_data.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Labeled data not found at {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if dataset_filter and rec.get("dataset") != dataset_filter:
                continue
            records.append(rec)

    if not records:
        raise ValueError(f"No records for dataset '{dataset_filter}'")

    # Build feature matrix + metadata
    rows = []
    feature_vecs = []
    for rec in records:
        r_vec = feature_vector(rec["retrieval_features"])
        q_vec = query_feature_vector(rec["query_features"])
        combined = np.concatenate([r_vec, q_vec])
        feature_vecs.append(combined)

        row = {
            "query": rec["query"][:80],  # Truncate for display
            "cheap_bertscore": rec.get("cheap_bertscore", 0.0),
            "full_bertscore": rec.get("full_bertscore", 0.0),
            "bertscore_gap": rec.get("bertscore_gap", 0.0),
            "cheap_succeeds": rec.get("cheap_succeeds", 0),
        }
        # Add individual features
        for fname, fval in zip(ALL_FEATURE_NAMES, combined):
            row[fname] = float(fval)
        rows.append(row)

    X = np.stack(feature_vecs)
    df = pd.DataFrame(rows)

    return df, X


def add_routing_predictions(df: pd.DataFrame, X: np.ndarray, threshold: float = None):
    """Add pre-router predictions and confidence to the dataframe."""
    try:
        router = PreRouter("logistic")
        router.load()

        # Override threshold if provided
        if threshold is not None:
            router.threshold = threshold
        else:
            # Try to load calibrated threshold
            thresh_path = Path(MODELS_DIR) / "pre_router_threshold.json"
            if thresh_path.exists():
                with open(thresh_path, "r") as f:
                    data = json.load(f)
                router.threshold = float(data["threshold"])

        probas = router.predict_proba_batch(X)
        df["routing_confidence"] = probas
        df["routing_decision"] = ["cheap" if p >= router.threshold else "full" for p in probas]
        df["threshold_used"] = router.threshold

        # Classify routing outcomes
        df["routing_correct"] = (
            ((df["routing_decision"] == "cheap") & (df["cheap_succeeds"] == 1)) |
            ((df["routing_decision"] == "full") & (df["cheap_succeeds"] == 0))
        ).astype(int)

        print(f"  Router threshold: {router.threshold:.4f}")
        print(f"  Routing accuracy: {df['routing_correct'].mean():.1%}")
        return True

    except Exception as e:
        print(f"  [WARN] Could not load pre-router: {e}")
        df["routing_confidence"] = 0.5
        df["routing_decision"] = "unknown"
        df["routing_correct"] = 0
        return False


def plot_routing_scatter(df: pd.DataFrame, dataset_name: str):
    """Figure 4: Score gap vs BERTScore gap, colored by routing decision.

    This scatter plot shows:
    - X-axis: score_gap (retrieval geometry feature — top-1 vs top-2 retrieval score)
    - Y-axis: BERTScore gap (full - cheap) — how much quality is lost by using cheap
    - Color: routing decision (cheap = blue, full = red)
    - Correct decisions are solid; incorrect are hollow with red edge
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Score gap vs BERTScore gap, colored by routing decision
    ax1 = axes[0]
    cheap_mask = df["routing_decision"] == "cheap"
    full_mask = df["routing_decision"] == "full"

    ax1.scatter(
        df.loc[cheap_mask, "score_gap"], df.loc[cheap_mask, "bertscore_gap"],
        c="#2196f3", alpha=0.6, s=30, label="Routed → Cheap", edgecolors="none",
    )
    ax1.scatter(
        df.loc[full_mask, "score_gap"], df.loc[full_mask, "bertscore_gap"],
        c="#e65100", alpha=0.6, s=30, label="Routed → Full", edgecolors="none",
    )
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("Score Gap (Retrieval Feature)", fontsize=12)
    ax1.set_ylabel("BERTScore Gap (Full − Cheap)", fontsize=12)
    ax1.set_title("Routing Decisions vs Quality Gap", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.2)

    # Panel 2: Confidence vs BERTScore gap, colored by correctness
    ax2 = axes[1]
    correct = df["routing_correct"] == 1
    incorrect = df["routing_correct"] == 0

    ax2.scatter(
        df.loc[correct, "routing_confidence"], df.loc[correct, "bertscore_gap"],
        c="#4caf50", alpha=0.5, s=25, label="Correct Routing", edgecolors="none",
    )
    ax2.scatter(
        df.loc[incorrect, "routing_confidence"], df.loc[incorrect, "bertscore_gap"],
        c="#f44336", alpha=0.7, s=40, label="Incorrect Routing",
        edgecolors="black", linewidths=0.5,
    )
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Router Confidence P(cheap succeeds)", fontsize=12)
    ax2.set_ylabel("BERTScore Gap (Full − Cheap)", fontsize=12)
    ax2.set_title("Routing Correctness vs Confidence", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()

    png_path = FIGURES_DIR / f"error_analysis_{dataset_name}.png"
    pdf_path = FIGURES_DIR / f"error_analysis_{dataset_name}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved error analysis plot to {png_path}")
    print(f"  Saved error analysis plot to {pdf_path}")


def save_misrouted_queries(df: pd.DataFrame, dataset_name: str):
    """Save the most costly misrouted queries to CSV.

    These are queries where the router sent to cheap, but full was
    significantly better (large positive bertscore_gap).
    """
    # Misrouted: sent to cheap, but full was much better
    misrouted = df[
        (df["routing_decision"] == "cheap") &
        (df["bertscore_gap"] > 0.05)
    ].sort_values("bertscore_gap", ascending=False)

    if len(misrouted) > 0:
        cols = ["query", "cheap_bertscore", "full_bertscore", "bertscore_gap",
                "routing_confidence", "score_gap", "score_entropy", "query_entropy"]
        cols = [c for c in cols if c in misrouted.columns]
        csv_path = TABLES_DIR / f"misrouted_queries_{dataset_name}.csv"
        misrouted[cols].to_csv(csv_path, index=False)
        print(f"  Saved {len(misrouted)} misrouted queries to {csv_path}")
    else:
        print("  No misrouted queries found (gap > 0.05)")

    return misrouted


def feature_distribution_analysis(df: pd.DataFrame, dataset_name: str):
    """Compare feature distributions: correct vs incorrect routing.

    Saves a summary CSV showing mean feature values for correctly vs
    incorrectly routed queries — highlights which features the router
    struggles with.
    """
    if df["routing_correct"].nunique() < 2:
        print("  [SKIP] All routing decisions are the same class; no comparison possible.")
        return

    correct = df[df["routing_correct"] == 1]
    incorrect = df[df["routing_correct"] == 0]

    rows = []
    for fname in ALL_FEATURE_NAMES:
        if fname not in df.columns:
            continue
        rows.append({
            "Feature": fname,
            "Mean_Correct": correct[fname].mean(),
            "Std_Correct": correct[fname].std(),
            "Mean_Incorrect": incorrect[fname].mean(),
            "Std_Incorrect": incorrect[fname].std(),
            "Diff_Abs": abs(correct[fname].mean() - incorrect[fname].mean()),
        })

    df_feat = pd.DataFrame(rows).sort_values("Diff_Abs", ascending=False)
    csv_path = TABLES_DIR / f"feature_distribution_{dataset_name}.csv"
    df_feat.to_csv(csv_path, index=False)
    print(f"  Saved feature distribution analysis to {csv_path}")

    # Print top differentiating features
    print("\n  Top differentiating features (correct vs incorrect routing):")
    print(f"  {'Feature':<25s} {'Mean(Correct)':>14s} {'Mean(Incorrect)':>16s} {'|Diff|':>8s}")
    print(f"  {'-'*25} {'-'*14} {'-'*16} {'-'*8}")
    for _, row in df_feat.head(8).iterrows():
        print(f"  {row['Feature']:<25s} {row['Mean_Correct']:>14.4f} "
              f"{row['Mean_Incorrect']:>16.4f} {row['Diff_Abs']:>8.4f}")


def print_summary_statistics(df: pd.DataFrame, dataset_name: str, has_router: bool):
    """Print summary statistics for the paper narrative."""
    print(f"\n{'='*60}")
    print(f"Error Analysis Summary: {dataset_name}")
    print(f"{'='*60}")
    print(f"  Total queries:          {len(df)}")
    print(f"  Cheap BERTScore mean:   {df['cheap_bertscore'].mean():.4f} "
          f"(std={df['cheap_bertscore'].std():.4f})")
    print(f"  Full BERTScore mean:    {df['full_bertscore'].mean():.4f} "
          f"(std={df['full_bertscore'].std():.4f})")
    print(f"  Gap (full-cheap) mean:  {df['bertscore_gap'].mean():.4f}")
    print(f"  Gap > 0 (full better):  {(df['bertscore_gap'] > 0).sum()}/{len(df)} "
          f"({(df['bertscore_gap'] > 0).mean():.1%})")
    print(f"  Gap > 0.05:             {(df['bertscore_gap'] > 0.05).sum()}/{len(df)}")
    print(f"  Gap > 0.10:             {(df['bertscore_gap'] > 0.10).sum()}/{len(df)}")
    print(f"  cheap_succeeds rate:    {df['cheap_succeeds'].mean():.1%}")

    if has_router:
        print(f"\n  Routing decisions:")
        print(f"    → Cheap: {(df['routing_decision'] == 'cheap').sum()} "
              f"({(df['routing_decision'] == 'cheap').mean():.1%})")
        print(f"    → Full:  {(df['routing_decision'] == 'full').sum()} "
              f"({(df['routing_decision'] == 'full').mean():.1%})")
        print(f"  Routing accuracy:       {df['routing_correct'].mean():.1%}")

        # Break down errors
        false_cheap = ((df["routing_decision"] == "cheap") & (df["cheap_succeeds"] == 0)).sum()
        false_full = ((df["routing_decision"] == "full") & (df["cheap_succeeds"] == 1)).sum()
        print(f"  False cheap (missed):   {false_cheap} (should have gone to full)")
        print(f"  False full (wasteful):  {false_full} (cheap would have sufficed)")

    print(f"{'='*60}")


def run_error_analysis(dataset_name: str = "pubmedqa", threshold: float = None):
    """Run the full error analysis pipeline."""
    print("=" * 60)
    print(f"Error Analysis: {dataset_name}")
    print("=" * 60)

    # Load data
    df, X = load_analysis_data(dataset_name)
    print(f"  Loaded {len(df)} samples")

    # Add routing predictions
    has_router = add_routing_predictions(df, X, threshold)

    # Generate all outputs
    if has_router:
        plot_routing_scatter(df, dataset_name)
        save_misrouted_queries(df, dataset_name)
        feature_distribution_analysis(df, dataset_name)

    print_summary_statistics(df, dataset_name, has_router)

    # Save full analysis dataframe
    csv_path = TABLES_DIR / f"error_analysis_full_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved full analysis to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-query routing error analysis")
    parser.add_argument("--dataset", type=str, default="pubmedqa")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override routing threshold (default: use calibrated)")
    args = parser.parse_args()

    run_error_analysis(args.dataset, args.threshold)
