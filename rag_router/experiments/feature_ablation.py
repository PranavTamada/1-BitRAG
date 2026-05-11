"""
Step 6: Feature Ablation Study
================================
Trains the pre-router with different feature subsets to isolate the
contribution of retrieval geometry features vs query features.

Feature groups:
    A: Retrieval geometry only (10 features) — the core hypothesis
    B: Query complexity only (8 features)    — baseline comparison
    C: All features (18)                     — full model

Expected finding:
    If retrieval-only (A) performs close to all features (C), and much
    better than query-only (B), the paper's claim is validated:
    "retrieval geometry is the primary driver of routing accuracy."

Output:
    - results/tables/feature_ablation.csv
    - results/tables/feature_ablation.tex
    - results/figures/feature_ablation.png / .pdf
    - Console: comparison table

Usage:
    python experiments/feature_ablation.py
    python experiments/feature_ablation.py --dataset healthcare_qa
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, TABLES_DIR, FIGURES_DIR, RANDOM_STATE
from features.retrieval_features import feature_vector, RETRIEVAL_FEATURE_NAMES
from features.query_features import query_feature_vector, QUERY_FEATURE_NAMES
from utils.logger import log_training_event

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score


def load_feature_matrices(dataset_filter: str | None = None):
    """Load labeled data and return separate feature matrices.

    Returns:
        X_retrieval: (n, 10) retrieval geometry features only
        X_query:     (n, 8)  query complexity features only
        X_all:       (n, 18) combined features
        y:           (n,)    binary labels
    """
    path = DATA_DIR / "labeled_routing_data.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Labeled data not found at {path}. "
            "Run experiments/collect_labels.py first."
        )

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
        raise ValueError(f"No records found for dataset '{dataset_filter}'")

    X_retrieval = []
    X_query = []
    y_list = []

    for rec in records:
        r_vec = feature_vector(rec["retrieval_features"])
        q_vec = query_feature_vector(rec["query_features"])
        X_retrieval.append(r_vec)
        X_query.append(q_vec)
        y_list.append(rec["cheap_succeeds"])

    X_retrieval = np.stack(X_retrieval)
    X_query = np.stack(X_query)
    X_all = np.concatenate([X_retrieval, X_query], axis=1)
    y = np.array(y_list, dtype=np.float64)

    return X_retrieval, X_query, X_all, y


def train_and_evaluate(X, y, feature_group_name, cv_folds=None):
    """Train a logistic regression on a feature subset and return metrics.

    Args:
        X:                  feature matrix
        y:                  binary labels
        feature_group_name: name for logging
        cv_folds:           number of CV folds (auto if None)

    Returns:
        dict with cv_auc_mean, cv_auc_std, train_auc, n_features
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_min_class = int(min(y.sum(), len(y) - y.sum()))
    if cv_folds is None:
        cv_folds = max(2, min(5, n_min_class))

    model = LogisticRegression(
        C=0.1, max_iter=2000, random_state=RANDOM_STATE, penalty="l2"
    )

    skf = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE
    )
    cv_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="roc_auc")

    # Fit on full data for train AUC
    model.fit(X_scaled, y)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    train_auc = roc_auc_score(y, y_prob)

    result = {
        "feature_group": feature_group_name,
        "n_features": X.shape[1],
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "train_auc": float(train_auc),
        "cv_folds": cv_folds,
        "n_samples": len(y),
    }

    # Extract coefficients
    feature_names = []
    if feature_group_name == "retrieval_only":
        feature_names = list(RETRIEVAL_FEATURE_NAMES)
    elif feature_group_name == "query_only":
        feature_names = list(QUERY_FEATURE_NAMES)
    elif feature_group_name == "all_features":
        feature_names = list(RETRIEVAL_FEATURE_NAMES) + list(QUERY_FEATURE_NAMES)

    if feature_names and len(feature_names) == X.shape[1]:
        coefficients = dict(zip(feature_names, model.coef_[0].tolist()))
        result["coefficients"] = coefficients

    return result


def plot_feature_ablation(results: list[dict], dataset_name: str):
    """Generate the feature ablation bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = [r["feature_group"] for r in results]
    display_names = {
        "retrieval_only": "Retrieval Geometry\n(10 features)",
        "query_only": "Query Complexity\n(8 features)",
        "all_features": "All Features\n(18 features)",
    }
    labels = [display_names.get(g, g) for g in groups]
    cv_aucs = [r["cv_auc_mean"] for r in results]
    cv_stds = [r["cv_auc_std"] for r in results]

    colors = ["#1565c0", "#ff9800", "#2e7d32"]
    bars = ax.bar(labels, cv_aucs, yerr=cv_stds, color=colors,
                  edgecolor="black", linewidth=0.5, capsize=5)

    for bar, val, std in zip(bars, cv_aucs, cv_stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    ax.set_ylabel("CV AUC (Stratified K-Fold)", fontsize=12)
    ax.set_title(
        f"Feature Ablation: Which Features Drive Routing? ({dataset_name})",
        fontsize=13, fontweight="bold",
    )
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label="Random baseline")
    ax.legend(fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    png_path = FIGURES_DIR / f"feature_ablation_{dataset_name}.png"
    pdf_path = FIGURES_DIR / f"feature_ablation_{dataset_name}.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved feature ablation plot to {png_path}")


def run_feature_ablation(dataset_name: str = None):
    """Execute the feature ablation study."""
    print("=" * 60)
    print(f"Feature Ablation Study")
    if dataset_name:
        print(f"  Dataset filter: {dataset_name}")
    print("=" * 60)

    X_retrieval, X_query, X_all, y = load_feature_matrices(
        dataset_filter=dataset_name
    )
    print(f"  Loaded {len(y)} samples")
    print(f"  Positive rate: {y.mean():.1%}")
    print(f"  Feature dims: retrieval={X_retrieval.shape[1]}, "
          f"query={X_query.shape[1]}, all={X_all.shape[1]}")

    results = []

    # Group A: Retrieval geometry only
    print("\n--- Group A: Retrieval Geometry Only (10 features) ---")
    r_a = train_and_evaluate(X_retrieval, y, "retrieval_only")
    results.append(r_a)
    print(f"  CV AUC: {r_a['cv_auc_mean']:.4f} +/- {r_a['cv_auc_std']:.4f}")
    print(f"  Train AUC: {r_a['train_auc']:.4f}")

    # Group B: Query complexity only
    print("\n--- Group B: Query Complexity Only (8 features) ---")
    r_b = train_and_evaluate(X_query, y, "query_only")
    results.append(r_b)
    print(f"  CV AUC: {r_b['cv_auc_mean']:.4f} +/- {r_b['cv_auc_std']:.4f}")
    print(f"  Train AUC: {r_b['train_auc']:.4f}")

    # Group C: All features
    print("\n--- Group C: All Features (18 features) ---")
    r_c = train_and_evaluate(X_all, y, "all_features")
    results.append(r_c)
    print(f"  CV AUC: {r_c['cv_auc_mean']:.4f} +/- {r_c['cv_auc_std']:.4f}")
    print(f"  Train AUC: {r_c['train_auc']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("FEATURE ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Retrieval only: CV AUC = {r_a['cv_auc_mean']:.4f}")
    print(f"  Query only:     CV AUC = {r_b['cv_auc_mean']:.4f}")
    print(f"  All features:   CV AUC = {r_c['cv_auc_mean']:.4f}")

    delta = r_a["cv_auc_mean"] - r_b["cv_auc_mean"]
    if delta > 0:
        print(f"\n  Retrieval geometry beats query features by {delta:.4f} AUC")
        print(f"  --> SUPPORTS core hypothesis")
    else:
        print(f"\n  Query features beat retrieval geometry by {-delta:.4f} AUC")
        print(f"  --> Retrieval geometry alone is insufficient")

    # Save results
    df = pd.DataFrame(results)
    display_cols = ["feature_group", "n_features", "cv_auc_mean",
                    "cv_auc_std", "train_auc", "n_samples"]
    display_cols = [c for c in display_cols if c in df.columns]

    ds_tag = dataset_name or "all"
    csv_path = TABLES_DIR / f"feature_ablation_{ds_tag}.csv"
    df[display_cols].to_csv(csv_path, index=False)
    print(f"\n  Saved to {csv_path}")

    tex_path = TABLES_DIR / f"feature_ablation_{ds_tag}.tex"
    df[display_cols].to_latex(tex_path, index=False, float_format="%.4f")
    print(f"  Saved to {tex_path}")

    # Generate plot
    plot_feature_ablation(results, ds_tag)

    log_training_event({
        "event": "feature_ablation_complete",
        "dataset": dataset_name,
        "retrieval_cv_auc": r_a["cv_auc_mean"],
        "query_cv_auc": r_b["cv_auc_mean"],
        "all_cv_auc": r_c["cv_auc_mean"],
    })

    print(f"\n{'='*60}")
    print("Feature Ablation Complete")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature ablation: retrieval vs query vs all features"
    )
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    run_feature_ablation(args.dataset)
