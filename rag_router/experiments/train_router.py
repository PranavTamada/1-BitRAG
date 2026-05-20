"""
Step 2: Train Pre-Router and Post-Router
==========================================
Reads labeled data from experiments/collect_labels.py output,
trains both routing classifiers, logs metrics, and saves models.

Expected runtime: < 1 minute (sklearn training on structured features)
Expected output:
    - models/pre_router_logistic.pkl
    - models/pre_router_gbt.pkl
    - models/post_router.pkl
    - results/training_log.jsonl (appended)
    - Console: CV AUC scores, feature coefficients

Usage:
    python experiments/train_router.py
    python experiments/train_router.py --dataset healthcare_qa
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, TABLES_DIR, MODELS_DIR, RANDOM_STATE
from router.pre_router import PreRouter
from router.post_router import PostRouter
from utils.logger import log_training_event
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


def load_labeled_data(
    dataset_filter: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load labeled routing data and build feature matrices.

    Args:
        dataset_filter: if set, only load samples from this dataset.

    Returns:
        X:       (n, 18) combined feature matrix
        y:       (n,) binary labels
        records: raw record dicts for post-router feature extraction
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
            record = json.loads(line)
            if dataset_filter and record.get("dataset") != dataset_filter:
                continue
            records.append(record)

    if not records:
        raise ValueError(
            f"No records found"
            + (f" for dataset '{dataset_filter}'" if dataset_filter else "")
        )

    # Build feature matrix
    X_list = []
    y_list = []
    for rec in records:
        r_vec = feature_vector(rec["retrieval_features"])
        q_vec = query_feature_vector(rec["query_features"])
        combined = np.concatenate([r_vec, q_vec])
        X_list.append(combined)
        y_list.append(rec["cheap_succeeds"])

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float64)

    return X, y, records


def calibrate_threshold(router: "PreRouter", X: np.ndarray, y: np.ndarray) -> float:
    """Find the optimal decision threshold using F1 on the minority class.

    Uses a stratified 20% held-out split so the threshold is not overfit
    to the training data.  Returns the threshold and saves it to
    models/pre_router_threshold.json for downstream scripts.

    Strategy:
        - Sweep thresholds in [0.50, 0.99] (100 steps)
        - Pick t that maximises F1 for label=0 ("full model needed" minority)
        - This prevents the router from always predicting the majority class

    Args:
        router: trained PreRouter instance (logistic)
        X:      full feature matrix
        y:      full label array

    Returns:
        best_threshold (float)
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import f1_score

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, val_idx = next(sss.split(X, y))
    X_val = X[val_idx]
    y_val = y[val_idx]

    X_val_scaled = router.scaler.transform(X_val)
    proba_val = router.model.predict_proba(X_val_scaled)[:, 1]

    thresholds = np.linspace(0.5, 0.99, 100)
    best_t = max(
        thresholds,
        key=lambda t: f1_score(
            y_val, proba_val >= t, pos_label=0, zero_division=0
        )
    )

    positive_rate = float(y.mean())
    threshold_data = {
        "threshold": float(best_t),
        "positive_rate": positive_rate,
        "method": "platt_f1_minority",
    }
    threshold_path = Path(MODELS_DIR) / "pre_router_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump(threshold_data, f, indent=2)

    print(f"  Calibrated threshold: {best_t:.4f} (F1-minority on held-out 20%)")
    print(f"  Saved threshold metadata to {threshold_path}")
    return float(best_t)


def train_pre_routers(X: np.ndarray, y: np.ndarray) -> dict:
    """Train both logistic and GBT pre-routers.

    After training the logistic router, runs Platt-scaling calibration and
    finds the optimal minority-class F1 threshold, persisting it to
    models/pre_router_threshold.json.

    Returns:
        Combined report dict with results from both models.
    """
    reports = {}

    for model_type in ["logistic", "gbt"]:
        print(f"\n--- Training PreRouter ({model_type}) ---")
        router = PreRouter(model_type=model_type)
        report = router.train(X, y)

        # ── Threshold calibration for logistic router ────────────────────────
        if model_type == "logistic":
            best_t = calibrate_threshold(router, X, y)
            # Bake the calibrated threshold into the saved model
            router.threshold = best_t
            report["calibrated_threshold"] = best_t

        router.save()

        print(f"  CV AUC:  {report['cv_auc_mean']:.4f} +/- {report['cv_auc_std']:.4f}")
        print(f"  Train AUC: {report['train_auc']:.4f}")
        print(f"  N samples: {report['n_samples']}")
        print(f"  Positive rate: {report['positive_rate']:.3f}")

        if "feature_coefficients" in report:
            print("\n  Feature Coefficients (Logistic Regression):")
            print(f"  {'Feature':<25s} {'Coefficient':>12s}")
            print(f"  {'-'*25} {'-'*12}")
            for name, coef in sorted(
                report["feature_coefficients"].items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            ):
                print(f"  {name:<25s} {coef:>12.4f}")

        reports[model_type] = report
        log_training_event({
            "event": f"pre_router_{model_type}_trained",
            **report,
        })

    # ── SHAP feature importance for GBT ──────────────────────────────────
    if "gbt" in reports:
        try:
            shap_importance = compute_shap_importance(X, reports)
            if shap_importance:
                save_feature_importance_comparison(reports, shap_importance)
        except Exception as e:
            print(f"  [WARN] SHAP analysis failed: {e}")
            print(f"         Install shap: pip install shap")

    return reports


def train_post_router(records: list[dict], y: np.ndarray) -> dict:
    """Train the post-generation confidence estimator.

    Extracts output features from cheap LLM answers in the labeled data.

    Returns:
        Training report dict.
    """
    print("\n--- Training PostRouter ---")
    post_router = PostRouter()

    # Extract features from cheap LLM outputs
    X_post = []
    for rec in records:
        query = rec["query"]
        cheap_answer = rec.get("cheap_answer", "")
        if not cheap_answer or cheap_answer.startswith("[ERROR]"):
            cheap_answer = ""
        features = post_router.extract_features(query, cheap_answer)
        X_post.append(features)

    X_post = np.stack(X_post)
    report = post_router.train(X_post, y)
    post_router.save()

    print(f"  Train AUC: {report['post_router_train_auc']:.4f}")
    print(f"  N samples: {report['n_samples']}")
    print(f"  Positive rate: {report['positive_rate']:.3f}")

    if "feature_coefficients" in report:
        print("\n  Feature Coefficients (Post-Router):")
        for name, coef in sorted(
            report["feature_coefficients"].items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        ):
            print(f"  {name:<25s} {coef:>12.4f}")

    log_training_event({"event": "post_router_trained", **report})
    return report


def save_coefficient_table(pre_report: dict) -> None:
    """Save feature coefficients as CSV for the paper (Table 3).

    Research purpose:
        The coefficient table is a core result — it shows which retrieval
        geometry features most strongly predict query difficulty.
        Expect score_gap and top_score to have the largest positive
        coefficients (easy queries have sharp, high-scoring retrievals).
    """
    if "feature_coefficients" not in pre_report:
        return

    import pandas as pd
    rows = [
        {"Feature": name, "Coefficient": coef}
        for name, coef in pre_report["feature_coefficients"].items()
    ]
    df = pd.DataFrame(rows).sort_values(
        "Coefficient", key=abs, ascending=False
    )
    csv_path = TABLES_DIR / "feature_coefficients.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved coefficient table to {csv_path}")

    # Also save as LaTeX
    latex_path = TABLES_DIR / "feature_coefficients.tex"
    df.to_latex(latex_path, index=False, float_format="%.4f")
    print(f"  Saved LaTeX table to {latex_path}")


def compute_shap_importance(X: np.ndarray, reports: dict) -> dict | None:
    """Compute SHAP feature importance for the GBT model.

    Research purpose:
        Logistic regression gives interpretable coefficients directly,
        but GBT models are opaque. SHAP values provide a principled,
        model-agnostic feature importance measure that complements the
        logistic coefficients. Comparing the two confirms whether the
        same features matter regardless of model family.

    Args:
        X:       (n, 18) feature matrix (unscaled).
        reports: dict of training reports from train_pre_routers().

    Returns:
        dict mapping feature names to mean |SHAP| values, or None on failure.
    """
    try:
        import shap
    except ImportError:
        print("  [WARN] shap not installed. Run: pip install shap")
        return None

    # Load the trained GBT model
    gbt_router = PreRouter(model_type="gbt")
    gbt_router.load()

    # SHAP expects the underlying estimator, not the calibrated wrapper
    model = gbt_router.model
    if hasattr(model, 'calibrated_classifiers_'):
        model = model.calibrated_classifiers_[0].estimator

    X_scaled = gbt_router.scaler.transform(X)

    print("\n  Computing SHAP values for GBT model...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    importance = np.abs(shap_values).mean(axis=0)
    shap_dict = dict(zip(ALL_FEATURE_NAMES, importance.tolist()))

    print("\n  SHAP Feature Importance (GBT):")
    print(f"  {'Feature':<25s} {'Mean |SHAP|':>12s}")
    print(f"  {'-'*25} {'-'*12}")
    for name, val in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:<25s} {val:>12.4f}")

    return shap_dict


def save_feature_importance_comparison(
    reports: dict, shap_importance: dict
) -> None:
    """Save a combined table comparing logistic coefficients and SHAP values.

    Research purpose:
        Table 3 in the paper — shows both interpretability lenses side by side.
        If the same features rank highly in both, it validates that the feature
        space genuinely captures routing-relevant signal (not model artifacts).
    """
    import pandas as pd

    logistic_coefs = reports.get("logistic", {}).get("feature_coefficients", {})

    rows = []
    for name in ALL_FEATURE_NAMES:
        rows.append({
            "Feature": name,
            "Logistic_Coefficient": logistic_coefs.get(name, 0.0),
            "Logistic_Abs_Coefficient": abs(logistic_coefs.get(name, 0.0)),
            "GBT_SHAP_Importance": shap_importance.get(name, 0.0),
        })

    df = pd.DataFrame(rows).sort_values("GBT_SHAP_Importance", ascending=False)

    csv_path = TABLES_DIR / "feature_importance_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved feature importance comparison to {csv_path}")

    tex_path = TABLES_DIR / "feature_importance_comparison.tex"
    df[["Feature", "Logistic_Coefficient", "GBT_SHAP_Importance"]].to_latex(
        tex_path, index=False, float_format="%.4f"
    )
    print(f"  Saved LaTeX table to {tex_path}")


def train_all_routers(dataset_filter: str = None):
    """Train all routers (callable from run_all.py or CLI)."""
    X, y, records = load_labeled_data(dataset_filter=dataset_filter)
    print(f"\nLoaded {len(records)} labeled samples")
    print(f"  Label distribution: {y.mean():.1%} positive (cheap succeeds)")

    pre_reports = train_pre_routers(X, y)
    if "logistic" in pre_reports:
        save_coefficient_table(pre_reports["logistic"])
    train_post_router(records, y)

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train pre-router and post-router from labeled data."
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Filter labeled data to this dataset only.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 2: Router Training")
    print("=" * 60)

    # Load labeled data
    X, y, records = load_labeled_data(dataset_filter=args.dataset)
    print(f"\nLoaded {len(records)} labeled samples")
    print(f"  Label distribution: {y.mean():.1%} positive (cheap succeeds)")

    # Train pre-routers (logistic + GBT)
    pre_reports = train_pre_routers(X, y)

    # Save coefficient table for paper
    if "logistic" in pre_reports:
        save_coefficient_table(pre_reports["logistic"])

    # Train post-router
    post_report = train_post_router(records, y)

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
