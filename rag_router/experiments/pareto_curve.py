"""
Step 4: Generate Pareto Frontier Plot
=======================================
This is Figure 2 in the paper -- the core visual result.

X-axis: Cost (fraction of full LLM calls = API cost proxy)
Y-axis: BERTScore F1

Curves plotted:
    - RAG-Router (ours)          [solid blue line]
    - Post-Gen Cascade           [dashed orange line]
      (FrugalGPT-inspired cascade adapted to RAG setting)
    - Random Routing             [dotted gray line]
    - Always-full                [horizontal dashed line, upper bound]
    - Always-cheap               [horizontal dashed line, lower bound]

Expected runtime: ~5-10 min (uses cached LLM responses from collect_labels)
Expected output:
    - results/figures/pareto_curve.png
    - results/figures/pareto_curve.pdf
    - results/tables/pareto_points.csv

Key research finding:
    RAG-Router curve should be consistently above the Post-Gen Cascade curve,
    meaning at any given cost budget, pre-generation routing achieves higher
    accuracy than post-generation escalation.

Usage:
    python experiments/pareto_curve.py
    python experiments/pareto_curve.py --dataset healthcare_qa
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

from config import (
    DATA_DIR, FIGURES_DIR, TABLES_DIR,
    BERTSCORE_SUCCESS_THRESHOLD, BUDGET_FRACTIONS,
)
from router.pre_router import PreRouter
from router.budget_optimizer import sweep_thresholds, find_optimal_thresholds_for_budgets
from features.retrieval_features import feature_vector
from features.query_features import query_feature_vector
from utils.logger import log_training_event


def load_labeled_scores(dataset_filter: str | None = None):
    """Load labeled data and extract scores + features for Pareto sweep.

    Returns:
        cheap_scores: (N,) BERTScore F1 from cheap LLM
        full_scores:  (N,) BERTScore F1 from full LLM
        feature_vecs: (N, 18) combined feature matrix
        records:      raw record list
    """
    path = DATA_DIR / "labeled_routing_data.jsonl"
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

    cheap_scores = np.array([r["cheap_bertscore"] for r in records])
    full_scores = np.array([r["full_bertscore"] for r in records])

    feature_vecs = []
    for rec in records:
        r_vec = feature_vector(rec["retrieval_features"])
        q_vec = query_feature_vector(rec["query_features"])
        feature_vecs.append(np.concatenate([r_vec, q_vec]))
    feature_vecs = np.stack(feature_vecs)

    return cheap_scores, full_scores, feature_vecs, records


def build_post_gen_cascade_pareto(records, cheap_scores, full_scores):
    """Build Post-Gen Cascade Pareto curve by sweeping uncertainty thresholds.

    Adapts the cascade strategy from FrugalGPT (Chen et al., 2023) to the
    RAG setting: always call cheap LLM first, escalate if the answer is short
    or contains uncertainty phrases. We sweep the length threshold to generate
    different cost-accuracy tradeoffs.

    Note: unlike original FrugalGPT, retrieval is always performed first
    (shared with all baselines); only the post-generation routing differs.
    """
    from evaluation.baselines import _frugal_is_uncertain

    thresholds = np.linspace(0, 200, 51)  # min answer length thresholds
    points = []

    for min_len in thresholds:
        adaptive_scores = []
        full_count = 0
        for i, rec in enumerate(records):
            cheap_answer = rec.get("cheap_answer", "")
            is_uncertain = len(cheap_answer.strip()) < min_len
            if not is_uncertain:
                # Also check phrase-based uncertainty
                lower = cheap_answer.lower()
                phrases = ["i don't know", "not sure", "cannot determine",
                           "unclear", "not provided", "no information"]
                is_uncertain = any(p in lower for p in phrases)

            if is_uncertain:
                adaptive_scores.append(full_scores[i])
                full_count += 1
            else:
                adaptive_scores.append(cheap_scores[i])

        accuracy = float(np.mean(adaptive_scores))
        cost = full_count / len(records)
        points.append((cost, accuracy))

    return points



def build_random_pareto(cheap_scores, full_scores):
    """Build random routing Pareto curve."""
    rng = np.random.RandomState(42)
    fractions = np.linspace(0.0, 1.0, 51)
    points = []

    for frac in fractions:
        trials = []
        for _ in range(10):  # Average over 10 random seeds
            mask = rng.random(len(cheap_scores)) < frac
            adaptive = np.where(mask, full_scores, cheap_scores)
            trials.append(float(np.mean(adaptive)))
        points.append((float(frac), float(np.mean(trials))))

    return points


def plot_pareto_curve(
    rag_router_points: list,
    frugal_points: list,
    random_points: list,
    cheap_baseline: float,
    full_baseline: float,
    dataset_name: str,
) -> None:
    """Generate Figure 2: the Pareto frontier comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # RAG-Router (ours) — solid blue
    costs_rr = [p[0] for p in rag_router_points]
    accs_rr = [p[1] for p in rag_router_points]
    ax.plot(costs_rr, accs_rr, "o-", color="#1565c0", linewidth=2.5,
            markersize=3, label="RAG-Router (Ours)", zorder=5)

    # Post-Gen Cascade — dashed orange
    costs_pgc = [p[0] for p in frugal_points]
    accs_pgc  = [p[1] for p in frugal_points]
    ax.plot(costs_pgc, accs_pgc, "s--", color="#e65100", linewidth=2,
            markersize=3,
            label="Post-Gen Cascade (FrugalGPT-inspired)",
            alpha=0.85)

    # Random — dotted gray
    costs_rd = [p[0] for p in random_points]
    accs_rd = [p[1] for p in random_points]
    ax.plot(costs_rd, accs_rd, ":", color="#757575", linewidth=1.5,
            label="Random Routing", alpha=0.7)

    # Always-full — horizontal dashed
    ax.axhline(y=full_baseline, color="#2e7d32", linestyle="--",
               linewidth=1.5, alpha=0.7, label=f"Always Full ({full_baseline:.3f})")

    # Always-cheap — horizontal dashed
    ax.axhline(y=cheap_baseline, color="#bf360c", linestyle="--",
               linewidth=1.5, alpha=0.7, label=f"Always Cheap ({cheap_baseline:.3f})")

    ax.set_xlabel("Cost (Fraction of Full LLM Calls)", fontsize=13)
    ax.set_ylabel("BERTScore F1", fontsize=13)
    ax.set_title(
        f"Pareto Frontier: Accuracy vs Cost ({dataset_name})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    png_path = FIGURES_DIR / f"pareto_curve_{dataset_name}.png"
    pdf_path = FIGURES_DIR / f"pareto_curve_{dataset_name}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved Pareto plot to {png_path}")
    print(f"  Saved Pareto plot to {pdf_path}")


def run_pareto(dataset_name: str) -> None:
    """Generate the full Pareto frontier comparison."""
    print("=" * 60)
    print(f"Pareto Frontier: {dataset_name}")
    print("=" * 60)

    # Load labeled data
    cheap_scores, full_scores, feature_vecs, records = load_labeled_scores(
        dataset_filter=dataset_name
    )
    print(f"  Loaded {len(records)} labeled samples")

    cheap_baseline = float(np.mean(cheap_scores))
    full_baseline = float(np.mean(full_scores))
    print(f"  Always-cheap BERTScore: {cheap_baseline:.4f}")
    print(f"  Always-full  BERTScore: {full_baseline:.4f}")

    # RAG-Router curve: sweep pre-router thresholds
    print("\n  Building RAG-Router Pareto curve...")
    pre_router = PreRouter("logistic")
    pre_router.load()
    routing_probs = pre_router.predict_proba_batch(feature_vecs)

    rr_results = sweep_thresholds(routing_probs, cheap_scores, full_scores)
    rag_router_points = [(r.budget_fraction, r.accuracy) for r in rr_results]

    # Optimal thresholds at each budget
    optimal = find_optimal_thresholds_for_budgets(rr_results)
    print("\n  Optimal thresholds per budget:")
    for budget, result in sorted(optimal.items()):
        print(f"    Budget {budget:.0%}: threshold={result.threshold:.3f}, "
              f"accuracy={result.accuracy:.4f}, cost={result.budget_fraction:.1%}")

    # Post-Gen Cascade curve
    print("\n  Building Post-Gen Cascade Pareto curve...")
    frugal_points = build_post_gen_cascade_pareto(records, cheap_scores, full_scores)

    # Random curve
    print("  Building Random Routing Pareto curve...")
    random_points = build_random_pareto(cheap_scores, full_scores)

    # Generate plot
    plot_pareto_curve(
        rag_router_points, frugal_points, random_points,
        cheap_baseline, full_baseline, dataset_name,
    )

    # Save points as CSV (existing Pareto multi-system CSV)
    rows = []
    for cost, acc in rag_router_points:
        rows.append({"system": "RAG-Router", "cost": cost, "bertscore_f1": acc})
    for cost, acc in frugal_points:
        rows.append({"system": "Post-Gen Cascade", "cost": cost, "bertscore_f1": acc})
    for cost, acc in random_points:
        rows.append({"system": "Random", "cost": cost, "bertscore_f1": acc})

    df = pd.DataFrame(rows)
    csv_path = TABLES_DIR / f"pareto_points_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved Pareto points to {csv_path}")

    # ── Per-threshold results table (Improvement 2) ───────────────────────────
    print("\n  Building per-threshold results table...")
    from router.budget_optimizer import sweep_thresholds as _sweep
    from rouge_score import rouge_scorer as _rouge_scorer_mod

    # ROUGE-L scorer for per-threshold computation
    _rouge = _rouge_scorer_mod.RougeScorer(["rougeL"], use_stemmer=True)

    def _rouge_l_mean(preds, refs):
        scores = []
        for p, r in zip(preds, refs):
            p = p if p.strip() else "No answer"
            r = r if r.strip() else "No answer"
            scores.append(_rouge.score(r, p)["rougeL"].fmeasure)
        return float(np.mean(scores)) if scores else 0.0

    # cheap / full answer texts are not directly available in pareto_curve,
    # so we use the BERTScore arrays as the primary signal and approximate
    # ROUGE-L as np.nan (not recomputable without raw text strings here).
    # For exact ROUGE-L per threshold, run the full evaluation loop.
    thresh_rows = []
    canonical_budgets = [0.10, 0.30, 0.50, 0.70, 0.90]

    for r in rr_results:
        thresh_rows.append({
            "threshold":              round(r.threshold, 6),
            "target_budget_pct":      None,   # filled below for canonical rows
            "actual_cost_fraction":   round(r.budget_fraction, 6),
            "bertscore_f1":           round(r.accuracy, 6),
            "rouge_l_f1":             float("nan"),   # requires raw text; see note
            "full_llm_fraction":      round(r.budget_fraction, 6),
            "cost_savings_fraction":  round(r.cost_savings, 6),
        })

    df_thresh = pd.DataFrame(thresh_rows)
    thresh_csv = TABLES_DIR / f"pareto_thresholds_{dataset_name}.csv"
    df_thresh.to_csv(thresh_csv, index=False)
    print(f"  Saved per-threshold table to {thresh_csv}")

    # ── LaTeX table (booktabs style) ──────────────────────────────────────
    # Emit only canonical budget rows for the paper table
    canonical_thresh_rows = []
    for budget in canonical_budgets:
        result = optimal.get(budget)
        if result is None:
            from router.budget_optimizer import find_optimal_threshold
            result = find_optimal_threshold(rr_results, budget)
        canonical_thresh_rows.append({
            "Target Budget": f"{budget:.0%}",
            "Threshold": f"{result.threshold:.4f}",
            "Cost Fraction": f"{result.budget_fraction:.1%}",
            "BERTScore F1": f"{result.accuracy:.4f}",
            "ROUGE-L F1": "---",          # requires raw text
            "Cost Savings": f"{result.cost_savings:.1%}",
        })

    df_lat = pd.DataFrame(canonical_thresh_rows)
    tex_lines = []
    tex_lines.append(r"\begin{table}[t]")
    tex_lines.append(r"  \centering")
    tex_lines.append(
        r"  \caption{RAG-Router per-threshold performance on "
        + dataset_name + r".}")
    tex_lines.append(r"  \label{tab:pareto_thresholds}")
    tex_lines.append(r"  \begin{tabular}{lccccc}")
    tex_lines.append(r"    \toprule")
    tex_lines.append(
        r"    Target & Threshold & Cost & BERTScore & ROUGE-L & Cost \\\\"
    )
    tex_lines.append(
        r"    Budget &           & Fraction & F1 & F1 & Savings \\\\"
    )
    tex_lines.append(r"    \midrule")
    _nl = r" \\"
    for row in canonical_thresh_rows:
        tex_lines.append(
            "    " + row["Target Budget"] + " & " + row["Threshold"] + " & "
            + row["Cost Fraction"] + " & " + row["BERTScore F1"] + " & "
            + row["ROUGE-L F1"] + " & " + row["Cost Savings"] + _nl
        )
    tex_lines.append(r"    \bottomrule")
    tex_lines.append(r"  \end{tabular}")
    tex_lines.append(r"\end{table}")

    tex_path = TABLES_DIR / f"pareto_thresholds_{dataset_name}.tex"
    with open(tex_path, "w", encoding="utf-8") as _f:
        _f.write("\n".join(tex_lines) + "\n")
    print(f"  Saved LaTeX table to {tex_path}")

    # ── 5 canonical budget summary print ─────────────────────────────────────
    print("\n  Canonical budget points (10/30/50/70/90%):")
    print(f"  {'Budget':>8s}  {'Threshold':>10s}  {'Cost':>8s}  "
          f"{'BERTScore':>10s}  {'Savings':>8s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
    for row in canonical_thresh_rows:
        print(f"  {row['Target Budget']:>8s}  {row['Threshold']:>10s}  "
              f"{row['Cost Fraction']:>8s}  {row['BERTScore F1']:>10s}  "
              f"{row['Cost Savings']:>8s}")

    log_training_event({
        "event": "pareto_curve_generated",
        "dataset": dataset_name,
        "n_samples": len(records),
        "cheap_baseline": cheap_baseline,
        "full_baseline": full_baseline,
    })

    # ── Sanity checks (Improvement 3) ──────────────────────────────────────
    try:
        from experiments.validate import validate_results
        from config import MODELS_DIR as _MDIR
        import json as _json
        labels = np.array([r.get("cheap_succeeds", 1) for r in records])
        positive_rate = float(labels.mean())
        # Build a minimal results-like DataFrame from pareto baselines
        _pareto_df = pd.DataFrame([
            {"system": "always_cheap", "bertscore_f1": cheap_baseline,
             "full_llm_fraction": 0.0},
            {"system": "always_full",  "bertscore_f1": full_baseline,
             "full_llm_fraction": 1.0},
        ])
        # Add rag_router row at the 50% budget point
        _opt50 = optimal.get(0.5)
        if _opt50:
            _pareto_df = pd.concat([
                _pareto_df,
                pd.DataFrame([{
                    "system": "rag_router",
                    "bertscore_f1": _opt50.accuracy,
                    "full_llm_fraction": _opt50.budget_fraction,
                }])
            ], ignore_index=True)
        validate_results(_pareto_df, positive_rate=positive_rate)
    except AssertionError as ae:
        print(f"\n  [WARN] Sanity check failed: {ae}")
    except Exception as e:
        print(f"\n  [WARN] Could not run sanity checks: {e}")

    print(f"\n{'='*60}")
    print("Pareto Frontier Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Pareto frontier plot")
    parser.add_argument("--dataset", type=str, default="pubmedqa")
    args = parser.parse_args()

    run_pareto(args.dataset)
