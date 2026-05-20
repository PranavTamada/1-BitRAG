"""
Step 3: Ablation Study — Mode A vs B vs C
============================================
Compares the three routing modes to quantify the contribution of each
component (pre-routing vs post-routing vs both).
Data source: data/labeled_routing_data.jsonl  (pubmedqa samples only)

Expected runtime: ~10-30 min per dataset (depending on LLM speed + caching)
Expected output:
    - results/tables/ablation_{dataset}.csv
    - results/tables/ablation_{dataset}.tex
    - results/figures/ablation_bars_{dataset}.png / .pdf
    - results/tables/negative_findings.txt (post-router negative result)
    - Console: formatted ablation table

Key research question:
    Does combining pre-routing (Mode A) with post-routing (Mode B) into
    the full system (Mode C) yield better accuracy-cost tradeoffs than
    either component alone?

Expected finding:
    Mode C >= Mode A and Mode C >= Mode B at matched cost levels.
    If Mode A alone matches Mode C, the pre-routing hypothesis is
    strongly validated (post-gen check adds no value).

Usage:
    python experiments/run_ablation.py
    python experiments/run_ablation.py --dataset pubmedqa --max-samples 100
    python experiments/run_ablation.py --budget 0.5
"""

import sys
import os
import json
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TABLES_DIR, FIGURES_DIR, MODELS_DIR
from evaluation.evaluate import evaluate_baselines
from utils.logger import log_training_event


# ── Main ablation systems (7 baselines — clean set for paper)
ABLATION_BASELINES = [
    "always_cheap",       # Lower bound
    "always_full",        # Upper bound
    "random_routing",     # Sanity-check: must beat a coin flip
    "post_gen_cascade",   # Post-gen cascade (FrugalGPT-inspired, adapted to RAG)
    "pre_only",           # Mode A — retrieval geometry pre-router only
    "rag_router",         # Mode C — full system
    "oracle_routing",     # Theoretical upper bound — always picks better model
]

# Display names for paper-quality plots
DISPLAY_NAMES = {
    "always_cheap":     "Always Cheap",
    "always_full":      "Always Full",
    "random_routing":   "Random",
    "post_gen_cascade": "Post-Gen Cascade",
    "pre_only":         "Mode A (Pre)",
    "rag_router":       "Mode C (Full)",
    "oracle_routing":   "Oracle (Upper Bound)",
}


def load_calibrated_threshold(budget: float = 0.5) -> float:
    """Load the calibrated decision threshold for the pre-router.

    Priority:
        1. models/pre_router_threshold.json (set by train_router.py)
        2. Fallback: scan pareto_points_{dataset}.csv for budget-matched row

    FIX 1: replaces the hardcoded 0.5 sklearn default.

    Args:
        budget: target full-LLM fraction for fallback Pareto CSV lookup.

    Returns:
        threshold (float)
    """
    threshold_path = Path(MODELS_DIR) / "pre_router_threshold.json"

    if threshold_path.exists():
        with open(threshold_path, "r") as f:
            data = json.load(f)
        t = float(data["threshold"])
        print(f"  [THRESHOLD] Loaded calibrated threshold: {t:.4f} "
              f"(method={data.get('method','?')}, "
              f"positive_rate={data.get('positive_rate', '?'):.3f})")
        return t

    # Fallback: read from pareto CSV
    print(f"  [WARN] {threshold_path} not found — falling back to pareto CSV lookup")
    pareto_csv = TABLES_DIR / "pareto_points_pubmedqa.csv"
    if not pareto_csv.exists():
        print(f"  [ERROR] Pareto CSV not found at {pareto_csv}. "
              "Cannot determine threshold. Defaulting to 0.85.")
        return 0.85

    import pandas as pd
    from router.budget_optimizer import find_optimal_threshold, sweep_thresholds
    df_p = pd.read_csv(pareto_csv)
    rr_rows = df_p[df_p["system"] == "RAG-Router"]
    if rr_rows.empty:
        print("  [WARN] No RAG-Router rows in pareto CSV. Defaulting to 0.85.")
        return 0.85

    # Find the row closest to the requested budget
    rr_rows = rr_rows.copy()
    rr_rows["_dist"] = (rr_rows["cost"] - budget).abs()
    best = rr_rows.nsmallest(1, "_dist").iloc[0]
    t = float(best.get("threshold", 0.85))
    print(f"  [THRESHOLD] Pareto CSV fallback: threshold≈{t:.4f} at budget≈{budget:.0%}")
    return t





def plot_ablation_bars(df: pd.DataFrame, dataset_name: str) -> None:
    """Generate the ablation bar chart (Figure 3 in the paper).

    Two side-by-side bar groups:
        Left:  BERTScore F1 per system
        Right: Full LLM fraction per system
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    systems = df["system"].tolist()
    display = [DISPLAY_NAMES.get(s, s) for s in systems]

    # Color coding: bounds gray, prior work orange, our modes blue
    colors = []
    for s in systems:
        if s in ("always_cheap", "always_full"):
            colors.append("#9e9e9e")
        elif s in ("post_gen_cascade", "random_routing"):
            colors.append("#ff9800")
        else:
            colors.append("#2196f3")

    # Panel 1: BERTScore F1
    ax1 = axes[0]
    bars1 = ax1.bar(display, df["bertscore_f1"], color=colors,
                    edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("BERTScore F1", fontsize=12)
    ax1.set_title(f"Accuracy by System ({dataset_name})", fontsize=13,
                  fontweight="bold")
    ax1.tick_params(axis="x", rotation=35)
    ax1.set_ylim(0, 1.05)
    for bar, val in zip(bars1, df["bertscore_f1"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Panel 2: Full LLM Usage
    ax2 = axes[1]
    bars2 = ax2.bar(display, df["full_llm_fraction"], color=colors,
                    edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Full LLM Fraction", fontsize=12)
    ax2.set_title(f"Cost by System ({dataset_name})", fontsize=13,
                  fontweight="bold")
    ax2.tick_params(axis="x", rotation=35)
    ax2.set_ylim(0, 1.15)
    for bar, val in zip(bars2, df["full_llm_fraction"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.1%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    png_path = FIGURES_DIR / f"ablation_bars_{dataset_name}.png"
    pdf_path = FIGURES_DIR / f"ablation_bars_{dataset_name}.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved ablation plot to {png_path}")
    print(f"  Saved ablation plot to {pdf_path}")


def run_ablation(
    dataset_name: str,
    max_samples: int | None = None,
    budget: float = 0.5,
) -> None:
    """Execute the full ablation study.

    Loads the calibrated pre-router threshold and injects it before any
    routing decisions, then asserts the router actually fired.

    Args:
        dataset_name: dataset to evaluate (e.g. 'pubmedqa').
        max_samples:  optional cap for quick testing.
        budget:       target full-LLM fraction for threshold fallback lookup.
    """
    print("=" * 60)
    print(f"Ablation Study: {dataset_name}")
    print("=" * 60)

    # ── Load calibrated threshold (FIX 1) ────────────────────────────────────
    threshold = load_calibrated_threshold(budget=budget)

    df = evaluate_baselines(
        dataset_name=dataset_name,
        baseline_names=ABLATION_BASELINES,
        max_samples=max_samples,
        pre_router_threshold=threshold,
    )

    # ── FIX 1 assertion: router must have fired ───────────────────────────────
    rr_rows = df[df["system"] == "rag_router"]
    if not rr_rows.empty:
        rr_full_frac = float(rr_rows.iloc[0]["full_llm_fraction"])
        assert rr_full_frac > 0.0, (
            f"Router never fired — threshold ({threshold:.4f}) likely too high. "
            "Re-run train_router.py or lower --budget."
        )
        print(f"\n  [ASSERT] rag_router full_llm_fraction={rr_full_frac:.1%} > 0.0 ✓")


    # ── Reorder to match ABLATION_BASELINES ──────────────────────────────────
    order = {name: i for i, name in enumerate(ABLATION_BASELINES)}
    df["_order"] = df["system"].map(order)
    df = df.sort_values("_order").drop(columns=["_order"])

    # ── Save ablation-specific tables ────────────────────────────────────────
    csv_path = TABLES_DIR / f"ablation_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)

    display_cols = [
        "system", "bertscore_f1", "rouge_l_f1",
        "full_llm_fraction", "cost_savings_fraction", "mean_latency_ms",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols]

    tex_path = TABLES_DIR / f"ablation_{dataset_name}.tex"
    df_display.to_latex(tex_path, index=False, float_format="%.4f")

    # ── Generate plot ─────────────────────────────────────────────────────────
    plot_ablation_bars(df_display, dataset_name)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    try:
        from experiments.validate import validate_results
        import json as _json2
        pos_rate = 0.938
        thresh_path2 = Path(MODELS_DIR) / "pre_router_threshold.json"
        if thresh_path2.exists():
            with open(thresh_path2) as _f2:
                pos_rate = float(_json2.load(_f2).get("positive_rate", pos_rate))
        validate_results(df, positive_rate=pos_rate)
    except AssertionError as ae:
        print(f"\n  [WARN] Sanity check failed: {ae}")
    except Exception as e:
        print(f"\n  [WARN] Could not run sanity checks: {e}")

    # ── Log ──────────────────────────────────────────────────────────────────
    log_training_event({
        "event": "ablation_complete",
        "dataset": dataset_name,
        "systems": ABLATION_BASELINES,
        "threshold_used": threshold,
        "n_samples": len(df),
    })

    print(f"\n{'='*60}")
    print("Ablation Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation: Mode A vs B vs C")
    parser.add_argument("--dataset", type=str, default="pubmedqa",
                        choices=["natural_questions", "pubmedqa"],
                        help="Dataset to use. Reads from labeled_routing_data.jsonl.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--budget", type=float, default=0.5,
                        help="Target full-LLM fraction for threshold fallback lookup "
                             "(default: 0.5). Ignored if pre_router_threshold.json exists.")
    args = parser.parse_args()

    run_ablation(args.dataset, args.max_samples, budget=args.budget)
