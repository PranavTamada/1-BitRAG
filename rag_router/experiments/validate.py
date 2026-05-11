"""
Pipeline Sanity-Check Assertions
==================================
validate_results() is called at the end of run_ablation.py and run_pareto.py
to catch threshold collapse, degenerate models, and extreme label distributions
before results are written to the paper tables.

Usage:
    from experiments.validate import validate_results
    validate_results(results_df, positive_rate=0.938)
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_results(results_df: pd.DataFrame, positive_rate: float) -> None:
    """Run all sanity-check assertions on the results DataFrame.

    Args:
        results_df:    DataFrame with at least columns:
                         system, bertscore_f1, full_llm_fraction
        positive_rate: fraction of labels where cheap_succeeds == 1 (e.g. 0.938)

    Raises:
        AssertionError: if any check fails, with the check name in the message.
    """
    print("\n  [VALIDATE] Running pipeline sanity checks...")

    def _get(system: str, col: str) -> float | None:
        """Return the value for (system, col) or None if not present."""
        row = results_df[results_df["system"] == system]
        if row.empty or col not in row.columns:
            return None
        return float(row.iloc[0][col])

    # ── Check 1: Always-cheap BERTScore sanity ───────────────────────────────
    cheap_bs = _get("always_cheap", "bertscore_f1")
    if cheap_bs is not None:
        assert cheap_bs > 0.5, (
            f"[FAIL] always_cheap BERTScore too low: {cheap_bs:.4f} <= 0.5  "
            "(check labeling / metric computation)"
        )

    # ── Check 2: Always-full BERTScore sanity ────────────────────────────────
    full_bs = _get("always_full", "bertscore_f1")
    if full_bs is not None:
        assert full_bs > 0.5, (
            f"[FAIL] always_full BERTScore too low: {full_bs:.4f} <= 0.5  "
            "(check full LLM answers)"
        )

    # ── Check 3: Router never fired ──────────────────────────────────────────
    if positive_rate < 0.96:
        rr_frac = _get("rag_router", "full_llm_fraction")
        if rr_frac is not None:
            assert rr_frac > 0.0, (
                f"[FAIL] Router never fired (full_llm_fraction=0.0) at "
                f"positive_rate={positive_rate:.3f} — threshold is too high. "
                "Check models/pre_router_threshold.json."
            )

    # ── Check 4: Router not much worse than always_cheap ────────────────────
    if cheap_bs is not None:
        rr_bs = _get("rag_router", "bertscore_f1")
        if rr_bs is not None:
            assert rr_bs >= cheap_bs * 0.95, (
                f"[FAIL] RAG-Router BERTScore ({rr_bs:.4f}) is >5% below "
                f"always_cheap ({cheap_bs:.4f}). Router may be misconfigured."
            )

    # ── Check 5: Label distribution extreme ─────────────────────────────────
    assert 0.4 <= positive_rate <= 0.96, (
        f"[FAIL] Label distribution extreme: positive_rate={positive_rate:.3f} "
        "outside [0.40, 0.96]. Double-check labeling logic / dataset."
    )

    print("  [PASS] All sanity checks passed")
