"""
Budget-Constrained Threshold Optimizer
=======================================
Given a maximum allowable fraction of full LLM calls (budget B),
find the optimal routing threshold that maximises accuracy subject to
cost(threshold) <= B.

Research contribution:
    This module produces the Pareto frontier plot (Figure 2 in the paper):
        X-axis: fraction of full LLM calls (cost proxy)
        Y-axis: BERTScore F1 (accuracy)

    Each point on the curve corresponds to one threshold value.
    The curve answers: "how much accuracy do you lose per unit of cost savings?"

    The key finding is that RAG-Router's curve dominates the FrugalGPT
    and heuristic baselines at every cost point.
"""

import numpy as np
from dataclasses import dataclass

from config import BUDGET_FRACTIONS


@dataclass
class ThresholdResult:
    """Result of evaluating a single routing threshold."""
    threshold: float
    budget_fraction: float          # Fraction of queries routed to full LLM
    accuracy: float                 # BERTScore F1
    cost_savings: float             # 1 - budget_fraction
    accuracy_vs_full: float         # accuracy / always_full_accuracy


def sweep_thresholds(
    routing_probs: np.ndarray,
    cheap_scores: np.ndarray,
    full_scores: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[ThresholdResult]:
    """Sweep routing thresholds to build the Pareto frontier.

    For each threshold t:
        - Route to cheap if routing_prob >= t, else full
        - Compute resulting accuracy and cost

    Args:
        routing_probs: (N,) pre-router P(cheap succeeds) for each query.
        cheap_scores:  (N,) BERTScore F1 from cheap LLM.
        full_scores:   (N,) BERTScore F1 from full LLM.
        thresholds:    thresholds to sweep (default: 101 points in [0, 1]).

    Returns:
        List of ThresholdResult sorted by budget_fraction (ascending).
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    full_accuracy = float(np.mean(full_scores))
    results = []

    for t in thresholds:
        use_cheap = routing_probs >= t
        adaptive_scores = np.where(use_cheap, cheap_scores, full_scores)
        accuracy = float(np.mean(adaptive_scores))
        budget_fraction = float(np.mean(~use_cheap))

        results.append(ThresholdResult(
            threshold=float(t),
            budget_fraction=budget_fraction,
            accuracy=accuracy,
            cost_savings=1.0 - budget_fraction,
            accuracy_vs_full=accuracy / (full_accuracy + 1e-9),
        ))

    return sorted(results, key=lambda r: r.budget_fraction)


def find_optimal_threshold(
    results: list[ThresholdResult],
    budget: float,
) -> ThresholdResult:
    """Find the best threshold within a cost budget.

    Args:
        results: output of sweep_thresholds().
        budget:  maximum fraction of full LLM calls allowed.

    Returns:
        ThresholdResult with highest accuracy where budget_fraction <= budget.
    """
    feasible = [r for r in results if r.budget_fraction <= budget]
    if not feasible:
        # No threshold meets budget; return the most conservative option
        return min(results, key=lambda r: r.budget_fraction)
    return max(feasible, key=lambda r: r.accuracy)


def find_optimal_thresholds_for_budgets(
    results: list[ThresholdResult],
    budgets: list[float] | None = None,
) -> dict[float, ThresholdResult]:
    """Find optimal thresholds for multiple budget levels.

    Research purpose:
        Generates the data for Table 2 in the paper — accuracy at each
        cost budget level (10%, 20%, ..., 90% full LLM calls).

    Args:
        results: output of sweep_thresholds().
        budgets: list of budget fractions (default: BUDGET_FRACTIONS from config).

    Returns:
        {budget: ThresholdResult} mapping.
    """
    if budgets is None:
        budgets = BUDGET_FRACTIONS

    return {b: find_optimal_threshold(results, b) for b in budgets}
