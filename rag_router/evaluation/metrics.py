"""
Evaluation Metrics
===================
BERTScore F1 as primary metric (NOT substring matching -- that was v1's flaw).
ROUGE-L as secondary. Exact match for sanity check only.

Research note:
    BERTScore is computed with distilbert-base-uncased for development speed.
    For camera-ready paper, switch to roberta-large in config.py.

Why NOT substring matching:
    v1 used contains_answer() which checks if pred in gt or gt in pred.
    This is fundamentally flawed for open-ended QA because:
    - "Yes" substring-matches "Yes, but only in certain conditions..."
    - Paraphrases are never captured
    - Score is binary, giving no gradient for threshold tuning
    BERTScore solves all three problems.

Statistical rigor (added for publication):
    - bootstrap_ci():              Bootstrap 95% confidence intervals
    - paired_significance_test():  Wilcoxon signed-rank test for paired comparisons
    - compute_dollar_cost():       Estimated dollar cost per system
"""

import numpy as np
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer as rouge_scorer_module
from scipy.stats import wilcoxon

from config import BERTSCORE_MODEL, COST_PER_TOKEN, AVG_TOKENS_PER_QUERY, CHEAP_MODEL, FULL_MODEL


# ═════════════════════════════════════════════════════════════════════════════
# Core Metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_bertscore(
    predictions: list[str], references: list[str]
) -> dict:
    """Compute BERTScore for a list of prediction-reference pairs.

    Research purpose:
        Primary accuracy metric for all result tables and Pareto curves.

    Returns:
        dict with bertscore_precision, bertscore_recall, bertscore_f1,
        bertscore_f1_std, and per-sample bertscore_f1_list.
    """
    # Handle empty inputs
    if not predictions or not references:
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
            "bertscore_f1_std": 0.0,
            "bertscore_f1_list": [],
        }

    # Replace empty strings to avoid bert-score errors
    predictions = [p if p.strip() else "No answer" for p in predictions]
    references = [r if r.strip() else "No answer" for r in references]

    P, R, F1 = bert_score_fn(
        predictions, references,
        model_type=BERTSCORE_MODEL,
        lang="en",
        verbose=False,
    )
    return {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
        "bertscore_f1_std": float(F1.std()),
        "bertscore_f1_list": F1.tolist(),
    }

def compute_bertscore_single(prediction: str, reference: str) -> float:
    """Compute BERTScore F1 for a single pair. Convenience wrapper."""
    result = compute_bertscore([prediction], [reference])
    return result["bertscore_f1"]

def compute_rouge_l(
    predictions: list[str], references: list[str]
) -> dict:
    """Compute ROUGE-L F1 for a list of prediction-reference pairs.

    Research purpose:
        Secondary metric -- reported alongside BERTScore for completeness.
        ROUGE-L captures longest common subsequence overlap.
    """
    scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        pred = pred if pred.strip() else "No answer"
        ref = ref if ref.strip() else "No answer"
        score = scorer.score(ref, pred)["rougeL"].fmeasure
        scores.append(score)
    return {
        "rouge_l_f1": float(np.mean(scores)),
        "rouge_l_f1_std": float(np.std(scores)),
    }

def compute_cost_metrics(routing_decisions: list[str]) -> dict:
    """Compute cost-related metrics from routing decisions.

    Args:
        routing_decisions: list of "cheap" or "full" for each query.
    """
    total = len(routing_decisions)
    if total == 0:
        return {
            "total_queries": 0, "full_llm_calls": 0,
            "cheap_llm_calls": 0, "full_llm_fraction": 0.0,
            "cost_savings_fraction": 1.0,
        }
    full_calls = sum(1 for d in routing_decisions if d == "full")
    return {
        "total_queries": total,
        "full_llm_calls": full_calls,
        "cheap_llm_calls": total - full_calls,
        "full_llm_fraction": full_calls / total,
        "cost_savings_fraction": 1.0 - (full_calls / total),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Statistical Rigor — Bootstrap CIs & Significance Tests
# ═════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean of a score list.

    Research purpose:
        Reviewers expect 95% CIs on all main results. This replaces
        naive std-based intervals with proper non-parametric bootstrapping.

    Args:
        scores:      per-sample metric values (e.g., BERTScore F1 per query).
        n_bootstrap: number of bootstrap resamples.
        ci:          confidence level (default: 0.95 for 95% CI).
        seed:        random seed for reproducibility.

    Returns:
        (lower, upper) bounds of the CI.
    """
    if not scores or len(scores) < 2:
        return (0.0, 0.0)
    rng = np.random.RandomState(seed)
    arr = np.array(scores, dtype=float)
    means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ])
    lower = float(np.percentile(means, (1 - ci) / 2 * 100))
    upper = float(np.percentile(means, (1 + ci) / 2 * 100))
    return lower, upper


def paired_significance_test(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> dict:
    """Wilcoxon signed-rank test for paired sample comparison.

    Research purpose:
        Tests whether the difference between two systems (e.g.,
        RAG-Router vs always_full) is statistically significant.
        Required for any claim of "significantly better/comparable"
        performance in a research paper.

    Args:
        scores_a: per-sample metrics for system A.
        scores_b: per-sample metrics for system B.
        alpha:    significance level (default: 0.05).

    Returns:
        dict with statistic, p_value, significant (bool), and effect_size.
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    diff = a - b

    # Wilcoxon requires non-zero differences
    nonzero = diff[diff != 0]
    if len(nonzero) < 10:
        return {
            "statistic": float("nan"),
            "p_value": 1.0,
            "significant": False,
            "effect_size": 0.0,
            "note": f"Too few non-zero differences ({len(nonzero)}) for Wilcoxon test",
        }

    stat, p = wilcoxon(a, b)
    # Effect size: r = Z / sqrt(N)
    z = (stat - len(nonzero) * (len(nonzero) + 1) / 4) / np.sqrt(
        len(nonzero) * (len(nonzero) + 1) * (2 * len(nonzero) + 1) / 24
    )
    effect_size = abs(z) / np.sqrt(len(nonzero))

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": p < alpha,
        "effect_size": float(effect_size),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Dollar Cost Estimation
# ═════════════════════════════════════════════════════════════════════════════

def compute_dollar_cost(routing_decisions: list[str]) -> dict:
    """Estimate dollar cost for a set of routing decisions.

    Research purpose:
        Reports cost in real dollars (not just % of full calls), making
        the cost savings claim concrete and actionable for practitioners.

    Args:
        routing_decisions: list of "cheap" or "full" for each query.

    Returns:
        dict with estimated costs.
    """
    total = len(routing_decisions)
    if total == 0:
        return {"estimated_cost_usd": 0.0, "always_full_cost_usd": 0.0, "dollar_savings_usd": 0.0}

    cheap_cost_per_query = COST_PER_TOKEN.get(CHEAP_MODEL, 0.0) * AVG_TOKENS_PER_QUERY
    full_cost_per_query = COST_PER_TOKEN.get(FULL_MODEL, 0.0) * AVG_TOKENS_PER_QUERY

    full_calls = sum(1 for d in routing_decisions if d == "full")
    cheap_calls = total - full_calls

    estimated_cost = (cheap_calls * cheap_cost_per_query) + (full_calls * full_cost_per_query)
    always_full_cost = total * full_cost_per_query

    return {
        "estimated_cost_usd": float(estimated_cost),
        "always_full_cost_usd": float(always_full_cost),
        "dollar_savings_usd": float(always_full_cost - estimated_cost),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Combined Metric Computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    routing_decisions: list[str],
    latencies: list[float],
) -> dict:
    """Compute all metrics for a complete evaluation run.

    Research purpose:
        This is the function called by evaluate.py and cross_domain.py
        to produce one row in the main result table.

    Returns:
        dict with BERTScore, ROUGE-L, cost metrics, dollar cost,
        bootstrap CIs, and latency stats.
    """
    metrics = {}
    metrics.update(compute_bertscore(predictions, references))
    metrics.update(compute_rouge_l(predictions, references))
    metrics.update(compute_cost_metrics(routing_decisions))
    metrics.update(compute_dollar_cost(routing_decisions))

    # Bootstrap 95% confidence intervals on BERTScore F1
    f1_list = metrics.get("bertscore_f1_list", [])
    if f1_list:
        ci_lower, ci_upper = bootstrap_ci(f1_list)
        metrics["bertscore_f1_ci_lower"] = ci_lower
        metrics["bertscore_f1_ci_upper"] = ci_upper

    if latencies:
        metrics["mean_latency_ms"] = float(np.mean(latencies) * 1000)
        metrics["median_latency_ms"] = float(np.median(latencies) * 1000)
    else:
        metrics["mean_latency_ms"] = 0.0
        metrics["median_latency_ms"] = 0.0
    return metrics
