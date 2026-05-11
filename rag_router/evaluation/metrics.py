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
"""

import numpy as np
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer as rouge_scorer_module

from config import BERTSCORE_MODEL


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
        dict with BERTScore, ROUGE-L, cost metrics, and latency stats.
    """
    metrics = {}
    metrics.update(compute_bertscore(predictions, references))
    metrics.update(compute_rouge_l(predictions, references))
    metrics.update(compute_cost_metrics(routing_decisions))
    if latencies:
        metrics["mean_latency_ms"] = float(np.mean(latencies) * 1000)
        metrics["median_latency_ms"] = float(np.median(latencies) * 1000)
    else:
        metrics["mean_latency_ms"] = 0.0
        metrics["median_latency_ms"] = 0.0
    return metrics
