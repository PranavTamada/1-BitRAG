"""
Full Evaluation Loop
=====================
Runs all baselines on a dataset and produces the main result table.
Data source: data/labeled_routing_data.jsonl  (pubmedqa samples only)

Expected runtime: depends on dataset size and LLM availability
Expected output:
    - results/tables/main_results_{dataset}.csv
    - results/tables/main_results_{dataset}.tex
    - Console: formatted result table

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --dataset pubmedqa
    python evaluation/evaluate.py --dataset pubmedqa --max-samples 50
    python evaluation/evaluate.py --baselines always_cheap always_full rag_router
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
TABLES_DIR, TOP_K, RANDOM_STATE, DATA_DIR,
)
from data.loaders import load_dataset_by_name
from retriever.dense import DenseRetriever
from retriever.sparse import SparseRetriever
from retriever.retrieve import retrieve
from features.retrieval_features import extract_retrieval_features, feature_vector
from features.query_features import extract_query_features, query_feature_vector
from evaluation.metrics import compute_all_metrics, paired_significance_test
from evaluation.baselines import BASELINE_REGISTRY
from router.pre_router import PreRouter
from router.post_router import PostRouter
from utils.prompt import build_summary_prompt, build_direct_prompt
from utils.logger import log_training_event


def build_prompt_from_retrieval(result, documents, completions, query):
    """Build the appropriate prompt from retrieval results."""
    if result.scores and result.scores[0] > 0:
        retrieved_pairs = []
        for idx in result.indices:
            if idx < len(documents) and idx < len(completions):
                retrieved_pairs.append((documents[idx], completions[idx]))
        return build_summary_prompt(retrieved_pairs, query)
    return build_direct_prompt(query)


def evaluate_baselines(
    dataset_name: str,
    baseline_names: list[str] | None = None,
    max_samples: int | None = None,
    pre_router_threshold: float | None = None,
) -> pd.DataFrame:
    """Run all specified baselines on a dataset and collect metrics.

    Args:
        dataset_name:         which dataset to evaluate on.
        baseline_names:       which baselines to run (default: all).
        max_samples:          optional cap for quick testing.
        pre_router_threshold: if provided, override the pre-router's decision
                              threshold (FIX 1 — calibrated threshold injection).

    Returns:
        DataFrame with one row per baseline, columns for all metrics.
    """
    if baseline_names is None:
        baseline_names = list(BASELINE_REGISTRY.keys())

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"\nLoading dataset: {dataset_name}...")
    dataset = load_dataset_by_name(dataset_name, max_samples=max_samples)
    print(f"  {len(dataset)} samples loaded")

    documents = [s["query"] for s in dataset]
    completions = [s["answer"] for s in dataset]

    # ── Build retrieval index ─────────────────────────────────────────────
    print("Initializing retrievers...")
    dense_retriever = DenseRetriever(documents)
    sparse_retriever = SparseRetriever(documents)

    # ── Load trained routers (if needed) ──────────────────────────────────
    pre_router = None
    post_router = None
    needs_pre = any(
        b in baseline_names for b in ["pre_only", "rag_router"]
    )
    needs_post = any(
        b in baseline_names for b in ["rag_router"]
    )

    if needs_pre:
        try:
            pre_router = PreRouter("logistic")
            pre_router.load()
            # FIX 1: inject calibrated threshold if provided
            if pre_router_threshold is not None:
                pre_router.threshold = pre_router_threshold
                print(f"  [THRESHOLD] Pre-router threshold overridden to "
                      f"{pre_router_threshold:.4f}")
            else:
                print(f"  [THRESHOLD] Using serialised threshold: "
                      f"{pre_router.threshold:.4f}")
        except Exception as e:
            print(f"  [WARN] Could not load pre-router: {e}")
            print(f"         Skipping baselines that need pre-router.")
            baseline_names = [
                b for b in baseline_names if b not in ["pre_only", "rag_router"]
            ]

    if needs_post:
        try:
            post_router = PostRouter()
            post_router.load()
        except Exception as e:
            print(f"  [WARN] Could not load post-router: {e}")
            print(f"         Skipping baselines that need post-router.")
            baseline_names = [
                b for b in baseline_names
                if b not in ["rag_router"]
            ]

    # ── Pre-compute retrieval + features for all queries ──────────────────
    print("Running retrieval for all queries...")
    all_results = []
    all_prompts = []
    all_feature_vecs = []
    rng = np.random.RandomState(RANDOM_STATE)

    for sample in tqdm(dataset, desc="Retrieval"):
        query = sample["query"]
        result = retrieve(
            query, documents, dense_retriever, sparse_retriever, top_k=TOP_K
        )
        all_results.append(result)

        prompt = build_prompt_from_retrieval(
            result, documents, completions, query
        )
        all_prompts.append(prompt)

        # Build combined feature vector
        bm25_ranks = [
            result.sparse_ranks.get(idx, TOP_K + 1) for idx in result.indices
        ]
        dense_ranks = [
            result.dense_ranks.get(idx, TOP_K + 1) for idx in result.indices
        ]
        r_feats = extract_retrieval_features(
            result.scores, bm25_ranks, dense_ranks,
            result.doc_embeddings, top_k=TOP_K,
        )
        q_feats = extract_query_features(query)
        r_vec = feature_vector(r_feats)
        q_vec = query_feature_vector(q_feats)
        all_feature_vecs.append(np.concatenate([r_vec, q_vec]))

    # ── Load labeled BERTScores for oracle baseline ───────────────────────
    labeled_scores = {}  # {query_text: {cheap_bertscore, full_bertscore}}
    if "oracle_routing" in baseline_names:
        labeled_path = DATA_DIR / "labeled_routing_data.jsonl"
        if labeled_path.exists():
            import json as _json
            with open(labeled_path, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _rec = _json.loads(_line)
                        if _rec.get("dataset") == dataset_name:
                            labeled_scores[_rec["query"]] = {
                                "cheap_bertscore": _rec.get("cheap_bertscore", 0.0),
                                "full_bertscore": _rec.get("full_bertscore", 0.0),
                            }
                    except Exception:
                        pass
            print(f"  Loaded {len(labeled_scores)} labeled BERTScores for oracle baseline")
        else:
            print(f"  [WARN] No labeled data found for oracle baseline; oracle will default to full.")

    # ── Run each baseline ─────────────────────────────────────────────────
    all_metrics = []

    for baseline_name in baseline_names:
        print(f"\n--- Running baseline: {baseline_name} ---")
        baseline_fn = BASELINE_REGISTRY[baseline_name]

        predictions = []
        decisions = []
        latencies = []
        references = [s["answer"] for s in dataset]

        for i, sample in enumerate(tqdm(dataset, desc=baseline_name)):
            query = sample["query"]
            prompt = all_prompts[i]
            fvec = all_feature_vecs[i]

            # Oracle baseline needs pre-computed BERTScores
            oracle_kwargs = {}
            if baseline_name == "oracle_routing" and query in labeled_scores:
                oracle_kwargs = labeled_scores[query]

            try:
                result = baseline_fn(
                    query=query,
                    prompt=prompt,
                    pre_router=pre_router,
                    post_router=post_router,
                    feature_vec=fvec,
                    full_fraction=0.5,
                    rng=rng,
                    **oracle_kwargs,
                )
                predictions.append(result.answer)
                decisions.append(result.decision)
                latencies.append(result.latency)
            except Exception as e:
                print(f"  [ERROR] {baseline_name} on query {i}: {e}")
                predictions.append("")
                decisions.append("full")
                latencies.append(0.0)

        # Compute metrics
        metrics = compute_all_metrics(
            predictions, references, decisions, latencies
        )
        metrics["system"] = baseline_name
        metrics["dataset"] = dataset_name
        all_metrics.append(metrics)

        print(f"  BERTScore F1: {metrics['bertscore_f1']:.4f}")
        print(f"  Full LLM %:  {metrics['full_llm_fraction']:.1%}")
        print(f"  Mean Latency: {metrics['mean_latency_ms']:.0f}ms")

    # ── Build result table ────────────────────────────────────────────────
    df = pd.DataFrame(all_metrics)
    display_cols = [
        "dataset", "system", "bertscore_f1", "rouge_l_f1",
        "full_llm_fraction", "cost_savings_fraction",
        "mean_latency_ms",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols].copy()

    print(f"\n{'='*80}")
    print(f"Results: {dataset_name}")
    print(f"{'='*80}")
    print(df_display.to_string(index=False))

    # ── Save results ──────────────────────────────────────────────────────
    csv_path = TABLES_DIR / f"main_results_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    tex_path = TABLES_DIR / f"main_results_{dataset_name}.tex"
    df_display.to_latex(tex_path, index=False, float_format="%.4f")
    print(f"Saved to {tex_path}")

    # ── Paired significance tests ─────────────────────────────────────────
    # Compare RAG-Router vs always_full (key claim in the paper)
    rr_metrics = next((m for m in all_metrics if m["system"] == "rag_router"), None)
    af_metrics = next((m for m in all_metrics if m["system"] == "always_full"), None)
    if rr_metrics and af_metrics:
        rr_f1_list = rr_metrics.get("bertscore_f1_list", [])
        af_f1_list = af_metrics.get("bertscore_f1_list", [])
        if len(rr_f1_list) == len(af_f1_list) and len(rr_f1_list) > 0:
            sig_result = paired_significance_test(rr_f1_list, af_f1_list)
            print(f"\n  Significance (RAG-Router vs Always-Full):")
            print(f"    Wilcoxon p-value:   {sig_result['p_value']:.4f}")
            print(f"    Significant (p<.05): {'YES' if sig_result['significant'] else 'NO'}")
            print(f"    Effect size (r):    {sig_result['effect_size']:.4f}")
            if 'note' in sig_result:
                print(f"    Note: {sig_result['note']}")

    log_training_event({
        "event": "evaluation_complete",
        "dataset": dataset_name,
        "n_baselines": len(baseline_names),
        "n_samples": len(dataset),
    })

    return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all baselines and produce result tables."
    )
    parser.add_argument(
        "--dataset", type=str, default="pubmedqa",
        choices=["natural_questions", "pubmedqa"],
        help="Dataset to evaluate on. Uses data/labeled_routing_data.jsonl as source.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
    )
    parser.add_argument(
        "--baselines", nargs="+", default=None,
        choices=list(BASELINE_REGISTRY.keys()),
        help="Which baselines to run (default: all).",
    )
    args = parser.parse_args()

    evaluate_baselines(
        dataset_name=args.dataset,
        baseline_names=args.baselines,
        max_samples=args.max_samples,
    )