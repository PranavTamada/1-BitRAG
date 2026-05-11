"""
Step 1: Label Collection
=========================
Run BOTH cheap and full LLMs on all dataset queries, record outputs,
compute BERTScore F1 for each, and extract all retrieval + query features.

Expected runtime: 4-8 hours for full dataset (depends on Ollama speed + Groq rate limits)
Expected output:  data/labeled_routing_data.jsonl

Label definition:
    cheap_succeeds = 1 if BERTScore(cheap_answer, ground_truth) >= BERTSCORE_SUCCESS_THRESHOLD
    cheap_succeeds = 0 otherwise

This file is the foundation of the entire research paper.
Run once and cache -- never re-run unless you change the dataset.

Usage:
    python experiments/collect_labels.py                          # Healthcare QA only
    python experiments/collect_labels.py --dataset healthcare_qa  # Explicit dataset
    python experiments/collect_labels.py --dataset pubmedqa       # PubMedQA
    python experiments/collect_labels.py --max-samples 50         # Quick test run
"""

import sys
import os
import json
import gc
import argparse
import numpy as np
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHEAP_MODEL, FULL_MODEL, TOP_K,
    BERTSCORE_SUCCESS_THRESHOLD, DATA_DIR, BERTSCORE_MODEL,
)
from data.loaders import load_dataset_by_name
from retriever.dense import DenseRetriever
from retriever.sparse import SparseRetriever
from retriever.retrieve import retrieve
from features.retrieval_features import extract_retrieval_features, feature_vector
from features.query_features import extract_query_features, query_feature_vector
from llm.cheap_llm import run_cheap_llm
from llm.full_llm import run_full_llm
from utils.cache import cached_llm_call
from utils.prompt import build_summary_prompt, build_direct_prompt
from utils.logger import log_training_event


def compute_bertscore_single(prediction: str, reference: str) -> float:
    """Compute BERTScore F1 for a single prediction-reference pair.

    Research note:
        We compute per-sample BERTScore here (not batch) because we need
        individual scores for label assignment.  Batch computation is
        used in evaluation/metrics.py for final result tables.
    """
    from bert_score import score as bert_score
    P, R, F1 = bert_score(
        [prediction], [reference],
        model_type=BERTSCORE_MODEL,
        lang="en",
        verbose=False,
    )
    return float(F1[0])


def collect_labels(dataset_name: str, max_samples: int | None = None) -> None:
    """Run the full label collection pipeline for a given dataset.

    Steps:
        1. Load dataset
        2. Build retrieval corpus from all queries
        3. For each query:
           a. Run hybrid retrieval -> extract features
           b. Call cheap LLM (cached) -> compute BERTScore
           c. Call full LLM (cached) -> compute BERTScore
           d. Assign binary label: cheap_succeeds
        4. Save to data/labeled_routing_data.jsonl

    Args:
        dataset_name: one of "healthcare_qa", "natural_questions", "pubmedqa"
        max_samples: optional cap on number of queries to process
    """
    print(f"\n{'='*60}")
    print(f"Label Collection: {dataset_name}")
    print(f"{'='*60}")

    # ── 1. Load dataset ───────────────────────────────────────────────────
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset_by_name(dataset_name, max_samples=max_samples)
    print(f"  Loaded {len(dataset)} samples")

    # ── 2. Build retrieval corpus ─────────────────────────────────────────
    documents = [sample["query"] for sample in dataset]
    completions = [sample["answer"] for sample in dataset]

    print("Initializing dense retriever...")
    dense_retriever = DenseRetriever(documents)
    print("Initializing sparse retriever...")
    sparse_retriever = SparseRetriever(documents)

    # ── 3. Process each query ─────────────────────────────────────────────
    output_path = DATA_DIR / "labeled_routing_data.jsonl"

    # Load already-processed queries for resume capability
    existing_queries = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f_existing:
            for line in f_existing:
                if line.strip():
                    try:
                        rec = json.loads(line.strip())
                        if rec.get("dataset") == dataset_name:
                            existing_queries.add(rec["query"])
                    except json.JSONDecodeError:
                        pass
    if existing_queries:
        print(f"  Resuming: {len(existing_queries)} queries already processed")

    # Append mode: allows running different datasets sequentially
    mode = "a" if output_path.exists() else "w"

    labeled_count = 0
    cheap_success_count = 0

    with open(output_path, mode, encoding="utf-8") as f_out:
        for i, sample in enumerate(tqdm(dataset, desc="Collecting labels")):
            query = sample["query"]
            ground_truth = sample["answer"]

            # Skip already-processed queries (resume support)
            if query in existing_queries:
                continue

            # ── 3a. Hybrid retrieval + feature extraction ─────────────
            result = retrieve(
                query, documents, dense_retriever, sparse_retriever, top_k=TOP_K
            )

            # Extract retrieval features
            # Map result indices to their BM25/dense ranks
            bm25_ranks_for_topk = [
                result.sparse_ranks.get(idx, TOP_K + 1)
                for idx in result.indices
            ]
            dense_ranks_for_topk = [
                result.dense_ranks.get(idx, TOP_K + 1)
                for idx in result.indices
            ]

            r_feats = extract_retrieval_features(
                fused_scores=result.scores,
                bm25_ranks=bm25_ranks_for_topk,
                dense_ranks=dense_ranks_for_topk,
                doc_embeddings=result.doc_embeddings,
                top_k=TOP_K,
            )

            # Extract query features
            q_feats = extract_query_features(query)

            # ── 3b. Build prompt with retrieved context ───────────────
            if result.scores and result.scores[0] > 0:
                # Build context from retrieved docs + their completions
                retrieved_pairs = []
                for idx in result.indices:
                    if idx < len(documents) and idx < len(completions):
                        retrieved_pairs.append(
                            (documents[idx], completions[idx])
                        )
                prompt = build_summary_prompt(retrieved_pairs, query)
            else:
                prompt = build_direct_prompt(query)

            # ── 3c. Call cheap LLM (cached) ───────────────────────────
            try:
                cheap_answer, cheap_latency = cached_llm_call(
                    CHEAP_MODEL, prompt, run_cheap_llm
                )
            except Exception as e:
                cheap_answer = f"[ERROR] {e}"
                cheap_latency = 0.0

            # ── 3d. Call full LLM (cached) ────────────────────────────
            direct_prompt = build_direct_prompt(query)
            try:
                full_answer, full_latency = cached_llm_call(
                    FULL_MODEL, direct_prompt, run_full_llm
                )
            except Exception as e:
                full_answer = f"[ERROR] {e}"
                full_latency = 0.0

            # ── 3e. Compute BERTScores ────────────────────────────────
            try:
                cheap_bertscore = compute_bertscore_single(
                    cheap_answer, ground_truth
                )
            except Exception:
                cheap_bertscore = 0.0

            try:
                full_bertscore = compute_bertscore_single(
                    full_answer, ground_truth
                )
            except Exception:
                full_bertscore = 0.0

            # ── 3f. Assign label ──────────────────────────────────────
            cheap_succeeds = int(
                cheap_bertscore >= BERTSCORE_SUCCESS_THRESHOLD
            )
            cheap_success_count += cheap_succeeds

            # ── 3g. Write labeled sample ──────────────────────────────
            record = {
                "query": query,
                "ground_truth": ground_truth,
                "cheap_answer": cheap_answer,
                "full_answer": full_answer,
                "cheap_bertscore": cheap_bertscore,
                "full_bertscore": full_bertscore,
                "cheap_succeeds": cheap_succeeds,
                "cheap_latency": cheap_latency,
                "full_latency": full_latency,
                "retrieval_features": r_feats,
                "query_features": q_feats,
                "retrieval_scores": result.scores[:TOP_K],
                "dataset": dataset_name,
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()  # Ensure each record is persisted immediately
            labeled_count += 1

            # Memory management: free BERTScore internals periodically
            if labeled_count % 10 == 0:
                gc.collect()

    # ── 4. Summary statistics ─────────────────────────────────────────────
    positive_rate = (
        cheap_success_count / labeled_count if labeled_count > 0 else 0.0
    )
    summary = {
        "event": "label_collection_complete",
        "dataset": dataset_name,
        "total_samples": labeled_count,
        "cheap_succeeds_count": cheap_success_count,
        "positive_rate": positive_rate,
        "output_file": str(output_path),
        "bertscore_threshold": BERTSCORE_SUCCESS_THRESHOLD,
    }
    log_training_event(summary)

    print(f"\n{'='*60}")
    print(f"Label Collection Complete: {dataset_name}")
    print(f"  Total samples:  {labeled_count}")
    print(f"  Cheap succeeds: {cheap_success_count} ({positive_rate:.1%})")
    print(f"  Output file:    {output_path}")
    print(f"  Label balance:  {'GOOD' if 0.3 <= positive_rate <= 0.7 else 'WARNING - imbalanced'}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect routing labels by running both LLMs on dataset queries."
    )
    parser.add_argument(
        "--dataset", type=str, default="pubmedqa",
        choices=["natural_questions", "pubmedqa"],
        help="Which dataset to collect labels for. healthcare_qa removed; use labeled_routing_data.jsonl directly.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Optional cap on number of samples (for quick testing).",
    )
    args = parser.parse_args()

    collect_labels(args.dataset, max_samples=args.max_samples)
