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
    python experiments/collect_labels.py                          # PubMedQA dataset
    python experiments/collect_labels.py --max-samples 50         # Quick test run
    python experiments/collect_labels.py --force                  # Wipe existing pubmedqa entries and re-label
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
    BERTSCORE_SUCCESS_THRESHOLD, DATA_DIR, BERTSCORE_MODEL,LABEL_MODE, GAP_RATIO,
)
from data.loaders import load_dataset_by_name
from retriever.dense import DenseRetriever
from retriever.sparse import SparseRetriever
from retriever.retrieve import retrieve
from features.retrieval_features import extract_retrieval_features, feature_vector
from features.query_features import extract_query_features, query_feature_vector
from llm.cheap_llm import run_cheap_llm
from llm.full_llm import run_full_llm, AllKeysExhaustedError
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


def collect_labels(max_samples: int | None = None, force: bool = False) -> None:
    """Run the full label collection pipeline for the pubmedqa dataset.

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
        max_samples: optional cap on number of queries to process
        force: if True, purge existing pubmedqa entries before re-labeling
    """
    dataset_name = "pubmedqa"
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

    # ── Force mode: strip existing pubmedqa entries so labels are recomputed ──
    if force and output_path.exists():
        print(f"  --force: purging existing '{dataset_name}' entries from {output_path.name}...")
        surviving = []
        purged = 0
        with open(output_path, "r", encoding="utf-8") as f_purge:
            for line in f_purge:
                if line.strip():
                    try:
                        rec = json.loads(line.strip())
                        if rec.get("dataset") == dataset_name:
                            purged += 1
                        else:
                            surviving.append(line)
                    except json.JSONDecodeError:
                        surviving.append(line)
        with open(output_path, "w", encoding="utf-8") as f_purge:
            f_purge.writelines(surviving)
        print(f"  Purged {purged} old '{dataset_name}' records. Kept {len(surviving)} records from other datasets.")

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
            # CRITICAL: both cheap and full LLMs receive the SAME prompt.
            # This ensures the quality comparison is about model capacity,
            # not about information access (prompt fairness).
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
            # Same prompt as cheap LLM — prompt fairness guarantee.
            full_answer, full_latency = "", 0.0
            try:
                full_answer, full_latency = cached_llm_call(
                    FULL_MODEL, prompt, run_full_llm
                )
            except AllKeysExhaustedError as e:
                # All API keys are rate-limited — save progress and stop.
                f_out.flush()
                print(f"\n\n{'='*60}")
                print("[STOPPED] All Groq API keys have hit their daily rate limit.")
                print(f"  Progress saved: {labeled_count} samples written to disk.")
                print(f"  Re-run tomorrow (or add more keys) to continue.")
                print(f"  Error: {e}")
                print(f"{'='*60}\n")
                break  # Exit the for-loop; summary stats will still print
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
            if LABEL_MODE == "gap":
                # Relative: cheap succeeds only if it is >= GAP_RATIO
                # of the full model's quality. Measures quality *loss*
                # from skipping the full model — correct for routing.
                cheap_succeeds = int(
                    cheap_bertscore >= full_bertscore * GAP_RATIO
                )
            else:  # "absolute" — original behaviour
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
                "bertscore_gap": round(full_bertscore - cheap_bertscore, 6),
                "label_mode": LABEL_MODE,
                "gap_ratio_used": GAP_RATIO if LABEL_MODE == "gap" else None,
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
    all_records = []
    with open(output_path, "r", encoding="utf-8") as f_diag:
        for line in f_diag:
            if line.strip():
                try: all_records.append(json.loads(line))
                except: pass
    this_run = [r for r in all_records if r.get("dataset") == dataset_name]
    if this_run:
        cheap_sc = [r.get("cheap_bertscore", 0.0) for r in this_run]
        full_sc  = [r.get("full_bertscore",  0.0) for r in this_run]
        gaps     = [r.get("bertscore_gap", round(r.get("full_bertscore", 0.0) - r.get("cheap_bertscore", 0.0), 6)) for r in this_run]
        print(f"\n  Score diagnostics:")
        print(f"    Cheap BERTScore : mean={np.mean(cheap_sc):.4f}  std={np.std(cheap_sc):.4f}")
        print(f"    Full  BERTScore : mean={np.mean(full_sc):.4f}  std={np.std(full_sc):.4f}")
        print(f"    Gap (full-cheap): mean={np.mean(gaps):.4f}  p25={np.percentile(gaps,25):.4f}  p75={np.percentile(gaps,75):.4f}")
        print(f"    Gap > 0.02      : {sum(g > 0.02 for g in gaps)}/{len(gaps)} samples")
        print(f"    Label mode      : {LABEL_MODE}" + (f"  (ratio={GAP_RATIO})" if LABEL_MODE=='gap' else ""))
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
        "--max-samples", type=int, default=None,
        help="Optional cap on number of samples (for quick testing).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Purge existing pubmedqa entries from the output file and re-label from scratch.",
    )
    args = parser.parse_args()

    collect_labels(max_samples=args.max_samples, force=args.force)
