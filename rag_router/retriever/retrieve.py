"""
Unified Retrieval Interface
============================
Orchestrates dense + sparse retrieval, RRF fusion, and returns all
metadata needed by the feature extractor and the routing pipeline.

Enhanced from v1:
  * Returns a ``RetrievalResult`` dataclass with score distributions,
    per-system ranks, and document embeddings — everything the
    retrieval-feature extractor needs in a single call.
  * No more scattered dicts — one clean return type.
"""

from dataclasses import dataclass, field
import numpy as np

from retriever.dense import DenseRetriever
from retriever.sparse import SparseRetriever
from retriever.fusion import fuse_scores
from config import TOP_K


@dataclass
class RetrievalResult:
    """All outputs of a single retrieval call.

    Designed so the feature extractor can consume this directly without
    any additional retriever calls — zero redundant work.
    """
    docs: list[str]                            # Retrieved document texts
    scores: list[float]                        # Fused RRF scores (descending)
    indices: list[int]                         # Corpus indices of retrieved docs
    dense_ranks: dict = field(default_factory=dict)   # {doc_idx: dense_rank}
    sparse_ranks: dict = field(default_factory=dict)  # {doc_idx: sparse_rank}
    doc_embeddings: np.ndarray = None          # (top_k, dim) embedding matrix


def retrieve(
    query: str,
    documents: list[str],
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever,
    top_k: int = TOP_K,
) -> RetrievalResult:
    """Hybrid retrieval: dense + sparse → union → score all → RRF fuse → top-k.

    Research purpose:
        This function is the entry point for every query in the pipeline.
        It returns not just the retrieved documents but the full score
        distribution and ranking metadata that the retrieval-geometry
        feature extractor uses to predict query difficulty *before*
        any LLM is called.

    Args:
        query:            the search query string
        documents:        list of corpus text strings
        dense_retriever:  pre-initialised DenseRetriever instance
        sparse_retriever: pre-initialised SparseRetriever instance
        top_k:            number of final results to return

    Returns:
        RetrievalResult containing docs, scores, indices, per-system ranks,
        and document embeddings.
    """
    # ── Step 1 & 2: top-k from each system ────────────────────────────────
    dense_results = dense_retriever.search(query, k=top_k)
    sparse_results = sparse_retriever.search(query, k=top_k)

    # ── Step 3: union of candidate indices ────────────────────────────────
    candidate_indices = set()
    for idx, _ in dense_results:
        candidate_indices.add(idx)
    for idx, _ in sparse_results:
        candidate_indices.add(idx)

    # ── Step 4: score every candidate in BOTH systems ─────────────────────
    dense_scores_dict = {}
    sparse_scores_dict = {}
    for idx in candidate_indices:
        dense_scores_dict[idx] = dense_retriever.score_document(query, idx)
        sparse_scores_dict[idx] = sparse_retriever.score_document(query, idx)

    # ── Step 5–7: RRF fusion (returns ranks for feature extraction) ───────
    fused_results, dense_ranks, sparse_ranks = fuse_scores(
        dense_scores_dict, sparse_scores_dict, k=top_k
    )

    # ── Step 8: map back to text + collect embeddings ─────────────────────
    result_indices = [idx for idx, _ in fused_results]
    result_docs = [documents[idx] for idx in result_indices]
    result_scores = [score for _, score in fused_results]

    # Get embeddings for context-density feature
    doc_embeddings = dense_retriever.get_embeddings(result_indices)

    return RetrievalResult(
        docs=result_docs,
        scores=result_scores,
        indices=result_indices,
        dense_ranks=dense_ranks,
        sparse_ranks=sparse_ranks,
        doc_embeddings=doc_embeddings,
    )
