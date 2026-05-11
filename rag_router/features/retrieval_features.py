"""
Retrieval Geometry Feature Extractor
=====================================
THE CORE NOVEL MODULE — this is the paper's main contribution.

Research motivation:
    We hypothesize that the shape of the retrieval score distribution,
    computed entirely from the retrieval step (no LLM call required),
    encodes query difficulty.  This module extracts a feature vector
    from the score distribution that is used to train a pre-generation
    routing classifier.

Features extracted (10 total):
    1.  score_gap          — rank1_score − rank2_score (sharpness of top match)
    2.  score_mean         — mean of top-k scores (overall retrieval quality)
    3.  score_variance     — variance of top-k scores (distribution spread)
    4.  score_entropy      — Shannon entropy of normalised top-k scores
    5.  top_score          — absolute similarity of best match
    6.  score_ratio        — rank1 / (rank2 + ε) (dominance ratio)
    7.  low_score_fraction — fraction of top-k docs below mean score
    8.  retrieval_hit      — binary: is top score ≥ RETRIEVAL_HIT_THRESHOLD?
    9.  bm25_dense_agreement — Spearman ρ between BM25 and dense rankings
                               (agreement ⇒ easier query)
    10. context_density    — mean pairwise cosine similarity of top-k doc
                            embeddings (coherent context ⇒ easier query)

Prior work contrast:
    FrugalGPT (Chen et al., 2023) uses *post-generation* signals.
    RouteLLM (Ong et al., 2024) uses human preference labels.
    Our features require zero LLM calls and zero human annotation —
    they are computed entirely from the retrieval step.
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.stats import spearmanr

from config import RETRIEVAL_HIT_THRESHOLD, TOP_K

# Canonical feature ordering — used everywhere for consistency
RETRIEVAL_FEATURE_NAMES = [
    "score_gap", "score_mean", "score_variance", "score_entropy",
    "top_score", "score_ratio", "low_score_fraction", "retrieval_hit",
    "bm25_dense_agreement", "context_density",
]


def extract_retrieval_features(
    fused_scores: list[float],
    bm25_ranks: list[int],
    dense_ranks: list[int],
    doc_embeddings: np.ndarray | None,
    top_k: int = TOP_K,
) -> dict:
    """Extract geometry features from the retrieval score distribution.

    Args:
        fused_scores:   RRF-fused scores for top-k documents, descending order.
        bm25_ranks:     Original BM25 rank of each of the top-k documents.
        dense_ranks:    Original dense rank of each of the top-k documents.
        doc_embeddings: (top_k, embedding_dim) array of retrieved doc embeddings.
        top_k:          Number of retrieved documents.

    Returns:
        dict of named features — one row in the training dataset.
    """
    scores = np.array(fused_scores[:top_k], dtype=np.float64)
    if len(scores) == 0:
        return _zero_features()

    eps = 1e-9
    norm_scores = scores / (scores.sum() + eps)

    features = {}

    # 1. Score gap — key hypothesis feature
    #    Sharp peak ⇒ corpus has a clear answer ⇒ cheap model likely succeeds
    features["score_gap"] = (
        float(scores[0] - scores[1]) if len(scores) > 1 else float(scores[0])
    )

    # 2–3. Mean and variance of distribution
    features["score_mean"] = float(np.mean(scores))
    features["score_variance"] = float(np.var(scores))

    # 4. Shannon entropy of normalised scores
    #    Low entropy ⇒ peaked distribution ⇒ high confidence retrieval
    features["score_entropy"] = float(scipy_entropy(norm_scores + eps))

    # 5. Absolute top score
    features["top_score"] = float(scores[0])

    # 6. Dominance ratio — how much does rank-1 dominate rank-2?
    features["score_ratio"] = (
        float(scores[0] / (scores[1] + eps)) if len(scores) > 1 else 10.0
    )

    # 7. Fraction of docs below mean (spread of low-quality retrievals)
    features["low_score_fraction"] = float(np.mean(scores < np.mean(scores)))

    # 8. Binary retrieval hit
    features["retrieval_hit"] = int(scores[0] >= RETRIEVAL_HIT_THRESHOLD)

    # 9. BM25-dense rank agreement (Spearman correlation)
    #    High agreement ⇒ both retrievers found the same answer ⇒ easier
    if len(bm25_ranks) >= 3 and len(dense_ranks) >= 3:
        n = min(len(bm25_ranks), len(dense_ranks))
        corr, _ = spearmanr(bm25_ranks[:n], dense_ranks[:n])
        features["bm25_dense_agreement"] = float(corr) if not np.isnan(corr) else 0.0
    else:
        features["bm25_dense_agreement"] = 0.0

    # 10. Context density — mean pairwise cosine sim of retrieved docs
    #     High density ⇒ docs agree ⇒ coherent context ⇒ easier
    if doc_embeddings is not None and len(doc_embeddings) >= 2:
        norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = doc_embeddings / norms
        sim_matrix = normed @ normed.T
        upper = sim_matrix[np.triu_indices(len(normed), k=1)]
        features["context_density"] = float(np.mean(upper))
    else:
        features["context_density"] = 0.0

    return features


def _zero_features() -> dict:
    """Return zero-valued features when retrieval completely fails.

    Research note: max entropy signals worst-case uncertainty; ratio of 1.0
    means no dominance; low_score_fraction of 1.0 means all docs are bad.
    """
    return {
        "score_gap": 0.0,
        "score_mean": 0.0,
        "score_variance": 0.0,
        "score_entropy": 5.0,           # High entropy = worst case
        "top_score": 0.0,
        "score_ratio": 1.0,
        "low_score_fraction": 1.0,
        "retrieval_hit": 0,
        "bm25_dense_agreement": 0.0,
        "context_density": 0.0,
    }


def feature_vector(features: dict) -> np.ndarray:
    """Convert feature dict to numpy array in canonical order.

    This ordering is used by both training and inference — never
    reorder without updating RETRIEVAL_FEATURE_NAMES.
    """
    return np.array(
        [features[k] for k in RETRIEVAL_FEATURE_NAMES], dtype=np.float32
    )
