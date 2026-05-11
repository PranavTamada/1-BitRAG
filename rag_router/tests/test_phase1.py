"""
Phase 1 Verification Tests
============================
Unit tests for the foundation layer: feature extractors, retriever
interfaces, and config import chain.

Run:  python -m pytest tests/test_phase1.py -v
  or: python tests/test_phase1.py           (standalone)

Expected: all assertions pass, feature shapes and value ranges are correct.
"""

import sys
import os
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TOP_K, RETRIEVAL_HIT_THRESHOLD, RRF_K


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Config loads without error
# ─────────────────────────────────────────────────────────────────────────────
def test_config():
    assert TOP_K == 10
    assert RRF_K == 50
    assert 0 < RETRIEVAL_HIT_THRESHOLD < 1.0
    print("[PASS] test_config")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Retrieval feature extractor — shapes and value ranges
# ─────────────────────────────────────────────────────────────────────────────
def test_retrieval_features():
    from features.retrieval_features import (
        extract_retrieval_features, feature_vector, RETRIEVAL_FEATURE_NAMES,
    )

    # Simulate a top-5 score distribution with a clear peak
    fused_scores = [0.040, 0.025, 0.020, 0.018, 0.015]
    bm25_ranks = [1, 3, 2, 5, 4]
    dense_ranks = [1, 2, 4, 3, 5]
    doc_embeddings = np.random.randn(5, 384).astype(np.float32)

    feats = extract_retrieval_features(
        fused_scores, bm25_ranks, dense_ranks, doc_embeddings, top_k=5
    )

    # All 10 features present
    assert len(feats) == 10, f"Expected 10 features, got {len(feats)}"
    for name in RETRIEVAL_FEATURE_NAMES:
        assert name in feats, f"Missing feature: {name}"

    # score_gap should be positive for a peaked distribution
    assert feats["score_gap"] > 0, f"score_gap should be > 0, got {feats['score_gap']}"

    # score_mean should be between min and max scores
    assert 0.015 <= feats["score_mean"] <= 0.040

    # entropy should be finite and non-negative
    assert 0 <= feats["score_entropy"] < 100

    # retrieval_hit: top score 0.040 > threshold 0.035
    assert feats["retrieval_hit"] == 1

    # Feature vector conversion
    vec = feature_vector(feats)
    assert vec.shape == (10,), f"Expected shape (10,), got {vec.shape}"
    assert vec.dtype == np.float32

    print("[PASS] test_retrieval_features")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Retrieval feature extractor — zero / edge cases
# ─────────────────────────────────────────────────────────────────────────────
def test_retrieval_features_edge_cases():
    from features.retrieval_features import extract_retrieval_features, feature_vector

    # Empty scores → zero features
    feats = extract_retrieval_features([], [], [], None, top_k=5)
    assert feats["score_entropy"] == 5.0, "Empty scores should give max entropy"
    assert feats["retrieval_hit"] == 0
    vec = feature_vector(feats)
    assert vec.shape == (10,)

    # Single score
    feats = extract_retrieval_features([0.05], [1], [1], None, top_k=1)
    assert feats["score_gap"] == 0.05
    assert feats["score_ratio"] == 10.0  # default for single-doc

    # All identical scores (flat distribution)
    feats = extract_retrieval_features(
        [0.02, 0.02, 0.02], [1, 2, 3], [2, 1, 3], None, top_k=3
    )
    assert feats["score_gap"] == 0.0
    assert feats["score_variance"] < 1e-10

    print("[PASS] test_retrieval_features_edge_cases")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Query feature extractor — shapes and value ranges
# ─────────────────────────────────────────────────────────────────────────────
def test_query_features():
    from features.query_features import (
        extract_query_features, query_feature_vector, QUERY_FEATURE_NAMES,
    )

    # Simple query
    feats = extract_query_features("What causes diabetes?")
    assert len(feats) == 8, f"Expected 8 features, got {len(feats)}"
    for name in QUERY_FEATURE_NAMES:
        assert name in feats, f"Missing feature: {name}"
    assert feats["query_length"] == 3.0
    assert feats["question_count"] == 1.0
    assert feats["has_negation"] == 0.0
    assert feats["has_comparison"] == 0.0

    vec = query_feature_vector(feats)
    assert vec.shape == (8,), f"Expected shape (8,), got {vec.shape}"
    assert vec.dtype == np.float32

    # Complex query with negation, conditional, comparison
    feats2 = extract_query_features(
        "If a patient does not respond to treatment, is Drug A better than Drug B?"
    )
    assert feats2["has_negation"] == 1.0
    assert feats2["has_conditional"] == 1.0
    assert feats2["has_comparison"] == 1.0
    assert feats2["question_count"] == 1.0

    # Empty query
    feats3 = extract_query_features("")
    assert feats3["query_length"] == 0.0
    vec3 = query_feature_vector(feats3)
    assert vec3.shape == (8,)

    print("[PASS] test_query_features")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: RRF fusion returns correct types and ranks
# ─────────────────────────────────────────────────────────────────────────────
def test_fusion():
    from retriever.fusion import fuse_scores

    dense_scores = {0: 0.9, 1: 0.7, 2: 0.5}
    sparse_scores = {0: 5.0, 1: 3.0, 3: 4.0}

    fused, dense_ranks, sparse_ranks = fuse_scores(dense_scores, sparse_scores, k=3)

    # Returns list of (idx, score) tuples
    assert isinstance(fused, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in fused)
    assert len(fused) <= 3

    # Dense ranks should be 1-based
    assert dense_ranks[0] == 1   # highest dense score
    assert dense_ranks[2] == 3   # lowest dense score

    # Sparse ranks should be 1-based
    assert sparse_ranks[0] == 1  # highest sparse score

    # Fused scores should be descending
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)

    print("[PASS] test_fusion")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: RetrievalResult dataclass structure
# ─────────────────────────────────────────────────────────────────────────────
def test_retrieval_result_dataclass():
    from retriever.retrieve import RetrievalResult
    import numpy as np

    result = RetrievalResult(
        docs=["doc1", "doc2"],
        scores=[0.5, 0.3],
        indices=[0, 1],
        dense_ranks={0: 1, 1: 2},
        sparse_ranks={0: 2, 1: 1},
        doc_embeddings=np.zeros((2, 384)),
    )
    assert len(result.docs) == 2
    assert result.scores[0] > result.scores[1]
    assert result.doc_embeddings.shape == (2, 384)

    print("[PASS] test_retrieval_result_dataclass")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Combined feature vector dimensionality
# ─────────────────────────────────────────────────────────────────────────────
def test_combined_feature_vector():
    from features.retrieval_features import extract_retrieval_features, feature_vector
    from features.query_features import extract_query_features, query_feature_vector

    r_feats = extract_retrieval_features(
        [0.04, 0.03, 0.02], [1, 2, 3], [1, 3, 2], np.random.randn(3, 384), top_k=3
    )
    q_feats = extract_query_features("What is hypertension?")

    r_vec = feature_vector(r_feats)
    q_vec = query_feature_vector(q_feats)
    combined = np.concatenate([r_vec, q_vec])

    # Pre-router expects 18 features: 10 retrieval + 8 query
    assert combined.shape == (18,), f"Expected (18,), got {combined.shape}"
    assert combined.dtype == np.float32

    print("[PASS] test_combined_feature_vector")


# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_config()
    test_retrieval_features()
    test_retrieval_features_edge_cases()
    test_query_features()
    test_fusion()
    test_retrieval_result_dataclass()
    test_combined_feature_vector()
    print("\n[OK] All Phase 1 tests passed!")
