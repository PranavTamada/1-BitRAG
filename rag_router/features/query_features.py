"""
Query-Side Feature Extractor
==============================
Complexity signals derived from the query text alone, combined with
retrieval features in the pre-router.

Research purpose:
    Query features capture structural difficulty (negation, multi-hop,
    comparison) that complements retrieval-geometry features.  The ablation
    in Section 5 of the paper shows their marginal contribution above
    retrieval features alone.

Features extracted (8 total):
    1. query_length       — token count
    2. query_entropy      — Shannon entropy of token unigram distribution
    3. has_negation       — binary: contains "not", "never", "no", etc.
    4. has_conditional    — binary: contains "if", "when", "unless", etc.
    5. question_count     — number of "?" (multi-question ⇒ harder)
    6. has_comparison     — binary: contains "vs", "difference", "compare"
    7. avg_word_length    — proxy for domain-specific vocabulary
    8. named_entity_count — approximate count of capitalised multi-word phrases
"""

import re
import numpy as np
from scipy.stats import entropy as scipy_entropy

NEGATION_WORDS = {"not", "never", "no", "without", "neither", "nor", "cannot", "can't"}
CONDITIONAL_WORDS = {"if", "when", "unless", "whether", "assuming", "provided"}
COMPARISON_WORDS = {"vs", "versus", "difference", "compare", "better", "worse", "than"}

# Canonical ordering — must match query_feature_vector()
QUERY_FEATURE_NAMES = [
    "query_length", "query_entropy", "has_negation", "has_conditional",
    "question_count", "has_comparison", "avg_word_length", "named_entity_count",
]


def extract_query_features(query: str) -> dict:
    """Extract complexity signals from the raw query text.

    Args:
        query: the user's search query.

    Returns:
        dict of named features.
    """
    tokens = query.lower().split()
    if not tokens:
        return {k: 0.0 for k in QUERY_FEATURE_NAMES}

    # Token unigram distribution for entropy
    token_counts: dict[str, int] = {}
    for t in tokens:
        token_counts[t] = token_counts.get(t, 0) + 1
    probs = np.array(list(token_counts.values()), dtype=float)
    probs /= probs.sum()

    return {
        "query_length": float(len(tokens)),
        "query_entropy": float(scipy_entropy(probs + 1e-9)),
        "has_negation": float(bool(set(tokens) & NEGATION_WORDS)),
        "has_conditional": float(bool(set(tokens) & CONDITIONAL_WORDS)),
        "question_count": float(query.count("?")),
        "has_comparison": float(bool(set(tokens) & COMPARISON_WORDS)),
        "avg_word_length": float(np.mean([len(t) for t in tokens])),
        "named_entity_count": float(
            len(re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+", query))
        ),
    }


def query_feature_vector(features: dict) -> np.ndarray:
    """Convert feature dict to numpy array in canonical order."""
    return np.array(
        [features[k] for k in QUERY_FEATURE_NAMES], dtype=np.float32
    )
