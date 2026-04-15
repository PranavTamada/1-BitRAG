"""
Confidence gating for 1-BitRAG project.

Evaluates the cheap model's answer to decide whether the expensive
full LLM needs to be called.  The gate uses fast, local heuristics
so it adds negligible latency.
"""
import re
import math

# ---------------------------------------------------------------------------
# Signal lists
# ---------------------------------------------------------------------------

_UNCERTAINTY_PHRASES = [
    "i don't know", "i do not know", "not sure", "cannot answer",
    "can't answer", "unable to answer", "no information", "not provided",
    "not mentioned", "does not contain", "doesn't contain",
    "not explicitly stated", "not enough information",
    "insufficient information", "cannot determine", "can't determine",
    "unclear", "no relevant", "not available", "outside the context",
    "beyond the context", "the context does not", "the document does not",
    "not in the context", "cannot be determined", "can't be determined",
]

_HEDGING_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could be",
    "it seems", "it appears", "likely", "unlikely", "uncertain",
    "not certain", "speculate", "guess",
]

# Patterns that indicate a direct, assertive answer
_ASSERTIVE_PATTERNS = [
    r"\b(is|are|was|were|has|have|had|will|does|do|did)\b",  # strong verbs
    r"\b\d{4}\b",                   # years
    r"\b\d+(\.\d+)?\s*(%|percent|kg|km|miles|dollars|usd|gbp)\b",  # quantities
    r"\b(because|therefore|thus|hence|as a result)\b",             # causal reasoning
    r"\b(the answer is|it is|this is|that is|they are)\b",         # direct assertion
]

# Minimum answer length (in characters) to be considered substantive
_MIN_ANSWER_LENGTH = 15

# Weight of each component in the final score
_W_RELEVANCE   = 0.40
_W_UNCERTAINTY = 0.30
_W_HEDGING     = 0.15
_W_ASSERTIVE   = 0.15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _token_overlap_score(query_norm: str, answer_norm: str) -> float:
    """
    Jaccard-like overlap between content words in query and answer.
    Stop words are excluded so filler doesn't inflate the score.
    Returns a value in [0.0, 1.0].
    """
    _STOP = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "or", "and", "but", "if", "that", "this", "it", "its",
        "i", "you", "we", "they", "he", "she", "what", "which", "who",
        "how", "when", "where", "why", "not", "no",
    }
    q_tokens = {t for t in re.findall(r"\w+", query_norm) if t not in _STOP}
    a_tokens = {t for t in re.findall(r"\w+", answer_norm) if t not in _STOP}

    if not q_tokens:
        return 0.5  # can't judge without a query — stay neutral

    overlap = q_tokens & a_tokens
    # Precision-weighted: how much of the query is covered by the answer?
    return len(overlap) / len(q_tokens)


def _uncertainty_score(norm: str) -> float:
    """
    Returns 1.0 (fully confident) down to 0.0 based on how many
    uncertainty phrases appear.  Uses diminishing returns so a single
    stray phrase doesn't collapse the score.
    """
    hits = sum(1 for p in _UNCERTAINTY_PHRASES if p in norm)
    if hits == 0:
        return 1.0
    # Logarithmic decay: 1 hit → ~0.50, 2 hits → ~0.25, 3+ → near 0
    return max(0.0, 1.0 - math.log1p(hits) / math.log1p(2) * 0.9)


def _hedging_score(norm: str) -> float:
    """
    Returns 1.0 (no hedging) down to 0.0 based on the proportion of
    hedging words relative to *content* words, not total words.
    """
    content_words = [
        w for w in re.findall(r"\w+", norm)
        if len(w) > 2          # drop trivial tokens
    ]
    if not content_words:
        return 0.0

    hits = sum(1 for w in _HEDGING_WORDS if w in norm)
    # Ratio against content-word count, then scale
    ratio = hits / len(content_words)
    return max(0.0, 1.0 - ratio * 8.0)


def _assertiveness_score(norm: str) -> float:
    """
    Rewards answers that contain direct assertions, numbers, causal
    language, or named entities (heuristically: title-cased words).
    Returns a value in [0.0, 1.0].
    """
    pattern_hits = sum(
        1 for pat in _ASSERTIVE_PATTERNS if re.search(pat, norm)
    )
    # Named-entity proxy: count title-cased words in the *original* (pre-norm)
    # We use the normalised form here but check capitalisation separately.
    # Since we work on norm (lowercase), use digit/special entity signals only.
    score = min(pattern_hits / len(_ASSERTIVE_PATTERNS), 1.0)
    return score

def compute_confidence(answer: str, query: str = "") -> float:
    """Return a confidence score in [0.0, 1.0] for a cheap-model answer.

    Components (weighted sum):
        1. Relevance      — token overlap between query and answer (0.40)
        2. Uncertainty    — absence of "I don't know"-style phrases  (0.30)
        3. Hedging        — absence of speculative hedging language   (0.15)
        4. Assertiveness  — presence of direct, factual language      (0.15)

    Hard penalties applied before weighted scoring:
        - Empty / whitespace-only answer → 0.0
        - Answer shorter than _MIN_ANSWER_LENGTH → ×0.4 scale
        - Answer that merely echoes the query → ×0.3 scale

    Args:
        answer: the cheap model's response text.
        query:  the original user query (strongly recommended). When
                omitted the relevance component defaults to 0.5.

    Returns:
        float: confidence score, where higher = more confident.
    """
    # Hard gate: empty answer
    if not answer or not answer.strip():
        return 0.0

    norm_answer = _normalize(answer)
    norm_query  = _normalize(query) if query else ""

    # Hard gate: trivially short answer
    length_factor = 1.0
    if len(answer.strip()) < _MIN_ANSWER_LENGTH:
        length_factor = 0.4

    # Hard gate: answer is almost entirely a copy of the query
    echo_factor = 1.0
    if norm_query and norm_answer:
        q_words = set(re.findall(r"\w+", norm_query))
        a_words = set(re.findall(r"\w+", norm_answer))
        if q_words and len(a_words) > 0:
            echo_ratio = len(q_words & a_words) / len(a_words)
            if echo_ratio > 0.85:   # >85 % of answer words came from query
                echo_factor = 0.3

    # Component scores
    relevance     = _token_overlap_score(norm_query, norm_answer) if norm_query else 0.5
    uncertainty   = _uncertainty_score(norm_answer)
    hedging       = _hedging_score(norm_answer)
    assertiveness = _assertiveness_score(norm_answer)

    raw = (
        _W_RELEVANCE   * relevance   +
        _W_UNCERTAINTY * uncertainty +
        _W_HEDGING     * hedging     +
        _W_ASSERTIVE   * assertiveness
    )

    return max(0.0, min(1.0, raw * length_factor * echo_factor))


def needs_full_llm(
    fused_scores,          # list of (doc_index, fused_score) from fuse_scores()
    llm_confidence,        # float from cheap answerer
    retrieval_threshold=0.2,
    confidence_threshold=0.7,
    top_k=3
):
    """
    Decide whether to escalate to full LLM.
    
    Uses both retrieval quality and LLM confidence as a combined signal.
    """
    # Average the top-k retrieval scores as a quality signal
    top_scores = [score for score in fused_scores[:top_k]]
    avg_retrieval_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

    retrieval_ok = avg_retrieval_score >= retrieval_threshold
    confidence_ok = llm_confidence >= confidence_threshold

    # Only skip full LLM if BOTH signals are good
    escalate = not (retrieval_ok and confidence_ok)

    return escalate