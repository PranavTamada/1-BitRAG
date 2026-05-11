"""
Baseline Systems for Comparison
=================================
Every claim in the paper must be compared against these baselines.
The paper's main result table has columns:
    System | BERTScore F1 | Full LLM % | Latency (ms)

Baselines:
    1. always_cheap   : Always use TinyLlama. Zero API cost. (lower bound)
    2. always_full    : Always use Llama-3.3-70B. Max cost. (upper bound)
    3. random_routing : Route randomly at matched cost fraction.
    4. frugal_gpt     : FrugalGPT-style: always call cheap, escalate if
                        output length < threshold OR uncertainty detected.
                        (closest prior work baseline)
    5. heuristic_v1   : Original 1-BitRAG hand-weighted heuristic system
                        (40/30/15/15 weights). Shows improvement over v1.
    6. pre_only       : Mode A -- pre-router only, no post-gen check.
    7. post_only      : Mode B -- always call cheap, calibrated post-router.
    8. rag_router     : Mode C -- FULL SYSTEM (pre + post routing).

Architecture note:
    All baselines share the same retrieval infrastructure. The ONLY
    difference is the routing decision logic. This ensures fair comparison.
"""

import re
import time
import numpy as np
from dataclasses import dataclass

from config import (
    CHEAP_MODEL, FULL_MODEL,
    DEFAULT_ROUTING_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD,
    RANDOM_STATE,
)
from llm.cheap_llm import run_cheap_llm
from llm.full_llm import run_full_llm
from utils.cache import cached_llm_call
from utils.prompt import build_summary_prompt, build_direct_prompt


@dataclass
class BaselineResult:
    """Output of a single baseline run on one query."""
    answer: str
    decision: str       # "cheap" or "full"
    latency: float      # seconds


# ── Uncertainty detection for FrugalGPT baseline ────────────────────────────
_FRUGAL_UNCERTAINTY = [
    "i don't know", "i'm not sure", "cannot determine", "unclear",
    "not provided", "no information", "cannot answer",
]


def _frugal_is_uncertain(answer: str) -> bool:
    """Simple FrugalGPT-style uncertainty heuristic."""
    lower = answer.lower()
    if len(answer.strip()) < 20:
        return True
    return any(p in lower for p in _FRUGAL_UNCERTAINTY)


# ── v1 Heuristic confidence (hand-weighted, 40/30/15/15) ────────────────────
_V1_UNCERTAINTY_PHRASES = [
    "i don't know", "cannot determine", "not provided", "unclear",
    "not sure", "cannot answer", "no information", "not mentioned",
]
_V1_HEDGE_WORDS = ["maybe", "perhaps", "possibly", "might", "could be",
                   "it seems", "it appears", "likely"]
_V1_ASSERTIVE_PATTERNS = [
    r"\b(is|are|was|were|has|have|had|will|does|do|did)\b",
    r"\b\d{4}\b",
    r"\b(because|therefore|thus|hence)\b",
    r"\b(the answer is|it is|this is)\b",
]


def _v1_heuristic_confidence(query: str, answer: str) -> float:
    """Original 1-BitRAG v1 hand-weighted confidence (40/30/15/15)."""
    if not answer or not answer.strip():
        return 0.0
    norm = re.sub(r"\s+", " ", answer.lower().strip())

    # Relevance (40%)
    q_tokens = set(re.findall(r"\w+", query.lower()))
    a_tokens = set(re.findall(r"\w+", norm))
    stop = {"a", "an", "the", "is", "are", "was", "in", "for", "on", "to",
            "of", "and", "or", "it", "i", "you", "we", "they", "not", "no"}
    q_content = {t for t in q_tokens if t not in stop}
    a_content = {t for t in a_tokens if t not in stop}
    relevance = len(q_content & a_content) / len(q_content) if q_content else 0.5

    # Uncertainty (30%)
    unc_hits = sum(1 for p in _V1_UNCERTAINTY_PHRASES if p in norm)
    uncertainty = max(0.0, 1.0 - unc_hits * 0.45)

    # Hedging (15%)
    hedge_hits = sum(1 for w in _V1_HEDGE_WORDS if w in norm)
    hedging = max(0.0, 1.0 - hedge_hits * 0.2)

    # Assertiveness (15%)
    assert_hits = sum(1 for p in _V1_ASSERTIVE_PATTERNS if re.search(p, norm))
    assertiveness = min(assert_hits / len(_V1_ASSERTIVE_PATTERNS), 1.0)

    raw = 0.40 * relevance + 0.30 * uncertainty + 0.15 * hedging + 0.15 * assertiveness

    # Length penalty
    if len(answer.strip()) < 15:
        raw *= 0.4

    return max(0.0, min(1.0, raw))


# ═════════════════════════════════════════════════════════════════════════════
# Baseline runner functions
# ═════════════════════════════════════════════════════════════════════════════

def run_always_cheap(
    query: str, prompt: str, **kwargs
) -> BaselineResult:
    """Baseline 1: Always use the cheap LLM."""
    answer, latency = cached_llm_call(CHEAP_MODEL, prompt, run_cheap_llm)
    return BaselineResult(answer=answer, decision="cheap", latency=latency)


def run_always_full(
    query: str, prompt: str, **kwargs
) -> BaselineResult:
    """Baseline 2: Always use the full LLM."""
    direct = build_direct_prompt(query)
    answer, latency = cached_llm_call(FULL_MODEL, direct, run_full_llm)
    return BaselineResult(answer=answer, decision="full", latency=latency)


def run_random_routing(
    query: str, prompt: str, full_fraction: float = 0.5,
    rng: np.random.RandomState = None, **kwargs
) -> BaselineResult:
    """Baseline 3: Route randomly at a matched cost fraction.

    Research purpose: if random routing at X% cost achieves similar
    accuracy to our system, then routing adds no value. We expect
    random to be significantly worse.
    """
    if rng is None:
        rng = np.random.RandomState(RANDOM_STATE)

    if rng.random() < full_fraction:
        direct = build_direct_prompt(query)
        answer, latency = cached_llm_call(FULL_MODEL, direct, run_full_llm)
        return BaselineResult(answer=answer, decision="full", latency=latency)
    else:
        answer, latency = cached_llm_call(CHEAP_MODEL, prompt, run_cheap_llm)
        return BaselineResult(answer=answer, decision="cheap", latency=latency)


def run_frugal_gpt(
    query: str, prompt: str, **kwargs
) -> BaselineResult:
    """Baseline 4: FrugalGPT-style -- call cheap first, escalate on uncertainty.

    This is the closest prior work baseline. Key difference from RAG-Router:
    FrugalGPT always calls the cheap LLM, then decides. RAG-Router can
    skip the cheap call entirely via pre-routing.
    """
    cheap_answer, cheap_latency = cached_llm_call(
        CHEAP_MODEL, prompt, run_cheap_llm
    )

    if _frugal_is_uncertain(cheap_answer):
        direct = build_direct_prompt(query)
        full_answer, full_latency = cached_llm_call(
            FULL_MODEL, direct, run_full_llm
        )
        return BaselineResult(
            answer=full_answer, decision="full",
            latency=cheap_latency + full_latency,
        )
    return BaselineResult(
        answer=cheap_answer, decision="cheap", latency=cheap_latency,
    )


def run_heuristic_v1(
    query: str, prompt: str, **kwargs
) -> BaselineResult:
    """Baseline 5: Original 1-BitRAG v1 hand-weighted heuristics.

    Uses the 40/30/15/15 weighted confidence score with manual threshold.
    """
    cheap_answer, cheap_latency = cached_llm_call(
        CHEAP_MODEL, prompt, run_cheap_llm
    )

    confidence = _v1_heuristic_confidence(query, cheap_answer)

    if confidence < 0.7:
        direct = build_direct_prompt(query)
        full_answer, full_latency = cached_llm_call(
            FULL_MODEL, direct, run_full_llm
        )
        return BaselineResult(
            answer=full_answer, decision="full",
            latency=cheap_latency + full_latency,
        )
    return BaselineResult(
        answer=cheap_answer, decision="cheap", latency=cheap_latency,
    )


def run_pre_only(
    query: str, prompt: str, pre_router=None,
    feature_vec: np.ndarray = None, **kwargs
) -> BaselineResult:
    """Mode A: Pre-routing only -- no post-gen check.

    Routes based purely on retrieval geometry features.
    If pre-router says "cheap", use cheap answer with no second-guessing.
    """
    assert pre_router is not None and feature_vec is not None

    decision, confidence = pre_router.route(feature_vec)

    if decision == "full":
        direct = build_direct_prompt(query)
        answer, latency = cached_llm_call(FULL_MODEL, direct, run_full_llm)
        return BaselineResult(answer=answer, decision="full", latency=latency)
    else:
        answer, latency = cached_llm_call(CHEAP_MODEL, prompt, run_cheap_llm)
        return BaselineResult(answer=answer, decision="cheap", latency=latency)


def run_post_only(
    query: str, prompt: str, post_router=None, **kwargs
) -> BaselineResult:
    """Mode B: Post-routing only -- always call cheap, calibrated gate.

    Always generates with cheap LLM first, then uses the trained
    (calibrated) post-router to decide if escalation is needed.
    """
    assert post_router is not None

    cheap_answer, cheap_latency = cached_llm_call(
        CHEAP_MODEL, prompt, run_cheap_llm
    )

    escalate, confidence = post_router.should_escalate(query, cheap_answer)

    if escalate:
        direct = build_direct_prompt(query)
        full_answer, full_latency = cached_llm_call(
            FULL_MODEL, direct, run_full_llm
        )
        return BaselineResult(
            answer=full_answer, decision="full",
            latency=cheap_latency + full_latency,
        )
    return BaselineResult(
        answer=cheap_answer, decision="cheap", latency=cheap_latency,
    )


def run_rag_router(
    query: str, prompt: str,
    pre_router=None, post_router=None,
    feature_vec: np.ndarray = None, **kwargs
) -> BaselineResult:
    """Mode C: FULL SYSTEM -- pre-routing + post-gen confidence gate.

    The complete RAG-Router pipeline:
    1. Pre-router classifies using retrieval geometry features
    2. If pre-router says "full" -> immediate escalation (zero cheap LLM cost)
    3. If pre-router says "cheap" -> generate with cheap LLM
    4. Post-router checks cheap answer confidence
    5. If post-router says escalate -> call full LLM
    6. Otherwise -> return cheap answer
    """
    assert pre_router is not None
    assert post_router is not None
    assert feature_vec is not None

    # Step 1: Pre-routing
    pre_decision, pre_confidence = pre_router.route(feature_vec)

    if pre_decision == "full":
        # Step 2: Immediate escalation -- no cheap LLM call at all
        direct = build_direct_prompt(query)
        answer, latency = cached_llm_call(FULL_MODEL, direct, run_full_llm)
        return BaselineResult(answer=answer, decision="full", latency=latency)

    # Step 3: Generate with cheap LLM
    cheap_answer, cheap_latency = cached_llm_call(
        CHEAP_MODEL, prompt, run_cheap_llm
    )

    # Step 4-5: Post-gen confidence gate
    escalate, post_confidence = post_router.should_escalate(
        query, cheap_answer
    )

    if escalate:
        direct = build_direct_prompt(query)
        full_answer, full_latency = cached_llm_call(
            FULL_MODEL, direct, run_full_llm
        )
        return BaselineResult(
            answer=full_answer, decision="full",
            latency=cheap_latency + full_latency,
        )

    # Step 6: Return cheap answer
    return BaselineResult(
        answer=cheap_answer, decision="cheap", latency=cheap_latency,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Baseline registry
# ═════════════════════════════════════════════════════════════════════════════
BASELINE_REGISTRY = {
    "always_cheap": run_always_cheap,
    "always_full": run_always_full,
    "random_routing": run_random_routing,
    "frugal_gpt": run_frugal_gpt,
    "heuristic_v1": run_heuristic_v1,
    "pre_only": run_pre_only,
    "post_only": run_post_only,
    "rag_router": run_rag_router,
}
