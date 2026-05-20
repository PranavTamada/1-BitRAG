"""
Baseline Systems for Comparison
=================================
Every claim in the paper must be compared against these baselines.
The paper's main result table has columns:
    System | BERTScore F1 | Full LLM % | Latency (ms)

Baselines:
    1. always_cheap       : Always use cheap LLM. Zero API cost. (lower bound)
    2. always_full        : Always use full LLM. Max cost. (upper bound)
    3. random_routing     : Route randomly at matched cost fraction.
    4. post_gen_cascade   : Post-generation cascade baseline — always call cheap
                            first, escalate to full LLM if output is short or
                            contains uncertainty phrases. This adapts the cascade
                            strategy from FrugalGPT (Chen et al., 2023) to the
                            RAG setting. Unlike original FrugalGPT (which has no
                            retrieval), all baselines here share the same RAG
                            retrieval pipeline; only the routing decision differs.
    5. pre_only           : Mode A -- pre-router only, no post-gen check.
    6. rag_router         : Mode C -- FULL SYSTEM (pre + post routing).
    7. oracle_routing     : Always picks the better model per query. (theoretical upper bound)

Prompt fairness:
    ALL baselines receive the SAME RAG prompt with retrieved context.
    The routing decision determines which *model* answers — not which
    *prompt* is used. This ensures the comparison is about model capacity,
    not information availability.
"""

import numpy as np
from dataclasses import dataclass

from config import (
    CHEAP_MODEL, FULL_MODEL,RANDOM_STATE,
)
from llm.cheap_llm import run_cheap_llm
from llm.full_llm import run_full_llm
from utils.cache import cached_llm_call

@dataclass
class BaselineResult:
    """Output of a single baseline run on one query."""
    answer: str
    decision: str
    latency: float      


# ── Uncertainty detection for post-generation cascade baseline ───────────────
# Adapted from the cascade escalation concept in FrugalGPT (Chen et al., 2023),
# applied here within a shared RAG retrieval pipeline.
_FRUGAL_UNCERTAINTY = [
    "i don't know", "i'm not sure", "cannot determine", "unclear",
    "not provided", "no information", "cannot answer",
]


def _frugal_is_uncertain(answer: str) -> bool:
    """Post-generation uncertainty heuristic for the cascade baseline."""
    lower = answer.lower()
    if len(answer.strip()) < 20:
        return True
    return any(p in lower for p in _FRUGAL_UNCERTAINTY)

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
    """Baseline 2: Always use the full LLM.

    CRITICAL FIX: uses the SAME RAG prompt as cheap LLM (not a bare
    direct prompt). Both models must receive identical information
    so that routing compares model capacity, not information access.
    """
    answer, latency = cached_llm_call(FULL_MODEL, prompt, run_full_llm)
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
        answer, latency = cached_llm_call(FULL_MODEL, prompt, run_full_llm)
        return BaselineResult(answer=answer, decision="full", latency=latency)
    else:
        answer, latency = cached_llm_call(CHEAP_MODEL, prompt, run_cheap_llm)
        return BaselineResult(answer=answer, decision="cheap", latency=latency)

def run_post_gen_cascade(
    query: str, prompt: str, **kwargs
) -> BaselineResult:
    """Baseline 4: Post-generation cascade -- call cheap first, escalate on uncertainty.

    Adapts the cascade strategy from FrugalGPT (Chen et al., 2023) to the RAG
    setting. The original FrugalGPT operates without retrieval; here, all
    baselines share the same RAG retrieval pipeline and differ only in routing.

    Escalation condition: cheap answer is short (<20 chars) OR contains
    common uncertainty phrases (e.g. "I don't know", "cannot determine").

    Key difference from RAG-Router:
        This baseline always calls the cheap LLM first, then decides.
        RAG-Router's pre-router can skip the cheap call entirely.
    """
    cheap_answer, cheap_latency = cached_llm_call(
        CHEAP_MODEL, prompt, run_cheap_llm
    )

    if _frugal_is_uncertain(cheap_answer):
        full_answer, full_latency = cached_llm_call(
            FULL_MODEL, prompt, run_full_llm
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
        answer, latency = cached_llm_call(FULL_MODEL, prompt, run_full_llm)
        return BaselineResult(answer=answer, decision="full", latency=latency)
    else:
        answer, latency = cached_llm_call(CHEAP_MODEL, prompt, run_cheap_llm)
        return BaselineResult(answer=answer, decision="cheap", latency=latency)


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
        answer, latency = cached_llm_call(FULL_MODEL, prompt, run_full_llm)
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
        full_answer, full_latency = cached_llm_call(
            FULL_MODEL, prompt, run_full_llm
        )
        return BaselineResult(
            answer=full_answer, decision="full",
            latency=cheap_latency + full_latency,
        )

    # Step 6: Return cheap answer
    return BaselineResult(
        answer=cheap_answer, decision="cheap", latency=cheap_latency,
    )


def run_oracle_routing(
    query: str, prompt: str,
    cheap_bertscore: float = 0.0, full_bertscore: float = 0.0,
    **kwargs
) -> BaselineResult:
    """Oracle: always picks the model that scored higher on this query.

    Provides the theoretical upper bound for any router — no real router
    can outperform this.  Used to contextualise RAG-Router results:
    "RAG-Router achieves X% of oracle performance."

    Note: this baseline requires pre-computed BERTScores for both models
    (from labeled_routing_data.jsonl) and cannot be used at inference time.
    """
    if cheap_bertscore >= full_bertscore:
        answer, latency = cached_llm_call(CHEAP_MODEL, prompt, run_cheap_llm)
        return BaselineResult(answer=answer, decision="cheap", latency=latency)
    else:
        answer, latency = cached_llm_call(FULL_MODEL, prompt, run_full_llm)
        return BaselineResult(answer=answer, decision="full", latency=latency)


# ═════════════════════════════════════════════════════════════════════════════
# Baseline registry
# ═════════════════════════════════════════════════════════════════════════════
BASELINE_REGISTRY = {
    "always_cheap":       run_always_cheap,
    "always_full":        run_always_full,
    "random_routing":     run_random_routing,
    "post_gen_cascade":   run_post_gen_cascade,
    "pre_only":           run_pre_only,
    "rag_router":         run_rag_router,
    "oracle_routing":     run_oracle_routing,
}
