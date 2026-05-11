"""
LLM Response Cache
===================
Persistent disk cache for LLM responses to avoid re-calling APIs during
iterative development and repeated experiments.

Research purpose:
    Label collection (experiments/collect_labels.py) requires calling both
    cheap and full LLMs on every query.  Re-running this step during
    debugging or parameter sweeps would be prohibitively expensive without
    caching.  This module guarantees that each unique (model, prompt) pair
    is called exactly once.

Cache format:
    Each entry is stored as a JSON file in .cache/ with the filename
    derived from the SHA-256 hash of (model_name + prompt).  This avoids
    filesystem issues with long prompts and ensures deterministic lookups.
"""

import json
import hashlib
from pathlib import Path

from config import CACHE_DIR


def _cache_key(model: str, prompt: str) -> str:
    """Deterministic hash key for a (model, prompt) pair."""
    raw = f"{model}|||{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def get_cached(model: str, prompt: str) -> dict | None:
    """Look up a cached LLM response.

    Returns:
        dict with keys {"response": str, "latency": float} if cached,
        None otherwise.
    """
    path = _cache_path(_cache_key(model, prompt))
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def set_cached(model: str, prompt: str, response: str, latency: float) -> None:
    """Store an LLM response in the cache."""
    key = _cache_key(model, prompt)
    path = _cache_path(key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"response": response, "latency": latency}, f)


def cached_llm_call(model: str, prompt: str, llm_fn) -> tuple[str, float]:
    """Call an LLM function with caching.

    If the (model, prompt) pair has been seen before, return the cached
    response.  Otherwise call llm_fn(prompt), cache the result, and return it.

    Args:
        model:   model name string (used as part of cache key)
        prompt:  the full prompt string
        llm_fn:  callable(prompt) -> (response_str, latency_float)

    Returns:
        (response_text, latency_seconds)
    """
    cached = get_cached(model, prompt)
    if cached is not None:
        return cached["response"], cached["latency"]

    response, latency = llm_fn(prompt)
    set_cached(model, prompt, response, latency)
    return response, latency
