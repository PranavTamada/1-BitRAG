"""
Full LLM — Llama-3.3-70B via Groq API
=======================================
High-capability model used for difficult queries that the cheap model
cannot handle.  API calls are expensive, so the routing system's goal
is to minimise calls to this model while preserving accuracy.

API Key Rotation:
    Reads up to 5 Groq API keys from environment variables:
        GROQ_API_KEY      ← primary key
        GROQ_API_KEY_2    ← rotated to when primary hits rate limit
        GROQ_API_KEY_3
        GROQ_API_KEY_4
        GROQ_API_KEY_5

    On every 429 rate-limit error, the next key is tried immediately
    with zero wait. Sleep only occurs after all keys in the pool have
    been cycled through once without success.

Usage — set keys in your shell before running:
    $env:GROQ_API_KEY   = "gsk_key1..."
    $env:GROQ_API_KEY_2 = "gsk_key2..."
    $env:GROQ_API_KEY_3 = "gsk_key3..."
    $env:GROQ_API_KEY_4 = "gsk_key4..."
    $env:GROQ_API_KEY_5 = "gsk_key5..."
"""

import time
import os
from groq import Groq

from config import FULL_MODEL, GROQ_API_KEY

# ── Custom exception ─────────────────────────────────────────────────────────
class AllKeysExhaustedError(RuntimeError):
    """Raised when every API key in the pool has hit its rate limit.

    Catching this in collect_labels.py causes the run to stop immediately
    and save progress, rather than crashing with an unhandled exception.
    """


# ── Key pool ──────────────────────────────────────────────────────────────────
# Collects all keys at module load time. Supports up to 5 keys.
# Keys are deduplicated and empty strings are filtered out.
def _load_key_pool() -> list[str]:
    candidates = [
        GROQ_API_KEY,                    # from config / GROQ_API_KEY env var
        os.getenv("GROQ_API_KEY_2", ""),
        os.getenv("GROQ_API_KEY_3", ""),
        os.getenv("GROQ_API_KEY_4", ""),
        os.getenv("GROQ_API_KEY_5", ""),
    ]
    seen = set()
    pool = []
    for k in candidates:
        if k and k not in seen:
            pool.append(k)
            seen.add(k)
    return pool

_KEY_POOL: list[str] = _load_key_pool()
_clients: list[Groq] = []        # One Groq client per key, built lazily
_current_key_idx: int = 0        # Index of the currently active key

MAX_RETRIES = 5   # Max total attempts across all keys per call
BASE_WAIT   = 30  # Seconds to sleep after a full cycle of all keys fails


def _get_client(idx: int) -> Groq:
    """Return (and lazily build) the Groq client for key index idx."""
    global _clients
    # Extend the list if this index hasn't been built yet
    while len(_clients) <= idx:
        _clients.append(None)
    if _clients[idx] is None:
        _clients[idx] = Groq(api_key=_KEY_POOL[idx])
    return _clients[idx]


def run_full_llm(prompt: str) -> tuple[str, float]:
    """Generate a response using the full-capability model via Groq.

    Key rotation strategy:
        - On a rate-limit (429/503) error, immediately rotate to the next
          API key in the pool — no sleep required.
        - After a full cycle through all keys fails, sleep BASE_WAIT seconds
          before retrying from the beginning of the pool.
        - Non-rate-limit errors raise immediately (no retry).

    Returns:
        (response_text, latency_seconds)
    """
    global _current_key_idx

    if not _KEY_POOL:
        raise RuntimeError(
            "No Groq API keys found. Set GROQ_API_KEY (and optionally "
            "GROQ_API_KEY_2 … GROQ_API_KEY_5) as environment variables."
        )

    start = time.time()
    last_error = None
    n_keys = len(_KEY_POOL)

    for attempt in range(MAX_RETRIES):
        client = _get_client(_current_key_idx)
        key_label = f"key[{_current_key_idx + 1}/{n_keys}]"

        try:
            response = client.chat.completions.create(
                model=FULL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            latency = time.time() - start
            return response.choices[0].message.content, latency

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            is_rate_limit = (
                "rate" in error_str or "429" in error_str
                or "503" in error_str or "500" in error_str
            )

            if not is_rate_limit:
                raise  # Non-retryable — propagate immediately

            # ── Rate limit hit: rotate to the next key ────────────────────
            next_idx = (_current_key_idx + 1) % n_keys
            completed_cycle = (next_idx == 0)   # wrapped back to key 1

            if completed_cycle:
                # Exhausted all keys in this cycle — sleep before next round
                print(
                    f"    [RATE LIMIT] All {n_keys} key(s) exhausted "
                    f"(attempt {attempt + 1}/{MAX_RETRIES}). "
                    f"Sleeping {BASE_WAIT}s before retrying... ({e})"
                )
                time.sleep(BASE_WAIT)
            else:
                print(
                    f"    [RATE LIMIT] {key_label} hit limit — "
                    f"rotating to key[{next_idx + 1}/{n_keys}]... ({e})"
                )

            _current_key_idx = next_idx

    raise AllKeysExhaustedError(
        f"All {n_keys} Groq API key(s) are rate-limited. "
        f"Stopping after {MAX_RETRIES} attempts. Last error: {last_error}"
    )
