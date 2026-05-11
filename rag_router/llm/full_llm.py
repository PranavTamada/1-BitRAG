"""
Full LLM — Llama-3.3-70B via Groq API
=======================================
High-capability model used for difficult queries that the cheap model
cannot handle.  API calls are expensive, so the routing system's goal
is to minimise calls to this model while preserving accuracy.

Includes exponential backoff retry logic to handle Groq rate limits
during large-scale label collection (1000+ samples).

Ported from 1-BitRAG v1.
"""

import time
import os
from groq import Groq

from config import FULL_MODEL, GROQ_API_KEY

_client = None

MAX_RETRIES = 5
BASE_WAIT = 10  # seconds


def run_full_llm(prompt: str) -> tuple[str, float]:
    """Generate a response using the full-capability model via Groq.

    Includes exponential backoff for rate limit errors (429).
    Max retries: 5, starting at 10s wait, doubling each retry.

    Returns:
        (response_text, latency_seconds)
    """
    global _client
    start = time.time()

    if _client is None:
        api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        _client = Groq(api_key=api_key)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = _client.chat.completions.create(
                model=FULL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            latency = time.time() - start
            return response.choices[0].message.content, latency
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Retry on rate limits or transient server errors
            if "rate" in error_str or "429" in error_str or "503" in error_str or "500" in error_str:
                wait = BASE_WAIT * (2 ** attempt)
                print(f"    [RATE LIMIT] Attempt {attempt+1}/{MAX_RETRIES}, "
                      f"waiting {wait}s... ({e})")
                time.sleep(wait)
            else:
                raise  # Non-retryable error

    raise RuntimeError(
        f"Groq API failed after {MAX_RETRIES} retries: {last_error}"
    )
