"""
Cheap LLM — TinyLlama via Ollama
==================================
Local inference at zero API cost. Used as the default answering path
when the routing classifier predicts the query is easy enough for a
small model to handle.

Ported from 1-BitRAG v1.
"""

import time
import ollama

from config import CHEAP_MODEL


def run_cheap_llm(prompt: str) -> tuple[str, float]:
    """Generate a response using the local cheap model.

    Returns:
        (response_text, latency_seconds)
    """
    start = time.time()
    response = ollama.chat(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 4096},
    )
    latency = time.time() - start
    return response["message"]["content"], latency
