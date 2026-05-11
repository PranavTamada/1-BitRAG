"""
Prompt Builders
================
Prompt templates for the cheap and full LLM pipelines.

Research note:
    The summary prompt is used when retrieved context exists (cheap LLM path).
    The direct prompt is used as a fallback when retrieval fails or the query
    is escalated to the full LLM without context.
"""


def build_summary_prompt(retrieved_pairs: list[tuple[str, str]], question: str) -> str:
    """Build a prompt that asks the LLM to synthesise retrieved Q&A matches.

    Args:
        retrieved_pairs: list of (prompt_text, completion_text) from retrieval.
        question: the user's original query.

    Returns:
        Formatted prompt string.
    """
    context = "\n\n".join(
        f"Prompt: {prompt}\nCompletion: {completion}"
        for prompt, completion in retrieved_pairs
    )

    return f"""You are given healthcare Q&A matches from a local dataset.
Synthesize a clear and concise answer to the user's question using these matches.
If multiple sub-questions are present, cover each one explicitly.

Retrieved Matches:
{context}

User Question:
{question}

Final Answer:
"""


def build_direct_prompt(question: str) -> str:
    """Build a prompt for direct answering without retrieved context.

    Used when:
        - Retrieval fails to find relevant documents
        - The query is escalated directly to the full LLM

    Args:
        question: the user's query.

    Returns:
        Formatted prompt string.
    """
    return f"""Answer the following healthcare question directly and clearly.

Question:
{question}

Answer:
"""


def build_context_prompt(context_docs: list[str], question: str) -> str:
    """Build a context-grounded prompt from raw document texts.

    Args:
        context_docs: list of retrieved document strings.
        question: the user's query.

    Returns:
        Formatted prompt string.
    """
    context = "\n\n".join(context_docs)
    return f"""Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""
