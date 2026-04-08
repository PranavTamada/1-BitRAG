def build_prompt(context_docs, question):
    context = "\n\n".join(context_docs)

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt