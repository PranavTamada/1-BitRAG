import json
from retriever.dense import DenseRetriever
from retriever.sparse import SparseRetriever
from retriever.retrieve import retrieve
from llm.cheap_llm import run_cheap_llm
from llm.full_llm import run_full_llm
from llm.confidence import needs_full_llm, compute_confidence
from utils.prompt import build_prompt
from tqdm import tqdm

# Load dataset
with open("data/hotpot_sample.json", "r") as f:
    dataset = json.load(f)

# Extract corpus texts once
documents = [item["context"] for item in dataset]

# Initialize retrievers once (expensive — encodes all documents)
print("Initializing dense retriever...")
dense_retriever = DenseRetriever(documents)
print("Initializing sparse retriever...")
sparse_retriever = SparseRetriever(documents)

results = []
full_llm_calls = 0
total_queries = 0

for sample in tqdm(dataset[5:8]):   # 10 queries
    query = sample["query"]
    total_queries += 1

    # 1. Retrieve docs (reuses pre-built indices)
    docs, scores = retrieve(query, documents, dense_retriever, sparse_retriever, top_k=3)

    # 2. Build prompt
    prompt = build_prompt(docs, query)

    # 3. Run cheap model first
    cheap_answer, cheap_latency = run_cheap_llm(prompt)

    # 4. Confidence gate — only call full model when cheap answer is uncertain
    confidence = compute_confidence(cheap_answer, query)
    escalated = needs_full_llm(scores, confidence)

    full_answer = None
    full_latency = 0.0

    if escalated:
        full_answer, full_latency = run_full_llm(prompt)
        full_llm_calls += 1

    # The final answer is the full model's if escalated, otherwise the cheap model's
    final_answer = full_answer if escalated else cheap_answer

    # 5. Log everything
    results.append({
        "query": query,
        "docs": docs,
        "scores": scores,
        "cheap_answer": cheap_answer,
        "cheap_confidence": confidence,
        "escalated_to_full": escalated,
        "full_answer": full_answer,
        "final_answer": final_answer,
        "cheap_latency": cheap_latency,
        "full_latency": full_latency
    })

# Save results
with open("logs/results.json", "w") as f:
    json.dump(results, f, indent=4)

skipped = total_queries - full_llm_calls
print(f"\n Phase 2 Complete. Results saved.")
print(f"Full LLM calls: {full_llm_calls}/{total_queries} "
      f"({skipped} skipped, {skipped/total_queries*100:.0f}% reduction)")