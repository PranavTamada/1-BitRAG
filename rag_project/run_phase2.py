import json
from retriever.dense import DenseRetriever
from retriever.sparse import SparseRetriever
from retriever.retrieve import retrieve
from llm.cheap_llm import run_cheap_llm
from llm.full_llm import run_full_llm
from llm.confidence import needs_full_llm, compute_confidence, compute_retrieval_signals
from utils.prompt import build_prompt
from tqdm import tqdm


# -------------------------
# Better evaluation (IMPORTANT)
# -------------------------
def normalize(text):
    return text.strip().lower()

def contains_answer(pred, gt):
    pred = normalize(pred)
    gt = normalize(gt)
    return gt in pred or pred in gt


# -------------------------
# Load dataset
# -------------------------
with open("data/hotpot_clean.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)


# -------------------------
# Build GLOBAL corpus
# -------------------------
documents = []
for sample in dataset:
    documents.extend(sample["docs"])


# -------------------------
# Initialize retrievers
# -------------------------
print("Initializing dense retriever...")
dense_retriever = DenseRetriever(documents)

print("Initializing sparse retriever...")
sparse_retriever = SparseRetriever(documents)


# -------------------------
# Metrics
# -------------------------
results = []
full_llm_calls = 0
total_queries = 0

cheap_score = 0
full_score = 0
adaptive_score = 0

total_cheap_latency = 0.0
total_full_latency = 0.0


# -------------------------
# MAIN LOOP
# -------------------------
for sample in tqdm(dataset[8:16]):   # start small

    query = sample["query"]
    gt = sample["answer"]
    total_queries += 1

    # 1. Retrieval
    docs, scores = retrieve(query, documents, dense_retriever, sparse_retriever, top_k=3)

    # Debug (IMPORTANT)
    print("\nQuery:", query)
    print("Top doc preview:", docs[0][:120])

    # 2. Prompt
    prompt = build_prompt(docs, query)

    # 3. Cheap model
    cheap_answer, cheap_latency = run_cheap_llm(prompt)
    total_cheap_latency += cheap_latency

    # 4. Full baseline (always run once)
    full_only_answer, full_only_latency = run_full_llm(prompt)

    # 5. Confidence + retrieval signals
    confidence = compute_confidence(cheap_answer, query)
    gap, mean, var = compute_retrieval_signals(scores)

    # Routing reason
    reason = []
    if gap < 0.0008:
        reason.append("low_gap")
    if mean < 0.038:
        reason.append("low_mean")
    if confidence < 0.7:
        reason.append("low_confidence")

    # Routing decision
    escalated = needs_full_llm(scores, confidence)

    full_answer = None
    full_latency = 0.0

    if escalated:
        full_answer = full_only_answer
        full_latency = full_only_latency
        full_llm_calls += 1
        total_full_latency += full_latency

    # Final answer
    final_answer = full_answer if escalated else cheap_answer

    # -------------------------
    # Evaluation
    # -------------------------
    cheap_correct = contains_answer(cheap_answer, gt)
    full_correct = contains_answer(full_only_answer, gt)
    adaptive_correct = contains_answer(final_answer, gt)

    cheap_score += cheap_correct
    full_score += full_correct
    adaptive_score += adaptive_correct

    # -------------------------
    # Logging
    # -------------------------
    results.append({
        "query": query,
        "ground_truth": gt,
        "docs": docs,
        "scores": scores,
        "retrieval_gap": gap,
        "retrieval_mean": mean,
        "retrieval_variance": var,
        "cheap_answer": cheap_answer,
        "cheap_confidence": confidence,
        "routing_reason": reason,
        "escalated_to_full": escalated,
        "full_answer": full_answer,
        "final_answer": final_answer,
        "cheap_correct": cheap_correct,
        "full_correct": full_correct,
        "adaptive_correct": adaptive_correct,
        "cheap_latency": cheap_latency,
        "full_latency": full_latency
    })


# -------------------------
# Save results
# -------------------------
with open("logs/results.json", "w") as f:
    json.dump(results, f, indent=4)


# -------------------------
# Final metrics
# -------------------------
skipped = total_queries - full_llm_calls

print("\n=== FINAL RESULTS ===")

print(f"Cheap Accuracy     : {cheap_score/total_queries:.2f}")
print(f"Full Accuracy      : {full_score/total_queries:.2f}")
print(f"Adaptive Accuracy  : {adaptive_score/total_queries:.2f}")

print(f"\nFull LLM Calls: {full_llm_calls}/{total_queries} "
      f"({skipped} saved)")

print(f"Avg Cheap Latency: {total_cheap_latency/total_queries:.2f}s")
if full_llm_calls > 0:
    print(f"Avg Full Latency : {total_full_latency/full_llm_calls:.2f}s")