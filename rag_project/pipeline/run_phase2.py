import json
from "C:\Users\prana\OneDrive\Desktop\1-bit LLM\rag_project\retriever\retrieve.py" import retrieve
from llm.cheap_llm import run_cheap_llm
from llm.full_llm import run_full_llm
from utils.prompt import build_prompt
from tqdm import tqdm

# Load dataset
with open("data/hotpot_sample.json", "r") as f:
    dataset = json.load(f)


results = []

for sample in tqdm(dataset[:10]):   # only 10 queries
    query = sample["question"]

    # 1. Retrieve docs
    docs, scores = retrieve(query, top_k=3)

    # 2. Build prompt
    prompt = build_prompt(docs, query)

    # 3. Run cheap model
    cheap_answer, cheap_latency = run_cheap_llm(prompt)

    # 4. Run full model
    full_answer, full_latency = run_full_llm(prompt)

    # 5. Log everything
    results.append({
        "query": query,
        "docs": docs,
        "cheap_answer": cheap_answer,
        "full_answer": full_answer,
        "cheap_latency": cheap_latency,
        "full_latency": full_latency
    })


# Save results
with open("logs/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ Phase 2 Complete. Results saved.")