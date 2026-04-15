import json
from retriever.dense import DenseRetriever
from retriever.sparse import SparseRetriever
from retriever.retrieve import retrieve
from utils.logger import log_results

# Load dataset
def load_data(path="data/dataset.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def main():
    data = load_data()

    # Extract corpus
    documents = [item["context"] for item in data]
    queries = [item["query"] for item in data]

    # Initialize retrievers once
    dense = DenseRetriever(documents)
    sparse = SparseRetriever(documents)

    all_logs = []

    for query in queries:
        docs, fused_scores = retrieve(query, documents, dense, sparse, top_k=5)

        log_entry = {
            "query": query,
            "fused_scores": fused_scores,
            "docs": docs
        }

        all_logs.append(log_entry)

    log_results(all_logs)

    # Quick test
    docs, scores = retrieve("Who wrote Harry Potter?", documents, dense, sparse, top_k=5)
    print(docs)
    print(scores)

if __name__ == "__main__":
    main()