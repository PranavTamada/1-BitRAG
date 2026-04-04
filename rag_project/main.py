import json
from retriever.dense import DenseRetriever
from retriever.sparse import SparseRetriever
from retriever.fusion import fuse_results
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

    # Initialize retrievers
    dense = DenseRetriever(documents)
    sparse = SparseRetriever(documents)

    all_logs = []

    for query in queries:
        docs_dense, scores_dense = dense.search(query)
        docs_sparse, scores_sparse = sparse.search(query)

        fused_docs, fused_scores = fuse_results(
            docs_dense, scores_dense,
            docs_sparse, scores_sparse
        )

        log_entry = {
            "query": query,
            "dense_scores": scores_dense,
            "sparse_scores": scores_sparse,
            "fused_scores": fused_scores,
            "docs": fused_docs
        }

        all_logs.append(log_entry)

    log_results(all_logs)
    docs, scores = dense.search("Who wrote Harry Potter?")
    print(docs)
    print(scores)

if __name__ == "__main__":
    main()