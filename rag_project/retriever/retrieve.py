from retriever.dense import dense_retrieve
from retriever.sparse import sparse_retrieve
from retriever.fusion import fuse_scores


def retrieve(query, top_k=3):
    # 1. Dense retrieval
    dense_docs, dense_scores = dense_retrieve(query, top_k)

    # 2. Sparse retrieval
    sparse_docs, sparse_scores = sparse_retrieve(query, top_k)

    # 3. Combine docs (assuming same ordering or already aligned)
    docs = dense_docs  # or merge if different

    # 4. Fuse scores
    fused_scores = fuse_scores(dense_scores, sparse_scores)

    # 5. Sort by fused score
    sorted_indices = sorted(range(len(fused_scores)), key=lambda i: fused_scores[i], reverse=True)

    top_docs = [docs[i] for i in sorted_indices[:top_k]]
    top_scores = [fused_scores[i] for i in sorted_indices[:top_k]]

    return top_docs, top_scores