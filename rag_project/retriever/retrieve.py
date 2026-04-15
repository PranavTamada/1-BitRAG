from retriever.dense import DenseRetriever
from retriever.sparse import SparseRetriever
from retriever.fusion import fuse_scores

def retrieve(query, documents, dense_retriever, sparse_retriever, top_k=5):
    """Hybrid retrieval: dense + sparse → union → fuse → top-k.

    Args:
        query: the search query string
        documents: list of corpus text strings (used to map indices → text)
        dense_retriever: pre-initialized DenseRetriever instance
        sparse_retriever: pre-initialized SparseRetriever instance
        top_k: number of final results to return

    Returns:
        (docs, scores): lists of top-k document texts and fused scores
    """
    # --- Step 1: Dense retrieval (top candidates) ---
    dense_results = dense_retriever.search(query, k=top_k)

    # --- Step 2: Sparse retrieval (top candidates) ---
    sparse_results = sparse_retriever.search(query, k=top_k)

    # --- Step 3: Union of candidate document indices ---
    candidate_indices = set()
    for idx, _ in dense_results:
        candidate_indices.add(idx)
    for idx, _ in sparse_results:
        candidate_indices.add(idx)

    # --- Step 4: Score every candidate in BOTH systems ---
    dense_scores_dict = {}
    sparse_scores_dict = {}

    for idx in candidate_indices:
        dense_scores_dict[idx] = dense_retriever.score_document(query, idx)
        sparse_scores_dict[idx] = sparse_retriever.score_document(query, idx)

    # --- Steps 5-7: Normalize, fuse, and sort (handled by fuse_scores) ---
    fused_results = fuse_scores(dense_scores_dict, sparse_scores_dict, k=top_k)

    # --- Step 8: Map indices back to document text ---
    docs = [documents[idx] for idx, _ in fused_results]
    scores = [score for _, score in fused_results]

    return docs, scores