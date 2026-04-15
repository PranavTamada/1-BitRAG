from rank_bm25 import BM25Okapi

class SparseRetriever:
    """Sparse retriever using BM25 (Okapi variant)."""

    def __init__(self, documents):
        """
        Args:
            documents: list of strings (the corpus texts).
        """
        self.documents = documents
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query, k=5):
        """Return top-k (doc_index, score) pairs ranked by BM25 score."""
        tokenized_query = query.lower().split()
        all_scores = self.bm25.get_scores(tokenized_query)

        # Pair each document index with its BM25 score and sort descending
        indexed_scores = [(i, float(score)) for i, score in enumerate(all_scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:k]

    def score_document(self, query, doc_index):
        """Compute the BM25 score for a single document by index."""
        tokenized_query = query.lower().split()
        # get_scores scores all docs; we only need one — this is unavoidable
        # with rank_bm25, but we make the limitation explicit
        all_scores = self.bm25.get_scores(tokenized_query)
        return float(all_scores[doc_index])