"""
Sparse Retriever — BM25Okapi
=============================
Lexical retriever using the Okapi BM25 ranking function.

Ported from 1-BitRAG v1 unchanged — BM25 is well-understood and needs no
modifications for the routing research. The key change is that
``retrieve.py`` now surfaces the BM25 ranking order alongside the fused
scores, which the retrieval-feature extractor uses to compute
``bm25_dense_agreement`` (Spearman rank correlation).
"""

from rank_bm25 import BM25Okapi


class SparseRetriever:
    """Sparse retriever using BM25 (Okapi variant)."""

    def __init__(self, documents: list[str]):
        """
        Args:
            documents: list of strings (the corpus texts).
        """
        self.documents = documents
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return top-k ``(doc_index, score)`` pairs ranked by BM25 score."""
        tokenized_query = query.lower().split()
        all_scores = self.bm25.get_scores(tokenized_query)

        indexed_scores = [(i, float(score)) for i, score in enumerate(all_scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:k]

    def score_document(self, query: str, doc_index: int) -> float:
        """Compute the BM25 score for a single document by index."""
        tokenized_query = query.lower().split()
        all_scores = self.bm25.get_scores(tokenized_query)
        return float(all_scores[doc_index])
