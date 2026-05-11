"""
Dense Retriever — SentenceTransformer + FAISS
=============================================
Encodes documents with ``all-MiniLM-L6-v2`` and indexes them in a FAISS
Inner-Product index for fast cosine-similarity search.

Ported from 1-BitRAG v1 with the following enhancements for RAG-Router:
  * Exposes raw embeddings via get_embeddings() for context-density feature.
  * Returns per-document ranking metadata needed by the feature extractor.
  * Offline-safe Jaccard fallback when the embedding model is unavailable.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class DenseRetriever:
    """Semantic vector retriever backed by FAISS Inner Product."""

    def __init__(self, documents: list[str]):
        self.documents = documents
        self.model = None
        self.index = None
        self.embeddings = None
        self.use_fallback = False

        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            embeddings = self.model.encode(documents, convert_to_numpy=True)
            # L2-normalise so Inner Product ≡ cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms
            self.embeddings = embeddings

            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
        except Exception:
            # Offline-safe lexical fallback
            self.use_fallback = True
            self.doc_tokens = [set(doc.lower().split()) for doc in documents]

    # ── Search ────────────────────────────────────────────────────────────
    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return top-k ``(doc_index, score)`` pairs, descending by score."""
        safe_k = min(k, len(self.documents))
        if safe_k <= 0:
            return []

        if self.use_fallback:
            return self._fallback_search(query, safe_k)

        query_vec = self._encode_query(query)
        scores, indices = self.index.search(query_vec, safe_k)
        return [
            (int(idx), float(sc))
            for idx, sc in zip(indices[0], scores[0])
            if idx >= 0
        ]

    # ── Score a single document ───────────────────────────────────────────
    def score_document(self, query: str, doc_index: int) -> float:
        """Cosine similarity between *query* and document at *doc_index*."""
        if self.use_fallback:
            return self._fallback_score(query, doc_index)
        query_vec = self._encode_query(query)
        return float(np.dot(query_vec[0], self.embeddings[doc_index]))

    # ── Embedding access (needed for context-density feature) ─────────────
    def get_embeddings(self, indices: list[int]) -> np.ndarray:
        """Return the embedding matrix for the given document indices.

        Research purpose: the context-density retrieval feature (feature #10)
        measures mean pairwise cosine similarity of retrieved doc embeddings
        to quantify whether the retrieved context is coherent or scattered.
        """
        if self.embeddings is None:
            return np.zeros((len(indices), 1))
        return self.embeddings[np.array(indices)]

    # ── Internal helpers ──────────────────────────────────────────────────
    def _encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], convert_to_numpy=True)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        return vec / norm

    def _fallback_search(self, query: str, k: int) -> list[tuple[int, float]]:
        q_tokens = set(query.lower().split())
        ranked = []
        for idx, tokens in enumerate(self.doc_tokens):
            union = q_tokens | tokens
            score = len(q_tokens & tokens) / len(union) if union else 0.0
            ranked.append((idx, float(score)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def _fallback_score(self, query: str, doc_index: int) -> float:
        q_tokens = set(query.lower().split())
        d_tokens = self.doc_tokens[doc_index]
        union = q_tokens | d_tokens
        return float(len(q_tokens & d_tokens) / len(union)) if union else 0.0
