from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DenseRetriever:
    """Dense retriever using SentenceTransformer + FAISS (L2 distance)."""

    # Sentinel value FAISS uses for invalid/padding results
    FAISS_INVALID_SCORE = -3.4e+38

    def __init__(self, documents):
        """
        Args:
            documents: list of strings (the corpus texts).
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = documents

        self.embeddings = self.model.encode(documents, show_progress_bar=True)
        self.embeddings = np.array(self.embeddings).astype("float32")
        
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-9)

        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query, k=5):
        """Return top-k (doc_index, score) pairs, filtering invalid FAISS results.

        Scores are negative L2 distance (higher = more similar).
        """
        query_vec = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vec, k * 2)  # fetch extra for dedup

        results = []  # list of (doc_index, score)
        seen = set()

        for idx, dist in zip(indices[0], distances[0]):
            # Filter out invalid FAISS entries
            if idx < 0 or dist <= self.FAISS_INVALID_SCORE:
                continue

            if idx not in seen:
                seen.add(idx)
                score = -float(dist)  # negate L2 so higher = better
                results.append((int(idx), score))

            if len(results) == k:
                break

        return results

    def score_document(self, query, doc_index):
        query_vec = self.model.encode([query]).astype("float32")
        doc_vec = self.embeddings[doc_index:doc_index + 1]

        # Normalize both vectors (L2 norm)
        query_vec /= np.linalg.norm(query_vec) + 1e-9
        doc_vec_norm = doc_vec / (np.linalg.norm(doc_vec) + 1e-9)

        dist = np.linalg.norm(query_vec - doc_vec_norm)
        return -float(dist)