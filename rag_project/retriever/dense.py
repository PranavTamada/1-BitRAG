import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Encode
        embeddings = self.model.encode(documents, convert_to_numpy=True)

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        dim = embeddings.shape[1]

        # Use INNER PRODUCT (not L2)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.embeddings = embeddings

    def search(self, query, k):
        query_vec = self.model.encode([query], convert_to_numpy=True)

        # Normalize query
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        scores, indices = self.index.search(query_vec, k)

        docs = []
        final_scores = []

        for idx, score in zip(indices[0], scores[0]):
            docs.append(self.documents[idx])
            final_scores.append(float(score))  # already similarity

        return docs, final_scores