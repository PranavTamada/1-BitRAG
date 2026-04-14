from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class dense_retrieve:
    def __init__(self, documents):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = documents

        self.embeddings = self.model.encode(documents, show_progress_bar=True)
        self.embeddings = np.array(self.embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query, k=5):
        query_vec = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vec, k * 2)  # get extra

        seen = set()
        docs = []
        scores = []

        for idx, dist in zip(indices[0], distances[0]):
            doc = self.documents[idx]
            if doc not in seen:
                seen.add(doc)
                docs.append(doc)
                scores.append(-float(dist))  # convert safely

            if len(docs) == k:
                break

        return docs, scores