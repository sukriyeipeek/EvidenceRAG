import threading

import numpy as np
import faiss


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        # Uploads and questions are served from different threads; the FAISS index and
        # the chunk list must change together, so every read and write holds this lock.
        self._lock = threading.Lock()

    @property
    def is_empty(self):
        return not self.chunks

    def add(self, chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have one embedding")
        if not chunks:
            return

        # Copy so normalizing in place doesn't modify the caller's array.
        embeddings = np.array(embeddings, dtype="float32")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional array")
        faiss.normalize_L2(embeddings)

        with self._lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(embeddings.shape[1])
            elif self.index.d != embeddings.shape[1]:
                raise ValueError("Embedding dimension does not match the vector store")

            self.index.add(embeddings)
            self.chunks.extend(chunks)

    def search(self, query_embedding, top_k=5):
        if self.is_empty or top_k <= 0:
            return []

        query_embedding = np.array(query_embedding, dtype="float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        with self._lock:
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
            matched = [
                (float(score), self.chunks[index])
                for score, index in zip(scores[0], indices[0])
                if index >= 0
            ]

        results = []
        for score, chunk in matched:
            metadata = dict(chunk["metadata"])
            results.append(
                {
                    "text": chunk["text"],
                    "score": score,
                    "metadata": metadata,
                    **metadata,
                }
            )
        return results
