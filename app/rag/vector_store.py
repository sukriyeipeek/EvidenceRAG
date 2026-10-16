import json
import os
import threading
from pathlib import Path

import numpy as np
import faiss

from .keyword_index import KeywordIndex


STORE_FORMAT_VERSION = 1
# The usual reciprocal rank fusion constant; it keeps the first few ranks from
# dominating the merged order.
RRF_K = 60


class DuplicateDocumentError(Exception):
    """Raised when a document with the same content is already indexed."""

    def __init__(self, document):
        super().__init__(
            f"This document is already indexed as '{document['filename']}'"
        )
        self.document = document


class IncompatibleStoreError(RuntimeError):
    """Raised when a saved store was built with different embedding settings."""


class VectorStore:
    def __init__(self, persist_path=None, signature=None):
        self.index = None
        self.chunks = []
        self.documents = {}
        # Identifies how the stored embeddings were produced, so a saved store isn't
        # silently searched with vectors from a different embedding model.
        self.signature = signature or {}
        self.persist_path = Path(persist_path) if persist_path else None
        # Uploads, deletions and questions are served from different threads; the FAISS
        # index, the chunk list and the document registry must change together, so every
        # read and write holds this lock.
        self._lock = threading.Lock()
        self._keyword_index = None

        if self.persist_path and self.persist_path.exists():
            self._load()

    @property
    def is_empty(self):
        return not self.chunks

    def add(self, chunks, embeddings):
        embeddings = self._prepare_embeddings(chunks, embeddings)
        if embeddings is None:
            return
        with self._lock:
            self._append(chunks, embeddings)
            self._save()

    def add_document(self, document, chunks, embeddings):
        embeddings = self._prepare_embeddings(chunks, embeddings)
        with self._lock:
            existing = self._find_by_hash(document.get("sha256"))
            if existing:
                raise DuplicateDocumentError(existing)
            if embeddings is not None:
                self._append(chunks, embeddings)
            self.documents[document["document_id"]] = dict(document)
            self._save()

    def find_document_by_hash(self, sha256):
        with self._lock:
            existing = self._find_by_hash(sha256)
            return dict(existing) if existing else None

    def list_documents(self):
        with self._lock:
            documents = [dict(document) for document in self.documents.values()]
        return sorted(documents, key=lambda document: document.get("uploaded_at") or "")

    def delete_document(self, document_id):
        with self._lock:
            if document_id not in self.documents:
                return False

            keep = [
                position
                for position, chunk in enumerate(self.chunks)
                if chunk["metadata"].get("document_id") != document_id
            ]
            # IndexFlat has no stable ids, so rebuild it from the remaining vectors.
            embeddings = self._all_embeddings()[keep]
            self.chunks = [self.chunks[position] for position in keep]
            self._keyword_index = None
            self.index = None
            if len(keep):
                self.index = faiss.IndexFlatIP(embeddings.shape[1])
                self.index.add(embeddings)

            del self.documents[document_id]
            self._save()
            return True

    def search(self, query_embedding, top_k=5):
        if self.is_empty or top_k <= 0:
            return []

        query_embedding = np.array(query_embedding, dtype="float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        with self._lock:
            if self.index is None:
                return []
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
            matched = [
                (float(score), self.chunks[index])
                for score, index in zip(scores[0], indices[0])
                if index >= 0
            ]

        return [_result(chunk, score) for score, chunk in matched]

    def hybrid_search(self, query_text, query_embedding, top_k=5, candidates=30, keyword_weight=1.0):
        """Merge vector and keyword (BM25) rankings with reciprocal rank fusion.

        Each search proposes its best `candidates` chunks; a chunk scores
        1 / (RRF_K + rank) in every list it appears in, so chunks both searches rank
        well come first. keyword_weight scales the BM25 list's votes against the
        vector list's. "score" stays the cosine similarity, which the relevance
        threshold is calibrated on, and "fused_score" holds the RRF score.
        """
        if self.is_empty or top_k <= 0:
            return []

        query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        with self._lock:
            if self.index is None:
                return []
            pool = min(candidates, len(self.chunks))
            _, indices = self.index.search(query_embedding, pool)
            vector_ranking = [int(index) for index in indices[0] if index >= 0]
            keyword_ranking = [position for position, _ in self._keywords().search(query_text, pool)]

            fused = {}
            for ranking, weight in ((vector_ranking, 1.0), (keyword_ranking, keyword_weight)):
                for rank, position in enumerate(ranking, start=1):
                    fused[position] = fused.get(position, 0.0) + weight / (RRF_K + rank)
            best = sorted(fused, key=fused.get, reverse=True)[:top_k]
            matched = [
                (
                    float(self.index.reconstruct(position) @ query_embedding[0]),
                    fused[position],
                    self.chunks[position],
                )
                for position in best
            ]

        return [_result(chunk, score, fused_score=fused_score) for score, fused_score, chunk in matched]

    def _keywords(self):
        # Built on first use after the chunks change; callers hold the lock.
        if self._keyword_index is None:
            self._keyword_index = KeywordIndex([chunk["text"] for chunk in self.chunks])
        return self._keyword_index

    @staticmethod
    def _prepare_embeddings(chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have one embedding")
        if not chunks:
            return None

        # Copy so normalizing in place doesn't modify the caller's array.
        embeddings = np.array(embeddings, dtype="float32")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional array")
        faiss.normalize_L2(embeddings)
        return embeddings

    def _append(self, chunks, embeddings):
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        elif self.index.d != embeddings.shape[1]:
            raise ValueError("Embedding dimension does not match the vector store")

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self._keyword_index = None

    def _find_by_hash(self, sha256):
        if not sha256:
            return None
        return next(
            (document for document in self.documents.values() if document.get("sha256") == sha256),
            None,
        )

    def _all_embeddings(self):
        if self.index is None or self.index.ntotal == 0:
            return np.empty((0, 0), dtype="float32")
        return self.index.reconstruct_n(0, self.index.ntotal)

    def _save(self):
        if not self.persist_path:
            return

        meta = {
            "version": STORE_FORMAT_VERSION,
            "signature": self.signature,
            "documents": list(self.documents.values()),
            "chunks": self.chunks,
        }
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.persist_path.with_name(self.persist_path.name + ".tmp")
        # Everything goes into one file that replaces the old one in a single step, so a
        # crash mid-write leaves the previous store intact instead of a mismatched pair.
        with open(temporary_path, "wb") as file:
            np.savez(file, embeddings=self._all_embeddings(), meta=np.array(json.dumps(meta)))
        os.replace(temporary_path, self.persist_path)

    def _load(self):
        with np.load(self.persist_path, allow_pickle=False) as data:
            embeddings = data["embeddings"]
            meta = json.loads(str(data["meta"]))

        if meta.get("version") != STORE_FORMAT_VERSION:
            raise IncompatibleStoreError(
                f"Unsupported store format version {meta.get('version')} in {self.persist_path}"
            )
        if self.signature and meta.get("signature") != self.signature:
            raise IncompatibleStoreError(
                f"The saved index at {self.persist_path} was built with embedding settings "
                f"{meta.get('signature')}, but the current settings are {self.signature}. "
                "Restore the previous settings, or delete the index file to start over and "
                "upload the documents again."
            )
        if len(meta["chunks"]) != len(embeddings):
            raise IncompatibleStoreError(
                f"The saved index at {self.persist_path} is inconsistent: "
                f"{len(meta['chunks'])} chunks but {len(embeddings)} embeddings."
            )

        self.chunks = meta["chunks"]
        self.documents = {document["document_id"]: document for document in meta["documents"]}
        if len(embeddings):
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(np.ascontiguousarray(embeddings, dtype="float32"))


def _result(chunk, score, **extra):
    metadata = dict(chunk["metadata"])
    return {"text": chunk["text"], "score": score, **extra, "metadata": metadata, **metadata}
