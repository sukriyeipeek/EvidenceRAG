import numpy as np

from app.rag.retrieval import retrieve
from app.rag.vector_store import VectorStore


def fake_embed(texts, model_name):
    return np.array([[1.0, 0.0] for _ in texts], dtype="float32")


def test_retrieval_returns_metadata_and_filters_low_scores():
    store = VectorStore()
    store.add(
        [
            {
                "text": "retrievable evidence",
                "metadata": {
                    "filename": "a.txt",
                    "document_id": "doc-1",
                    "page": None,
                    "source": "a.txt",
                    "chunk_id": "doc-1-0",
                },
            },
        ],
        np.array([[1.0, 0.0]], dtype="float32"),
    )

    results = retrieve(
        "question",
        store,
        relevance_threshold=0.9,
        embedding_function=fake_embed,
    )

    assert results[0]["text"] == "retrievable evidence"
    assert results[0]["filename"] == "a.txt"
    assert results[0]["chunk_id"] == "doc-1-0"
    assert results[0]["score"] >= 0.9
