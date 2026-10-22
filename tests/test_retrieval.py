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
            {"text":"exact threshold evidence","metadata":{"filename":"equal.txt","chunk_id":"equal"}},
            {"text":"below threshold evidence","metadata":{"filename":"low.txt","chunk_id":"low"}},
        ],
        np.array([[1.0, 0.0],[0.0,1.0],[-1.0,0.0]], dtype="float32"),
    )

    results = retrieve(
        "question",
        store,
        relevance_threshold=0.0,
        embedding_function=fake_embed,
    )

    assert results[0]["text"] == "retrievable evidence"
    assert results[0]["filename"] == "a.txt"
    assert results[0]["chunk_id"] == "doc-1-0"
    assert results[0]["score"] >= 0.9
    assert len(results)==2
    assert results[1]["text"]=="exact threshold evidence" and results[1]["score"]==0.0
    assert all(result['chunk_id']!='low' for result in results)


def test_vector_store_does_not_modify_caller_embeddings():
    store = VectorStore()
    embeddings = np.array([[3.0, 4.0]], dtype="float32")

    store.add([{"text": "t", "metadata": {}}], embeddings)
    store.search(embeddings[0])

    assert embeddings.tolist() == [[3.0, 4.0]]


def test_reranker_chooses_top_k_from_a_wider_pool_and_threshold_still_applies():
    store = VectorStore()
    vectors = [[1.0, 0.0], [0.95, 0.31], [0.9, 0.44], [0.0, 1.0]]
    store.add(
        [
            {"text": f"chunk {i}", "metadata": {"document_id": "doc", "chunk_id": f"doc-{i}"}}
            for i in range(len(vectors))
        ],
        np.array(vectors, dtype="float32"),
    )
    seen = {}

    def fake_rerank(query, results, model_name, top_k):
        seen["pool"] = [result["chunk_id"] for result in results]
        # Prefer the chunk the vector search ranked lowest.
        return sorted(results, key=lambda result: result["score"])[:top_k]

    results = retrieve(
        "question",
        store,
        top_k=2,
        relevance_threshold=0.5,
        embedding_function=fake_embed,
        reranker_model_name="fake",
        rerank_candidates=4,
        rerank_function=fake_rerank,
    )

    assert len(seen["pool"]) == 4
    # doc-3 was ranked first by the reranker but is below the cosine threshold.
    assert [result["chunk_id"] for result in results] == ["doc-2"]
