import numpy as np
import pytest

from app.rag.vector_store import DuplicateDocumentError, IncompatibleStoreError, VectorStore


SIGNATURE = {"embedding_model_name": "model-a"}


def make_document(document_id, sha256, vectors):
    document = {"document_id": document_id, "filename": f"{document_id}.txt", "sha256": sha256}
    chunks = [
        {"text": f"{document_id} chunk {i}", "metadata": {"document_id": document_id, "chunk_id": f"{document_id}-{i}"}}
        for i in range(len(vectors))
    ]
    return document, chunks, np.array(vectors, dtype="float32")


def test_store_persists_documents_and_vectors(tmp_path):
    path = tmp_path / "store.npz"
    store = VectorStore(persist_path=path, signature=SIGNATURE)
    store.add_document(*make_document("doc-1", "hash-1", [[1.0, 0.0], [0.0, 1.0]]))

    reloaded = VectorStore(persist_path=path, signature=SIGNATURE)

    assert [document["document_id"] for document in reloaded.list_documents()] == ["doc-1"]
    assert reloaded.search([0.0, 1.0], top_k=1)[0]["chunk_id"] == "doc-1-1"
    assert not (tmp_path / "store.npz.tmp").exists()


def test_store_rejects_duplicate_content(tmp_path):
    store = VectorStore(persist_path=tmp_path / "store.npz")
    store.add_document(*make_document("doc-1", "same-hash", [[1.0, 0.0]]))

    with pytest.raises(DuplicateDocumentError) as error:
        store.add_document(*make_document("doc-2", "same-hash", [[0.0, 1.0]]))

    assert error.value.document["document_id"] == "doc-1"
    assert len(store.chunks) == 1


def test_delete_removes_only_that_documents_chunks(tmp_path):
    path = tmp_path / "store.npz"
    store = VectorStore(persist_path=path, signature=SIGNATURE)
    store.add_document(*make_document("doc-1", "hash-1", [[1.0, 0.0], [0.9, 0.1]]))
    store.add_document(*make_document("doc-2", "hash-2", [[0.0, 1.0]]))

    assert store.delete_document("doc-1") is True
    assert store.delete_document("doc-1") is False

    for current in (store, VectorStore(persist_path=path, signature=SIGNATURE)):
        assert [document["document_id"] for document in current.list_documents()] == ["doc-2"]
        results = current.search([1.0, 0.0], top_k=5)
        assert [result["chunk_id"] for result in results] == ["doc-2-0"]


def test_deleting_last_document_leaves_an_empty_store(tmp_path):
    path = tmp_path / "store.npz"
    store = VectorStore(persist_path=path)
    store.add_document(*make_document("doc-1", "hash-1", [[1.0, 0.0]]))

    store.delete_document("doc-1")
    reloaded = VectorStore(persist_path=path)

    assert reloaded.is_empty
    assert reloaded.list_documents() == []
    assert reloaded.search([1.0, 0.0]) == []
    # A document can be added again after the store was emptied.
    reloaded.add_document(*make_document("doc-1", "hash-1", [[1.0, 0.0]]))
    assert len(reloaded.chunks) == 1


def test_store_refuses_index_built_with_other_embedding_settings(tmp_path):
    path = tmp_path / "store.npz"
    VectorStore(persist_path=path, signature=SIGNATURE).add_document(
        *make_document("doc-1", "hash-1", [[1.0, 0.0]])
    )

    with pytest.raises(IncompatibleStoreError, match="model-a"):
        VectorStore(persist_path=path, signature={"embedding_model_name": "model-b"})
