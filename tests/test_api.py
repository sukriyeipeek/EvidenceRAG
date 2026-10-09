import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
from app.rag.config import Settings
from app.rag.pipeline import RAGPipeline


def fake_embed(texts, model_name):
    return np.array([[1.0, 0.0] for _ in texts], dtype="float32")


def fake_generate(messages, model_name, max_new_tokens):
    return "Etkinlik 2024'te gerçekleşti. [Kanıt 1]"


@pytest.fixture
def client(monkeypatch):
    pipeline = RAGPipeline(
        settings=Settings(relevance_threshold=0.3, max_upload_mb=1),
        embedding_function=fake_embed,
        generation_function=fake_generate,
    )
    monkeypatch.setattr(main, "rag_pipeline", pipeline)
    return TestClient(main.app)


def test_upload_then_ask_returns_answer_with_sources(client):
    upload = client.post(
        "/documents",
        files={"file": ("rapor.txt", "Etkinlik 2024'te gerçekleşti.".encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 200
    assert upload.json()["chunks_indexed"] == 1
    assert client.get("/health").json() == {"status": "ok", "indexed_chunks": 1}

    answer = client.post("/ask", json={"question": "Etkinlik ne zaman oldu?"})
    assert answer.status_code == 200
    body = answer.json()
    assert body["answer"] == "Etkinlik 2024'te gerçekleşti. [Kanıt 1]"
    assert body["sources"][0]["filename"] == "rapor.txt"


def test_upload_rejects_unsupported_file_type(client):
    response = client.post("/documents", files={"file": ("tablo.csv", b"a,b", "text/csv")})
    assert response.status_code == 400


def test_upload_rejects_empty_file(client):
    response = client.post("/documents", files={"file": ("bos.txt", b"", "text/plain")})
    assert response.status_code == 400


def test_upload_rejects_files_over_size_limit(client):
    oversized = b"a" * (1024 * 1024 + 1)
    response = client.post("/documents", files={"file": ("buyuk.txt", oversized, "text/plain")})
    assert response.status_code == 413
    assert client.get("/health").json()["indexed_chunks"] == 0


def test_ask_rejects_empty_question(client):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
