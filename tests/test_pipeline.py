from pathlib import Path

import numpy as np

from app.rag.config import Settings
from app.rag.pipeline import INSUFFICIENT_EVIDENCE, RAGPipeline


def fake_embed(texts, model_name):
    return np.array([[1.0, 0.0] for _ in texts], dtype="float32")


def fake_generate(prompt, model_name, max_new_tokens):
    assert "Only answer using the supplied evidence" in prompt
    return "The evidence says the event happened in 2024. [doc-1-0]"


def test_pipeline_indexes_documents_and_returns_sources(tmp_path: Path):
    document = tmp_path / "report.txt"
    document.write_text("The event happened in 2024.", encoding="utf-8")
    settings = Settings(chunk_size=100, chunk_overlap=10, relevance_threshold=0.3)
    pipeline = RAGPipeline(
        settings=settings,
        embedding_function=fake_embed,
        generation_function=fake_generate,
    )

    result = pipeline.index_file(document)
    answer = pipeline.ask("When did the event happen?")

    assert result["chunks_indexed"] == 1
    assert answer["answer"].startswith("The evidence says")
    assert answer["sources"][0]["filename"] == "report.txt"
    assert answer["sources"][0]["text"] == "The event happened in 2024."
    assert answer["sources"][0]["evidence_number"] == 1


def test_pipeline_reports_insufficient_evidence(tmp_path: Path):
    document = tmp_path / "report.txt"
    document.write_text("The event happened in 2024.", encoding="utf-8")
    settings = Settings(relevance_threshold=1.1)
    pipeline = RAGPipeline(settings=settings, embedding_function=fake_embed)
    pipeline.index_file(document)

    result = pipeline.ask("What is unrelated?")

    assert result == {"answer": INSUFFICIENT_EVIDENCE, "sources": []}
