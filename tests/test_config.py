from app.rag.config import Settings


def test_settings_read_environment_when_created(monkeypatch):
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.5")
    monkeypatch.setenv("LLM_MODEL_NAME", "some/model")

    settings = Settings()

    assert settings.top_k == 7
    assert settings.relevance_threshold == 0.5
    assert settings.llm_model_name == "some/model"


def test_empty_embedding_prefix_disables_it(monkeypatch):
    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "")
    monkeypatch.delenv("EMBEDDING_PASSAGE_PREFIX", raising=False)

    settings = Settings()

    assert settings.embedding_query_prefix == ""
    assert settings.embedding_passage_prefix == "passage: "
