import os
from dataclasses import dataclass


def _int_env(name, default):
    value = os.getenv(name)
    return int(value) if value else default


def _float_env(name, default):
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME") or "intfloat/multilingual-e5-small"
    # E5 models expect these prefixes; set them to an empty string for models that don't.
    embedding_query_prefix: str = os.getenv("EMBEDDING_QUERY_PREFIX", "query: ")
    embedding_passage_prefix: str = os.getenv("EMBEDDING_PASSAGE_PREFIX", "passage: ")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME") or "Qwen/Qwen2.5-1.5B-Instruct"
    chunk_size: int = _int_env("CHUNK_SIZE", 800)
    chunk_overlap: int = _int_env("CHUNK_OVERLAP", 100)
    top_k: int = _int_env("TOP_K", 5)
    relevance_threshold: float = _float_env("RELEVANCE_THRESHOLD", 0.80)
    max_new_tokens: int = _int_env("MAX_NEW_TOKENS", 256)
