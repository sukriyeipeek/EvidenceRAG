import os
from dataclasses import dataclass, field


def _str_env(name, default):
    return lambda: os.getenv(name) or default


def _prefix_env(name, default):
    # Unlike other settings, an empty value is meaningful here: it disables the prefix.
    return lambda: os.getenv(name, default)


def _int_env(name, default):
    def read():
        value = os.getenv(name)
        return int(value) if value else default

    return read


def _float_env(name, default):
    def read():
        value = os.getenv(name)
        return float(value) if value else default

    return read


# Environment variables are read when Settings() is created, not when this module is
# imported, so values loaded from a .env file at startup are picked up.
@dataclass(frozen=True)
class Settings:
    embedding_model_name: str = field(
        default_factory=_str_env("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
    )
    # E5 models expect these prefixes; set them to an empty string for models that don't.
    embedding_query_prefix: str = field(default_factory=_prefix_env("EMBEDDING_QUERY_PREFIX", "query: "))
    embedding_passage_prefix: str = field(
        default_factory=_prefix_env("EMBEDDING_PASSAGE_PREFIX", "passage: ")
    )
    llm_model_name: str = field(default_factory=_str_env("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"))
    chunk_size: int = field(default_factory=_int_env("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=_int_env("CHUNK_OVERLAP", 100))
    top_k: int = field(default_factory=_int_env("TOP_K", 5))
    relevance_threshold: float = field(default_factory=_float_env("RELEVANCE_THRESHOLD", 0.80))
    max_new_tokens: int = field(default_factory=_int_env("MAX_NEW_TOKENS", 256))
    max_upload_mb: int = field(default_factory=_int_env("MAX_UPLOAD_MB", 20))
