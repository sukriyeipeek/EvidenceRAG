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


def _bool_env(name, default):
    def read():
        value = os.getenv(name)
        if not value:
            return default
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{name} must be true or false, got {value!r}")

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
    # Combine the vector search with BM25 keyword search (see keyword_index.py).
    hybrid_search: bool = field(default_factory=_bool_env("HYBRID_SEARCH", True))
    # How many chunks each search proposes before the rankings are merged.
    retrieval_candidates: int = field(default_factory=_int_env("RETRIEVAL_CANDIDATES", 30))
    # Weight of the keyword ranking against the vector ranking when they are merged. At
    # equal weight, keyword matches pushed aside the right chunk for English questions
    # about Turkish documents, whose words the Turkish stemmer splits into noise.
    keyword_weight: float = field(default_factory=_float_env("KEYWORD_WEIGHT", 0.5))
    # A cross-encoder that reorders the best candidates (see reranking.py); empty disables it.
    # BAAI/bge-reranker-v2-m3 put every retrievable hard-set answer first but adds about
    # 5 s per question on a CPU, so it is off by default.
    reranker_model_name: str = field(default_factory=_prefix_env("RERANKER_MODEL_NAME", ""))
    rerank_candidates: int = field(default_factory=_int_env("RERANK_CANDIDATES", 20))
    max_new_tokens: int = field(default_factory=_int_env("MAX_NEW_TOKENS", 256))
    max_upload_mb: int = field(default_factory=_int_env("MAX_UPLOAD_MB", 20))
    # Relative paths are resolved against the project root.
    index_dir: str = field(default_factory=_str_env("INDEX_DIR", "data/index"))

    def embedding_signature(self):
        """Settings that determine the stored vectors; changing any of them needs a re-index."""
        return {
            "embedding_model_name": self.embedding_model_name,
            "embedding_query_prefix": self.embedding_query_prefix,
            "embedding_passage_prefix": self.embedding_passage_prefix,
        }
