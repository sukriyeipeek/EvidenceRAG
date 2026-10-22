from .runtime import SingleFlightCache

_cache = SingleFlightCache(maxsize=2)


def get_reranker(model_name, revision=None):
    # Imported lazily for the same reason as in embedding.py: torch is slow to load.
    from sentence_transformers import CrossEncoder

    import re
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Reranker requires an immutable revision")
    return _cache.get((model_name, revision), lambda: CrossEncoder(model_name, revision=revision))


def rerank(query, results, model_name, top_k=5, *, revision=None):
    """Reorder search results by a cross-encoder's judgement of each (question, chunk) pair.

    The embedding model encodes the question and each chunk separately and compares the
    vectors; a cross-encoder reads both together, so it can tell "iptal edilebilir" from
    "iptal edilemez". It is too slow to run over the whole index, so it only reorders
    the candidates the searches already found.
    """
    if not results:
        return results
    scores = get_reranker(model_name, revision).predict([(query, result["text"]) for result in results])
    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)
    return sorted(results, key=lambda result: result["rerank_score"], reverse=True)[:top_k]
