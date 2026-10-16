from functools import lru_cache


@lru_cache(maxsize=2)
def get_reranker(model_name):
    # Imported lazily for the same reason as in embedding.py: torch is slow to load.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(query, results, model_name, top_k=5):
    """Reorder search results by a cross-encoder's judgement of each (question, chunk) pair.

    The embedding model encodes the question and each chunk separately and compares the
    vectors; a cross-encoder reads both together, so it can tell "iptal edilebilir" from
    "iptal edilemez". It is too slow to run over the whole index, so it only reorders
    the candidates the searches already found.
    """
    if not results:
        return results
    scores = get_reranker(model_name).predict([(query, result["text"]) for result in results])
    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)
    return sorted(results, key=lambda result: result["rerank_score"], reverse=True)[:top_k]
