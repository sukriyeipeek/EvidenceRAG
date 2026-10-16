from .embedding import embed_texts
from .reranking import rerank


def retrieve(
    query,
    vector_store,
    top_k=5,
    relevance_threshold=0.80,
    embedding_function=embed_texts,
    embedding_model_name="intfloat/multilingual-e5-small",
    query_prefix="",
    hybrid=False,
    candidates=30,
    keyword_weight=1.0,
    reranker_model_name="",
    rerank_candidates=20,
    rerank_function=rerank,
):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if vector_store.is_empty:
        return []

    # With a reranker, the searches gather a wider pool for it to choose the top_k from.
    pool = max(rerank_candidates, top_k) if reranker_model_name else top_k
    query_embedding = embedding_function([query_prefix + query], embedding_model_name)[0]
    if hybrid:
        results = vector_store.hybrid_search(
            query, query_embedding, top_k=pool, candidates=max(candidates, pool), keyword_weight=keyword_weight
        )
    else:
        results = vector_store.search(query_embedding, top_k=pool)
    if reranker_model_name:
        results = rerank_function(query, results, reranker_model_name, top_k=top_k)
    # The threshold is on the cosine score whichever search ranked the results, so a
    # keyword match alone can't make an unrelated chunk count as evidence.
    return [result for result in results if result["score"] >= relevance_threshold]
