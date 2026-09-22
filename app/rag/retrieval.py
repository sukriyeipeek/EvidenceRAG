from .embedding import embed_texts


def retrieve(
    query,
    vector_store,
    top_k=5,
    relevance_threshold=0.35,
    embedding_function=embed_texts,
    embedding_model_name="all-MiniLM-L6-v2",
):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if vector_store.is_empty:
        return []

    query_embedding = embedding_function([query], embedding_model_name)[0]
    results = vector_store.search(query_embedding, top_k=top_k)
    return [result for result in results if result["score"] >= relevance_threshold]
