from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def get_embedding_model(model_name="intfloat/multilingual-e5-small"):
    return SentenceTransformer(model_name)


def embed_texts(texts, model_name="intfloat/multilingual-e5-small"):
    if not texts:
        return np.empty((0, 0), dtype="float32")

    model = get_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")
