import numpy as np

from app.rag.keyword_index import KeywordIndex, tokenize
from app.rag.vector_store import VectorStore


def test_tokenize_lowercases_turkish_letters_and_stems_suffixes():
    assert tokenize("FATURALARIMIZIN İptali") == tokenize("faturalarımızın iptali")
    assert tokenize("faturalarımızın") == tokenize("fatura") == ["fatur"]
    assert "1043" in tokenize("E-1043 hatası")


def test_rare_words_outweigh_common_ones():
    index = KeywordIndex(
        [
            "fatura fatura iptal süresi yedi gün",
            "fatura düzenleme kuralları",
            "fatura arşiv saklama",
        ]
    )

    assert index.search("fatura iptal")[0][0] == 0
    assert index.search("bilinmeyen kelime") == []


def test_hybrid_search_finds_exact_codes_the_vectors_miss():
    store = VectorStore()
    chunks = [
        {"text": text, "metadata": {"document_id": "doc", "chunk_id": f"doc-{i}"}}
        for i, text in enumerate(["vergi numarası geçersiz", "E-1043 posta kutusu dolu"])
    ]
    # The vectors favour the first chunk; only the keyword search knows the code.
    store.add_document(
        {"document_id": "doc", "filename": "kodlar.md", "sha256": "h"},
        chunks,
        np.array([[1.0, 0.0], [0.6, 0.8]], dtype="float32"),
    )

    vector_only = store.search([1.0, 0.0], top_k=2)
    hybrid = store.hybrid_search("E-1043 ne demek", [1.0, 0.0], top_k=2, keyword_weight=1.0)

    assert vector_only[0]["chunk_id"] == "doc-0"
    assert hybrid[0]["chunk_id"] == "doc-1"
    # The cosine score is kept for the relevance threshold.
    assert hybrid[0]["score"] == np.float32(0.6)
    assert hybrid[0]["fused_score"] > hybrid[1]["fused_score"]


def test_keyword_index_follows_deletions():
    store = VectorStore()
    for name in ("a", "b"):
        store.add_document(
            {"document_id": name, "filename": f"{name}.md", "sha256": name},
            [{"text": {"a": "alfa kelime", "b": "beta kelime"}[name], "metadata": {"document_id": name, "chunk_id": f"{name}-0"}}],
            np.array([[1.0, 0.0]], dtype="float32"),
        )
    assert store.hybrid_search("beta", [1.0, 0.0], top_k=1, keyword_weight=5)[0]["chunk_id"] == "b-0"

    store.delete_document("b")

    assert [r["chunk_id"] for r in store.hybrid_search("beta", [1.0, 0.0], top_k=5)] == ["a-0"]
