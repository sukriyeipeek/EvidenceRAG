from app.rag.chunking import chunk_documents


def test_chunking_preserves_metadata_and_overlap():
    pages = [
        {
            "text": "one two three four five six seven eight nine ten",
            "metadata": {
                "filename": "notes.txt",
                "document_id": "doc-1",
                "page": None,
                "source": "notes.txt",
            },
        }
    ]

    chunks = chunk_documents(pages, chunk_size=20, chunk_overlap=5)

    assert len(chunks) > 1
    assert chunks[0]["metadata"]["filename"] == "notes.txt"
    assert chunks[0]["metadata"]["chunk_id"] == "doc-1-0"
    assert chunks[0]["text"].split()[-1] in chunks[1]["text"].split()


def _page(text):
    return {
        "text": text,
        "metadata": {"filename": "rapor.md", "document_id": "doc-1", "page": None, "source": "rapor.md"},
    }


def test_chunks_carry_their_heading_path():
    text = "# Yeşil Vadi Raporu\n\n## Su\nKayıp oranı düştü.\n\n## Bütçe\n### Gelir\nGelir arttı."

    chunks = chunk_documents([_page(text)])

    assert [chunk["text"] for chunk in chunks] == [
        "Yeşil Vadi Raporu > Su: Kayıp oranı düştü.",
        "Yeşil Vadi Raporu > Bütçe > Gelir: Gelir arttı.",
    ]
    assert chunks[1]["metadata"]["section"] == "Yeşil Vadi Raporu > Bütçe > Gelir"


def test_long_sections_repeat_the_heading_in_every_chunk():
    text = "# Rapor\n## Su\n" + " ".join(f"kelime{i}" for i in range(60))

    chunks = chunk_documents([_page(text)], chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk["text"].startswith("Rapor > Su: ") for chunk in chunks)
    assert all(len(chunk["text"]) <= 120 for chunk in chunks)


def test_overlap_starts_at_a_word_boundary():
    words = [f"kelime{i}" for i in range(30)]

    chunks = chunk_documents([_page(" ".join(words))], chunk_size=60, chunk_overlap=15)

    for chunk in chunks:
        assert all(word in words for word in chunk["text"].split())
