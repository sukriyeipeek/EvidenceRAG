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
