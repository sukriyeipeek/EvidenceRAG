import re


def chunk_documents(pages, chunk_size=800, chunk_overlap=100):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size")

    chunks = []
    for page in pages:
        text = _clean_text(page["text"])
        if not text:
            continue

        for chunk_text in _split_text(text, chunk_size, chunk_overlap):
            metadata = dict(page["metadata"])
            metadata["chunk_id"] = f"{metadata['document_id']}-{len(chunks)}"
            chunks.append({"text": chunk_text, "metadata": metadata})

    return chunks


def _clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def _split_text(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks
