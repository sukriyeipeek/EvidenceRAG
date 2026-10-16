import re


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def chunk_documents(pages, chunk_size=800, chunk_overlap=100):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size")

    chunks = []
    for page in pages:
        for headings, section_text in _split_sections(page["text"]):
            text = _clean_text(section_text)
            if not text:
                continue

            # Every chunk carries its heading path, so a chunk from the middle of a
            # section still says which document and section it belongs to.
            prefix = " > ".join(headings)
            body_size = chunk_size - len(prefix) - 2 if prefix else chunk_size
            body_size = max(body_size, chunk_size // 2)
            overlap = min(chunk_overlap, body_size - 1)

            for chunk_text in _split_text(text, body_size, overlap):
                metadata = dict(page["metadata"])
                metadata["chunk_id"] = f"{metadata['document_id']}-{len(chunks)}"
                if headings:
                    metadata["section"] = prefix
                    chunk_text = f"{prefix}: {chunk_text}"
                chunks.append({"text": chunk_text, "metadata": metadata})

    return chunks


def _split_sections(text):
    """Split Markdown-style text at headings into (heading path, body) pairs.

    Text without headings comes back as a single section with an empty path.
    """
    sections = []
    headings = []
    lines = []
    for line in text.splitlines():
        match = HEADING_PATTERN.match(line.strip())
        if not match:
            lines.append(line)
            continue

        sections.append((list(headings), "\n".join(lines)))
        lines = []
        level = len(match.group(1))
        headings = headings[: level - 1] + [match.group(2)]

    sections.append((list(headings), "\n".join(lines)))
    return sections


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
        # Start the overlap at a word boundary instead of in the middle of a word.
        if text[start - 1] != " ":
            next_space = text.find(" ", start, end)
            if next_space != -1:
                start = next_space + 1

    return chunks
