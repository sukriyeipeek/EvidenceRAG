import re
import hashlib

from .config import InputLimitError


def chunk_documents(pages, chunk_size=800, chunk_overlap=100, max_chunks=2000, *, strategy="legacy_character", ingestion_version=1):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size")
    if strategy not in {"legacy_character", "structured_v2", "markdown_v1"}:
        raise ValueError("Unsupported chunker strategy")
    if strategy=="structured_v2" and ingestion_version!=2:
        raise ValueError("structured_v2 requires ingestion_version=2")

    chunks = []
    for page in pages:
        if strategy == "markdown_v1":
            for chunk in _markdown_chunks([page], chunk_size, chunk_overlap):
                if len(chunks) >= max_chunks:
                    raise InputLimitError(f"Document exceeds {max_chunks} chunks")
                chunk["metadata"]["chunk_id"] = f"{page['metadata']['document_id']}-{len(chunks)}"
                chunks.append(chunk)
            continue
        if strategy=="structured_v2":
            source=page["text"].replace("\r\n","\n").replace("\r","\n")
            source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest()
            for start,end,headings,hard in _structured_spans(source,chunk_size,chunk_overlap):
                if len(chunks)>=max_chunks:raise InputLimitError(f"Document exceeds {max_chunks} chunks")
                metadata={**page["metadata"],"chunk_id":f"{page['metadata']['document_id']}-{len(chunks)}",
                    "char_start":start,"char_end":end,"source_sha256":source_hash,
                    "source_representation":"extracted_page_lf_v1" if page["metadata"].get("page") is not None else "utf8_lf_v1",
                    "offset_unit":"unicode_character","strategy":strategy,"ingestion_version":2,
                    "normalization_version":2,"heading_path":headings,"hard_split":hard}
                chunks.append({"text":source[start:end],"metadata":metadata})
            continue
        text = _clean_text(page["text"])
        if not text:
            continue

        for chunk_text in _split_text(text, chunk_size, chunk_overlap):
            if len(chunks) >= max_chunks:
                raise InputLimitError(f"Document exceeds {max_chunks} chunks")
            metadata = dict(page["metadata"])
            metadata["chunk_id"] = f"{metadata['document_id']}-{len(chunks)}"
            chunks.append({"text": chunk_text, "metadata": metadata})

    return chunks


def _sections(text):
    """Streaming ATX heading sections; no HTML/Markdown parser dependency."""
    cursor=0;headings=[]
    for match in re.finditer(r"(?m)^(#{1,6})[ \t]+([^\n]+)",text):
        if match.start()>cursor:yield cursor,match.start(),[title for level,title in headings]
        level=len(match[1]);title=match[2].rstrip(" #\t")[:128]
        headings=[h for h in headings if h[0]<level]+[(level,title)]
        cursor=match.start()
    if cursor<len(text):yield cursor,len(text),[title for level,title in headings]


def _structured_spans(text,size,overlap):
    for section_start,section_end,headings in _sections(text):
        start=section_start
        previous_end=section_start
        while start<section_end:
            limit=min(start+size,section_end)
            end=limit
            if limit<section_end:
                # Preserve whole paragraphs/lines where possible before word fallback.
                paragraph=text.rfind("\n\n",start,limit)
                newline=text.rfind("\n",start,limit)
                if paragraph>=start and paragraph+2>max(start,previous_end):end=paragraph+2
                elif newline+1>max(start,previous_end):end=newline+1
                elif not text[limit].isspace() and not text[limit-1].isspace():
                    boundary=limit
                    while boundary>start and not text[boundary-1].isspace():boundary-=1
                    if boundary>max(start,previous_end):end=boundary
            previous_end=end
            left,right=start,end
            if text[left:right].strip():
                hard=(left>section_start and not text[left-1].isspace() and not text[left].isspace()) or (right<section_end and not text[right-1].isspace() and not text[right].isspace())
                yield left,right,list(headings),bool(hard)
            if end>=section_end:break
            next_start=max(end-overlap,start+1)
            # Move right to the next word boundary; overlap is a target, not padding.
            if next_start>section_start and not text[next_start-1].isspace():
                while next_start<end and not text[next_start].isspace():next_start+=1
            while next_start<end and text[next_start].isspace():next_start+=1
            start=max(next_start,start+1)


def _clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def _split_text(text, chunk_size, chunk_overlap):
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            yield chunk

        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# The row under a Markdown table's header: |---|:---:|---|
TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")


def _markdown_chunks(pages, chunk_size=800, chunk_overlap=100):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size")

    count = 0
    for page in pages:
        for headings, section_text in _split_sections(page["text"]):
            # Every chunk carries its heading path, so a chunk from the middle of a
            # section still says which document and section it belongs to.
            prefix = " > ".join(title[:128] for title in headings)[:max(0, chunk_size // 4 - 2)]
            body_size = chunk_size - len(prefix) - 2 if prefix else chunk_size
            body_size = max(body_size, chunk_size // 2)
            overlap = min(chunk_overlap, body_size - 1)

            for block in _split_tables(section_text):
                text = _clean_text(block)
                if not text:
                    continue
                for chunk_text in _markdown_split_text(text, body_size, overlap):
                    metadata = dict(page["metadata"])
                    metadata["chunk_id"] = f"{metadata['document_id']}-{count}"
                    if prefix:
                        metadata["section"] = prefix
                        chunk_text = f"{prefix}: {chunk_text}"
                    count += 1
                    yield {"text": chunk_text, "metadata": metadata}



def _split_tables(text):
    """Split text into prose blocks and one block per Markdown table row.

    A table row on its own ("| Kırtasiye | %14 | 4 iş günü |") doesn't say what its
    values are, so each row is rewritten with its column names: "Ürün grubu:
    Kırtasiye; Bayi iskontosu: %14; ...". Rows become separate chunks, so a long table
    isn't cut into pieces that lose the header row.
    """
    prose = []
    lines = text.splitlines()
    position = 0
    while position < len(lines):
        line = lines[position].strip()
        is_table = (
            line.startswith("|")
            and position + 1 < len(lines)
            and TABLE_SEPARATOR.match(lines[position + 1].strip())
        )
        if not is_table:
            prose.append(lines[position])
            position += 1
            continue

        yield "\n".join(prose)
        prose = []
        header = _table_cells(line)
        position += 2
        while position < len(lines) and lines[position].strip().startswith("|"):
            cells = _table_cells(lines[position].strip())
            yield "; ".join(f"{name}: {value}" for name, value in zip(header, cells) if value)
            position += 1

    yield "\n".join(prose)


def _table_cells(line):
    return [cell.strip() for cell in line.strip("|").split("|")]


def _split_sections(text):
    """Split Markdown-style text at headings into (heading path, body) pairs.

    Text without headings comes back as a single section with an empty path.
    """
    headings = []
    lines = []
    for line in text.splitlines():
        match = HEADING_PATTERN.match(line.strip())
        if not match:
            lines.append(line)
            continue

        yield list(headings), "\n".join(lines)
        lines = []
        level = len(match.group(1))
        headings = headings[: level - 1] + [match.group(2)]

    yield list(headings), "\n".join(lines)



def _markdown_split_text(text, chunk_size, chunk_overlap):
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        if text[start:end].strip():
            yield text[start:end].strip()
        if end >= len(text):
            break
        next_start = max(end - chunk_overlap, start + 1)
        if next_start > 0 and not text[next_start - 1].isspace():
            boundary = text.find(" ", next_start, end)
            next_start = boundary + 1 if boundary >= 0 else end
        start = max(next_start, start + 1)
