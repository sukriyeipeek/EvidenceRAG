from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

from .ocr import OCRUnavailableError, read_page_images


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class DocumentError(Exception):
    """Base exception for document loading errors."""


class UnsupportedFileError(DocumentError):
    """Raised when a file format is not supported."""


class EmptyDocumentError(DocumentError):
    """Raised when a document contains no extractable text."""


def load_document(path, filename=None, source=None, document_id=None, ocr_languages=()):
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileError(
            f"Unsupported file type '{extension}'. Supported types: {supported}"
        )

    document_id = document_id or uuid4().hex
    filename = filename or path.name
    source = source or str(path)

    if extension == ".pdf":
        pages = _load_pdf(path, ocr_languages)
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentError(f"Document is not valid UTF-8 text: {filename}") from exc
        pages = [{"text": text, "page": None}]

    if not any(page["text"].strip() for page in pages):
        raise EmptyDocumentError(f"Document contains no extractable text: {filename}")

    return [
        {
            "text": page["text"],
            "metadata": {
                "filename": filename,
                "document_id": document_id,
                "page": page["page"],
                "source": source,
                # Lets the caller warn that the text was read from an image and may
                # contain recognition errors.
                **({"ocr": True} if page.get("ocr") else {}),
            },
        }
        for page in pages
        if page["text"].strip()
    ]


def _load_pdf(path, ocr_languages=()):
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise DocumentError(f"Could not read PDF document: {path.name}") from exc

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip() or not ocr_languages or not page.images:
            pages.append({"text": text, "page": page_number})
            continue
        # A page without a text layer but with images is most likely scanned.
        try:
            text = read_page_images(page, ocr_languages)
        except OCRUnavailableError as exc:
            raise EmptyDocumentError(
                f"{path.name} looks scanned and has no text layer, and OCR is unavailable: {exc}"
            ) from exc
        pages.append({"text": text, "page": page_number, "ocr": True})
    return pages
