import pytest
from PIL import Image

from app.rag import document_loader
from app.rag.document_loader import EmptyDocumentError, load_document
from app.rag.ocr import OCRUnavailableError


@pytest.fixture
def scanned_pdf(tmp_path):
    path = tmp_path / "taranmis.pdf"
    Image.new("L", (200, 100), color=255).save(path, "PDF")
    return path


def test_scanned_pages_are_read_with_ocr(scanned_pdf, monkeypatch):
    calls = []

    def fake_ocr(page, languages):
        calls.append(tuple(languages))
        return "Sayımda 315 koli bulundu."

    monkeypatch.setattr(document_loader, "read_page_images", fake_ocr)

    pages = load_document(scanned_pdf, ocr_languages=["tr"])

    assert calls == [("tr",)]
    assert pages[0]["text"] == "Sayımda 315 koli bulundu."
    assert pages[0]["metadata"]["page"] == 1
    assert pages[0]["metadata"]["ocr"] is True


def test_scanned_pdf_is_rejected_when_ocr_is_disabled(scanned_pdf):
    with pytest.raises(EmptyDocumentError):
        load_document(scanned_pdf, ocr_languages=[])


def test_scanned_pdf_is_rejected_when_ocr_is_not_installed(scanned_pdf, monkeypatch):
    def missing_ocr(page, languages):
        raise OCRUnavailableError("Reading scanned pages needs the easyocr package")

    monkeypatch.setattr(document_loader, "read_page_images", missing_ocr)

    with pytest.raises(EmptyDocumentError, match="OCR is unavailable"):
        load_document(scanned_pdf, ocr_languages=["tr"])


def test_text_files_are_not_marked_as_ocr(tmp_path):
    path = tmp_path / "notlar.txt"
    path.write_text("düz metin", encoding="utf-8")

    pages = load_document(path, ocr_languages=["tr"])

    assert "ocr" not in pages[0]["metadata"]
