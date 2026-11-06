
import pytest

from src.pdf_processor import (OCRRequiredError, PDFProcessingError,
                               PDFProcessor)
from tests.conftest import make_pdf

processor = PDFProcessor()


def test_sample_pdf_extraction(sample_pdf):
    result = processor.extract_text(sample_pdf)

    assert result["file_name"] == "sample.pdf"
    assert result["total_pages"] == len(result["pages"]) > 0
    assert result["pages"][0]["page_number"] == 1
    assert len(result["full_text"]) > 1000
    assert result["title"]


def test_page_numbers_preserved(tmp_path):
    pdf = make_pdf(tmp_path / "two.pdf", ["First page text here " * 3,
                                          "Second page text here " * 3])
    result = processor.extract_text(pdf)

    assert [p["page_number"] for p in result["pages"]] == [1, 2]
    assert "Second page" in result["pages"][1]["text"]
    assert result["ocr_required"] is False


def test_extract_from_bytes(tmp_path):
    pdf = make_pdf(tmp_path / "b.pdf", ["Bytes based extraction works " * 3])
    result = processor.extract_text(pdf_bytes=pdf.read_bytes(),
                                    file_name="upload.pdf")
    assert result["file_name"] == "upload.pdf"
    assert "Bytes based" in result["full_text"]


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        processor.extract_text("does_not_exist.pdf")


def test_wrong_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(ValueError):
        processor.extract_text(path)


def test_invalid_pdf(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf at all")
    with pytest.raises(PDFProcessingError):
        processor.extract_text(path)


def test_empty_bytes():
    with pytest.raises(PDFProcessingError):
        processor.extract_text(pdf_bytes=b"")


def test_scanned_pdf_requires_ocr(tmp_path):
    # A PDF with blank pages behaves like an image-only scan.
    pdf = make_pdf(tmp_path / "scan.pdf", ["", ""])
    with pytest.raises(OCRRequiredError, match="OCR"):
        processor.extract_text(pdf)


def test_partially_scanned_pdf_warns(tmp_path):
    pdf = make_pdf(tmp_path / "mixed.pdf",
                   ["Readable text on this page " * 3, ""])
    result = processor.extract_text(pdf)
    assert result["pages_without_text"] == [2]
    assert result["warnings"] and "OCR" in result["warnings"][0]


if __name__ == "__main__":
    # Original manual check, kept for convenience:
    #   python -m tests.test_pdf_processor
    result = processor.extract_text("data/uploaded_papers/sample.pdf")
    print("File:", result["file_name"])
    print("Total Pages:", result["total_pages"])
    print("\nExtracted Text:\n")
    print(result["full_text"][:2000])
