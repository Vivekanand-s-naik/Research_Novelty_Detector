"""
Module 1: PDF Processor

Extracts text from PDFs locally with PyMuPDF while preserving page numbers.
Invalid, encrypted and scanned (image-only) PDFs are reported clearly.
"""

from pathlib import Path

import pymupdf

# A page with fewer characters than this is treated as "probably scanned".
MIN_CHARS_PER_PAGE = 25


class PDFProcessingError(ValueError):
    """Raised when a PDF cannot be processed (invalid, encrypted, scanned)."""


class OCRRequiredError(PDFProcessingError):
    """Raised when a PDF contains no extractable text (needs OCR)."""


class PDFProcessor:

    def extract_text(self, pdf_path=None, pdf_bytes=None, file_name=None):
        """
        Extract text from a PDF file path or from raw bytes.

        Returns a dict with:
            file_name, title, metadata, total_pages, pages (page_number,
            text, char_count), full_text, pages_without_text, ocr_required,
            warnings
        """
        if pdf_bytes is None:
            if pdf_path is None:
                raise ValueError("Provide either pdf_path or pdf_bytes")

            pdf_path = Path(pdf_path)

            if not pdf_path.exists():
                raise FileNotFoundError(
                    f"PDF not found: {pdf_path}"
                )

            if pdf_path.suffix.lower() != ".pdf":
                raise ValueError("Only PDF files are supported")

            file_name = file_name or pdf_path.name
            open_kwargs = {"filename": str(pdf_path)}
        else:
            if not pdf_bytes:
                raise PDFProcessingError("The uploaded PDF file is empty.")
            file_name = file_name or "uploaded.pdf"
            open_kwargs = {"stream": pdf_bytes, "filetype": "pdf"}

        try:
            document = pymupdf.open(**open_kwargs)
        except Exception as error:  # PyMuPDF raises several error types
            raise PDFProcessingError(
                f"Invalid or corrupted PDF '{file_name}': {error}"
            ) from error

        with document:

            if document.needs_pass:
                raise PDFProcessingError(
                    f"'{file_name}' is password protected. "
                    "Please upload an unencrypted PDF."
                )

            if document.page_count == 0:
                raise PDFProcessingError(f"'{file_name}' has no pages.")

            pages = []

            for page_number, page in enumerate(document, start=1):

                text = page.get_text("text").strip()

                pages.append({
                    "page_number": page_number,
                    "text": text,
                    "char_count": len(text)
                })

            metadata = {
                key: value
                for key, value in (document.metadata or {}).items()
                if value
            }
            title = self._detect_title(document, metadata)

        full_text = "\n".join(
            page["text"] for page in pages
        )

        if not full_text.strip():
            raise OCRRequiredError(
                "No extractable text found. "
                "This PDF may require OCR."
            )

        pages_without_text = [
            page["page_number"]
            for page in pages
            if page["char_count"] < MIN_CHARS_PER_PAGE
        ]

        warnings = []
        if pages_without_text:
            warnings.append(
                f"{len(pages_without_text)} page(s) contain little or no "
                f"extractable text (pages {pages_without_text}). They may "
                "be scanned images and would require OCR to be analysed."
            )

        return {
            "file_name": file_name,
            "title": title,
            "metadata": metadata,
            "total_pages": len(pages),
            "pages": pages,
            "full_text": full_text,
            "pages_without_text": pages_without_text,
            # OCR is "required" if most pages have no text layer.
            "ocr_required": len(pages_without_text) > len(pages) / 2,
            "warnings": warnings
        }

    @staticmethod
    def _detect_title(document, metadata):
        """
        Use the PDF metadata title when meaningful, otherwise take the
        largest-font text on the first page (a common title heuristic).
        """
        meta_title = (metadata.get("title") or "").strip()
        bad_titles = {"untitled", "microsoft word", "title", "paper"}

        if (len(meta_title) > 10
                and not meta_title.lower().endswith((".pdf", ".doc", ".docx"))
                and not any(b in meta_title.lower() for b in bad_titles)):
            return meta_title

        try:
            blocks = document[0].get_text("dict")["blocks"]
        except Exception:
            return meta_title or None

        spans = []
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if len(text) > 1:
                        spans.append((round(span.get("size", 0), 1), text))

        if not spans:
            return meta_title or None

        largest = max(size for size, _ in spans)
        title = " ".join(text for size, text in spans if size == largest)
        title = " ".join(title.split())

        # Reject implausible titles (e.g. a single big letter or a logo).
        if 10 <= len(title) <= 300:
            return title
        return meta_title or None
