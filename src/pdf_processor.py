
import pymupdf
from pathlib import Path


class PDFProcessor:

    def extract_text(self, pdf_path):

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported")

        pages = []

        with pymupdf.open(pdf_path) as document:

            for page_number, page in enumerate(document, start=1):

                text = page.get_text("text").strip()

                pages.append({
                    "page_number": page_number,
                    "text": text
                })

        full_text = "\n".join(
            page["text"] for page in pages
        )

        if not full_text.strip():
            raise ValueError(
                "No extractable text found. "
                "This PDF may require OCR."
            )

        return {
            "file_name": pdf_path.name,
            "total_pages": len(pages),
            "pages": pages,
            "full_text": full_text
        }
