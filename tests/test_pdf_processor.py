
from src.pdf_processor import PDFProcessor

processor = PDFProcessor()

result = processor.extract_text(
    "data/uploaded_papers/sample.pdf"
)

print("File:", result["file_name"])
print("Total Pages:", result["total_pages"])

print("\nExtracted Text:\n")

print(result["full_text"][:2000])
