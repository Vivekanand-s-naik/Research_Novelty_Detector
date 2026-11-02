
from src.pdf_processor import PDFProcessor
from src.section_extractor import SectionExtractor

# Extract PDF text
pdf = PDFProcessor()

paper = pdf.extract_text(
    "data/uploaded_papers/sample.pdf"
)

# Extract sections locally
extractor = SectionExtractor()

sections = extractor.extract_sections(
    paper["full_text"]
)

# Display results
for section, content in sections.items():

    print("\n", "=" * 40)
    print(section.upper())
    print("=" * 40)

    print(content[:500] if content else "Not found")
