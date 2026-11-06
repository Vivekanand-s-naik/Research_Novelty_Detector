
from src.pdf_processor import PDFProcessor
from src.section_extractor import SectionExtractor

extractor = SectionExtractor()

PAPER = """Some Title of a Paper
Abstract
We study graph neural networks for molecule property prediction.
However, existing models ignore 3D geometry.
Keywords: graphs, molecules
1
INTRODUCTION
Molecules are important. Our main contributions are as follows: (i) a
new geometric encoder; (ii) a benchmark.
2. Related Work
Prior GNNs such as GCN were proposed.
3 Proposed Method
We use an equivariant message passing net-
work with attention.
3.1 Training details
Adam optimizer was used.
Figure 2
4 Datasets
QM9 and ZINC were downloaded.
5 Experimental Results
Our model improves MAE by 10%.
6 Conclusion and Future Work
We presented a geometric GNN.
References
[1] Kipf and Welling. GCN. 2017.
"""


def test_basic_sections():
    sections = extractor.extract_sections(PAPER)

    assert sections["abstract"].startswith("We study graph neural networks")
    assert "Molecules are important" in sections["introduction"]
    assert "Prior GNNs" in sections["literature_review"]
    assert "equivariant message passing network" in sections["methodology"]
    assert "Our model improves" in sections["results"]
    assert "geometric GNN" in sections["conclusion"]


def test_keywords_line_ends_abstract():
    sections = extractor.extract_sections(PAPER)
    assert "Keywords" not in sections["abstract"]
    assert "graphs, molecules" not in sections["abstract"]


def test_unknown_subheading_stays_in_parent():
    sections = extractor.extract_sections(PAPER)
    assert "Adam optimizer" in sections["methodology"]


def test_unknown_top_level_heading_ends_section():
    sections = extractor.extract_sections(PAPER)
    # "4 Datasets" is not a target section: its text must not leak
    # into methodology.
    assert "QM9" not in sections["methodology"]
    assert all("QM9" not in text for text in sections.values())


def test_references_not_in_conclusion():
    sections = extractor.extract_sections(PAPER)
    assert "Kipf" not in sections["conclusion"]


def test_hyphenation_repaired():
    sections = extractor.extract_sections(PAPER)
    assert "net- work" not in sections["methodology"]


def test_inferred_contributions_and_problem_are_labelled():
    result = extractor.extract(PAPER)
    assert "geometric encoder" in result["sections"]["contributions"]
    assert "inferred" in result["headings"]["contributions"][0]
    assert "ignore 3D geometry" in result["sections"]["problem_statement"]


def test_original_headings_preserved():
    result = extractor.extract(PAPER)
    assert result["headings"]["methodology"] == ["3 Proposed Method"]
    assert result["headings"]["introduction"] == ["1 INTRODUCTION"]


def test_inline_abstract_and_roman_numerals():
    text = """Abstract\u2014This paper proposes a fast parser for logs.
Index Terms\u2014parsing, logs
I. INTRODUCTION
Logs are everywhere.
II. METHODOLOGY
We tokenize lines.
III. CONCLUSION
It works.
"""
    sections = extractor.extract_sections(text)
    assert sections["abstract"] == "This paper proposes a fast parser for logs."
    assert sections["introduction"] == "Logs are everywhere."
    assert sections["methodology"] == "We tokenize lines."
    assert sections["conclusion"] == "It works."


def test_figure_labels_are_not_headings():
    text = """1 Introduction
Intro text here.
2 Motion
More intro text.
"""
    sections = extractor.extract_sections(text)
    # "2 Motion" is a valid next number and Title Case, so it IS treated
    # as a heading; but a repeated number like "1 Motion" must not be.
    text2 = "1 Introduction\nIntro text.\n1 Motion\nStill intro.\n"
    sections2 = extractor.extract_sections(text2)
    assert "Still intro." in sections2["introduction"]
    assert "Intro text here." in sections["introduction"]


def test_empty_text():
    sections = extractor.extract_sections("")
    assert set(sections) == set(SectionExtractor.SECTION_PATTERNS)
    assert all(value == "" for value in sections.values())


def test_sample_pdf_sections(sample_pdf):
    paper = PDFProcessor().extract_text(sample_pdf)
    result = extractor.extract(paper["full_text"])
    sections = result["sections"]

    assert sections["abstract"]
    assert sections["introduction"]
    assert sections["methodology"]
    # Reference list must not be appended to the conclusion
    assert "[1]" not in sections["conclusion"]


if __name__ == "__main__":
    # Original manual check, kept for convenience:
    #   python -m tests.test_sections
    pdf = PDFProcessor()
    paper = pdf.extract_text("data/uploaded_papers/sample.pdf")
    sections = extractor.extract_sections(paper["full_text"])
    for section, content in sections.items():
        print("\n", "=" * 40)
        print(section.upper())
        print("=" * 40)
        print(content[:500] if content else "Not found")
