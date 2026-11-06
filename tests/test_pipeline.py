"""End-to-end pipeline test with all external services mocked."""

from src.pipeline import analyze_paper, run_research
from src.web_search import WebSearcher
from tests.conftest import make_pdf
from tests.test_comparator_and_report import FakeGroq
from tests.test_content_extractor import (FakeResponse, FakeSession,
                                          html_page)
from tests.test_query_generator import FakeExtractor
from tests.test_web_search import FakeTavily, result

PAPER_TEXT = """Protein Folding with Geometric Transformers
Abstract
We propose a novel transformer model for protein folding that uses
geometric attention over residue graphs and improves accuracy.
1 Introduction
Protein structure prediction is a central problem in biology. However,
existing models struggle with long-range contacts.
2 Methodology
We build residue graphs and apply geometric attention layers with
equivariant updates trained on structural data.
3 Conclusion
Geometric attention helps protein folding.
"""


def test_full_pipeline_mocked(tmp_path, fake_embedder, monkeypatch):
    import src.report_generator as rg
    monkeypatch.setattr(rg, "OUTPUT_DIR", tmp_path)

    pdf = make_pdf(tmp_path / "paper.pdf", [PAPER_TEXT])
    extractor = FakeExtractor([
        [("protein folding", 0.8), ("geometric attention", 0.7)],
        [("residue graphs", 0.7), ("equivariant updates", 0.6)],
    ])
    analysis = analyze_paper(pdf_path=pdf, query_generator=extractor and
                             __import__("src.query_generator",
                                        fromlist=["QueryGenerator"])
                             .QueryGenerator(keyword_extractor=extractor))
    assert analysis["sections"]["abstract"]
    assert analysis["queries"]

    urls = ["https://journal.org/a", "https://journal.org/b"]
    tavily = FakeTavily({q["query"]: [result(urls[0], "Paper A", 0.9),
                                      result(urls[1], "Paper B", 0.7)]
                         for q in analysis["queries"]})
    session = FakeSession({urls[0]: FakeResponse(text=html_page()),
                           urls[1]: FakeResponse(403)})
    from src.content_extractor import ContentExtractor
    from src.paper_comparator import PaperComparator
    from src.similarity_engine import SimilarityEngine
    groq = FakeGroq()

    results = run_research(
        analysis,
        searcher=WebSearcher(client=tavily, cache_dir=tmp_path / "s"),
        extractor=ContentExtractor(session=session,
                                   cache_dir=tmp_path / "c"),
        similarity=SimilarityEngine(fake_embedder),
        comparator=PaperComparator(client=groq, model="test-model"),
        save_report=False,
    )

    assert len(results["search"]["results"]) == 2       # deduplicated
    levels = {r["url"]: r["evidence_level"] for r in results["retrieved"]}
    assert levels == {urls[0]: "abstract_only", urls[1]: "snippet_only"}
    assert len(groq.calls) == 1                        # single LLM call
    assert results["comparison"]["status"] == "completed"
    assert [s["url"] for s in results["comparison"]["sources"]] == [urls[0]]
    assert "## 11. Source References" in results["report_markdown"]


def test_pipeline_reports_missing_groq_key(tmp_path, fake_embedder,
                                           monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from src.query_generator import QueryGenerator
    pdf = make_pdf(tmp_path / "paper.pdf", [PAPER_TEXT])
    analysis = analyze_paper(pdf_path=pdf, query_generator=QueryGenerator(
        keyword_extractor=FakeExtractor([[("protein folding", 0.8)]])))

    url = "https://journal.org/a"
    from src.content_extractor import ContentExtractor
    from src.similarity_engine import SimilarityEngine
    results = run_research(
        analysis,
        searcher=WebSearcher(client=FakeTavily(
            {q["query"]: [result(url)] for q in analysis["queries"]}),
            cache_dir=tmp_path / "s"),
        extractor=ContentExtractor(
            session=FakeSession({url: FakeResponse(text=html_page())}),
            cache_dir=tmp_path / "c"),
        similarity=SimilarityEngine(fake_embedder),
        save_report=False,
    )
    assert results["comparison"]["status"] == "failed"
    assert any("GROQ_API_KEY" in e for e in results["errors"])

