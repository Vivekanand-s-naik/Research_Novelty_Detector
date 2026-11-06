from types import SimpleNamespace

import pytest

from src.config import ConfigError
from src.paper_comparator import ComparisonError, PaperComparator
from src.report_generator import ReportGenerator, split_llm_sections

LLM_ANSWER = """## Paper-by-Paper Comparison
- [P1] tackles the same molecule property task.
## Similarities
- Both use GNNs [P1].
## Differences
- Uploaded paper adds geometry.
## Potential Research Gaps
- Insufficient evidence for [P2] methods (abstract only).
## Potential Unique Contributions of the Uploaded Paper
- Geometric attention appears new relative to [P1].
## Evidence Quality and Limitations
- [P2] abstract only.
"""


class FakeGroq:
    def __init__(self, answer=LLM_ANSWER, error=None):
        self.calls = []
        self.answer = answer
        self.error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(
            create=self._create))

    def _create(self, **params):
        self.calls.append(params)
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.answer)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=usage)


def ranked_item(title, evidence="abstract_only", same=False, score=0.6):
    return {
        "title": title, "url": f"https://x.org/{title.replace(' ', '')}",
        "evidence_level": evidence, "evidence_description": evidence,
        "overall_similarity": score, "band": "moderate",
        "explanation": "explained", "basis": "basis",
        "likely_same_paper": same, "rank": None if same else 1,
        "aspects": {
            name: {"label": name.title(), "score": 0.5, "band": "low",
                   "explanation": "e", "evidence_passage": f"{title} passage"}
            for name in ("research_problem", "methodology", "contributions",
                         "objectives")
        },
        "record": {"title": title, "abstract": f"Abstract of {title}.",
                   "authors": ["A. Author"], "errors": []},
    }


PAPER = {"title": "Geometric GNNs", "file_name": "paper.pdf",
         "total_pages": 8, "warnings": [], "metadata": {}}
SECTIONS = {"abstract": "We study geometric GNNs.", "methodology": "GNN."}


def test_single_llm_call_with_selected_sources():
    client = FakeGroq()
    comparator = PaperComparator(client=client, model="openai/gpt-oss-20b",
                                 top_k=2)
    ranked = [ranked_item("Itself", same=True),
              ranked_item("Paper One"),
              ranked_item("Snippet Paper", evidence="snippet_only"),
              ranked_item("Paper Two", evidence="full_text"),
              ranked_item("Paper Three")]
    result = comparator.compare(PAPER, SECTIONS, ranked)

    assert len(client.calls) == 1
    assert result["status"] == "completed"
    assert [s["title"] for s in result["sources"]] == ["Paper One",
                                                       "Paper Two"]
    prompt = client.calls[0]["messages"][1]["content"]
    assert "Itself" not in prompt and "Snippet Paper" not in prompt
    assert "SOURCE [P1]" in prompt and "https://x.org/PaperOne" in prompt
    assert "Evidence level: abstract_only" in prompt
    assert "[not found in extracted text]" in prompt   # missing sections
    assert client.calls[0]["reasoning_effort"] == "low"


def test_no_usable_sources_skips_llm():
    client = FakeGroq()
    comparator = PaperComparator(client=client)
    result = comparator.compare(PAPER, SECTIONS, [
        ranked_item("Snippet", evidence="snippet_only")])
    assert client.calls == []
    assert result["status"] == "skipped"
    assert "Insufficient evidence" in result["markdown"]


def test_citation_validation_flags_unknown_ids():
    sources = [{"id": "P1"}, {"id": "P2"}]
    warnings = PaperComparator.validate_citations("Claim [P1] and [P7]",
                                                  sources)
    assert any("P7" in w for w in warnings)
    assert any("P2" in w for w in warnings)
    assert PaperComparator.validate_citations("no cites", sources)


def test_groq_error_wrapped():
    comparator = PaperComparator(client=FakeGroq(error=RuntimeError("429")))
    with pytest.raises(ComparisonError, match="429"):
        comparator.compare(PAPER, SECTIONS, [ranked_item("Paper One")])


def test_missing_groq_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        _ = PaperComparator().client


def test_model_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    client = FakeGroq()
    comparator = PaperComparator(client=client)
    comparator.compare(PAPER, SECTIONS, [ranked_item("Paper One")])
    assert client.calls[0]["model"] == "llama-3.1-8b-instant"
    assert "reasoning_effort" not in client.calls[0]


# --- Report generator -------------------------------------------------------
def test_split_llm_sections():
    parts = split_llm_sections(LLM_ANSWER)
    assert set(parts) == {"comparison", "similarities", "differences",
                          "gaps", "unique", "limitations"}


def test_report_contains_all_sections_and_saves(tmp_path):
    comparison = PaperComparator(client=FakeGroq(), top_k=2).compare(
        PAPER, SECTIONS, [ranked_item("Paper One"), ranked_item("Paper Two")])
    results = {
        "paper": PAPER, "sections": SECTIONS,
        "headings": {"abstract": ["Abstract"]},
        "queries": [{"query": "geometric gnn", "type": "title",
                     "keyphrases": []}],
        "search": {"api_calls": 1, "errors": []},
        "ranked": [ranked_item("Paper One"), ranked_item("Paper Two")],
        "comparison": comparison,
    }
    reporter = ReportGenerator(output_dir=tmp_path)
    markdown = reporter.build_markdown(results)

    for number in range(1, 12):
        assert f"## {number}. " in markdown
    assert "not** novelty percentages" in markdown
    assert "https://x.org/PaperOne" in markdown
    assert "[P1] Paper One" in markdown

    paths = reporter.save(results, markdown)
    assert paths["markdown"].exists() and paths["json"].exists()


def test_report_without_llm(tmp_path):
    results = {"paper": PAPER, "sections": {}, "queries": [], "ranked": [],
               "search": {}, "comparison": None}
    markdown = ReportGenerator(output_dir=tmp_path).build_markdown(results)
    assert "LLM comparison was not run" in markdown
    assert "No related papers" in markdown

