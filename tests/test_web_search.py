import pytest

from src.config import ConfigError
from src.web_search import WebSearcher, normalize_url


class FakeTavily:
    def __init__(self, responses=None, fail_on=None):
        self.responses = responses or {}
        self.fail_on = fail_on or set()
        self.calls = []

    def search(self, **params):
        self.calls.append(params)
        if params["query"] in self.fail_on:
            raise RuntimeError("quota exceeded")
        return {"results": self.responses.get(params["query"], [])}


def result(url, title="T", score=0.5, content="snippet"):
    return {"url": url, "title": title, "score": score, "content": content}


@pytest.mark.parametrize("a, b", [
    ("https://arxiv.org/abs/2401.01234v2", "http://arxiv.org/pdf/2401.01234"),
    ("https://arxiv.org/pdf/2401.01234v1.pdf", "arxiv.org/abs/2401.01234"),
    ("https://www.example.com/paper/", "http://example.com/paper"),
    ("https://example.com/p?utm_source=x&id=3", "https://example.com/p?id=3"),
    ("https://example.com/p#section-2", "https://example.com/p"),
    ("https://doi.org/10.1000/ABC", "https://doi.org/10.1000/abc"),
])
def test_normalize_url_equivalents(a, b):
    assert normalize_url(a) == normalize_url(b)


def test_normalize_url_keeps_distinct_pages():
    assert normalize_url("https://example.com/a") != \
        normalize_url("https://example.com/b")
    assert normalize_url("") == ""


def test_search_deduplicates_and_keeps_fields(tmp_path):
    client = FakeTavily({
        "q1": [result("https://arxiv.org/abs/2401.01234", "Paper A", 0.9),
               result("https://example.org/b", "Paper B", 0.4)],
        "q2": [result("https://arxiv.org/pdf/2401.01234v2", "Paper A", 0.95)],
    })
    searcher = WebSearcher(client=client, cache_dir=tmp_path)
    output = searcher.search([{"query": "q1"}, {"query": "q2"}])

    results = output["results"]
    assert len(results) == 2
    top = results[0]
    assert top["title"] == "Paper A"
    assert top["query"] == "q1"
    assert top["also_found_by"] == ["q2"]
    assert top["score"] == 0.95            # best score kept
    assert top["snippet"] == "snippet"
    assert output["api_calls"] == 2


def test_uses_basic_depth_and_limits(tmp_path):
    client = FakeTavily()
    searcher = WebSearcher(client=client, max_queries=2,
                           results_per_query=4, cache_dir=tmp_path)
    searcher.search(["a", "b", "c", "a"])

    assert len(client.calls) == 2
    assert all(c["search_depth"] == "basic" for c in client.calls)
    assert all(c["max_results"] == 4 for c in client.calls)


def test_cache_avoids_repeat_api_calls(tmp_path):
    client = FakeTavily({"q": [result("https://x.org/1")]})
    WebSearcher(client=client, cache_dir=tmp_path).search(["q"])
    second = WebSearcher(client=client, cache_dir=tmp_path).search(["q"])

    assert len(client.calls) == 1
    assert second["api_calls"] == 0
    assert len(second["results"]) == 1


def test_search_errors_are_reported_not_raised(tmp_path):
    client = FakeTavily({"ok": [result("https://x.org/1")]},
                        fail_on={"bad"})
    output = WebSearcher(client=client, cache_dir=tmp_path).search(
        ["bad", "ok"])
    assert len(output["results"]) == 1
    assert output["errors"][0]["query"] == "bad"


def test_results_without_url_are_skipped(tmp_path):
    client = FakeTavily({"q": [{"title": "no url", "content": "x"}]})
    output = WebSearcher(client=client, cache_dir=tmp_path).search(["q"])
    assert output["results"] == []


def test_missing_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    searcher = WebSearcher(cache_dir=tmp_path)
    with pytest.raises(ConfigError, match="TAVILY_API_KEY"):
        _ = searcher.client

