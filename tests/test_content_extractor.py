import requests

from src.content_extractor import ContentExtractor
from tests.conftest import make_pdf

ABSTRACT = ("We propose a novel transformer model for protein folding that "
            "uses geometric attention and achieves strong results. " * 4)


class FakeResponse:
    def __init__(self, status=200, text="", content=None,
                 content_type="text/html", url=None):
        self.status_code = status
        self.text = text
        self.content = content if content is not None else text.encode()
        self.headers = {"Content-Type": content_type}
        self.url = url


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.headers = {}
        self.requested = []

    def get(self, url, timeout=None, allow_redirects=True):
        self.requested.append(url)
        if url.endswith("/robots.txt"):
            return self.routes.get(url, FakeResponse(404))
        value = self.routes.get(url)
        if isinstance(value, Exception):
            raise value
        if value is None:
            return FakeResponse(404)
        value.url = value.url or url
        return value


def html_page(abstract=ABSTRACT, pdf_url=None, extra_meta=""):
    pdf = f'<meta name="citation_pdf_url" content="{pdf_url}">' \
        if pdf_url else ""
    return f"""<html><head><title>Page</title>
<meta name="citation_title" content="Protein Folding with Transformers">
<meta name="citation_author" content="Ada Lovelace">
<meta name="citation_publication_date" content="2023/05/01">
<meta name="citation_abstract" content="{abstract}">
{pdf}{extra_meta}</head><body><p>short</p></body></html>"""


def make_extractor(routes, tmp_path, **kwargs):
    return ContentExtractor(session=FakeSession(routes),
                            cache_dir=tmp_path / "cache", **kwargs)


def search_result(url, snippet="tavily snippet"):
    return {"url": url, "title": "Search title", "snippet": snippet,
            "score": 0.8, "query": "q"}


def test_abstract_only_from_meta_tags(tmp_path):
    url = "https://journal.org/paper/1"
    extractor = make_extractor({url: FakeResponse(text=html_page())},
                               tmp_path)
    record = extractor.extract(search_result(url))

    assert record["evidence_level"] == "abstract_only"
    assert record["title"] == "Protein Folding with Transformers"
    assert record["authors"] == ["Ada Lovelace"]
    assert record["published"] == "2023/05/01"
    assert record["content"].startswith("We propose a novel transformer")
    assert record["url"] == url
    assert record["status"] == "success"


def test_full_text_from_linked_pdf(tmp_path):
    url = "https://journal.org/paper/2"
    pdf_url = "https://journal.org/paper/2.pdf"
    body = "\n".join(["Abstract", ABSTRACT[:300], "1 Introduction"] +
                     [f"Line {i} discussing protein folding in detail."
                      for i in range(45)])
    pdf_path = make_pdf(tmp_path / "p.pdf", [body, body, body, body])
    routes = {
        url: FakeResponse(text=html_page(pdf_url=pdf_url)),
        pdf_url: FakeResponse(content=pdf_path.read_bytes(),
                              content_type="application/pdf"),
    }
    record = make_extractor(routes, tmp_path).extract(search_result(url))

    assert record["evidence_level"] == "full_text"
    assert record["full_text_url"] == pdf_url
    assert "protein folding in detail" in record["content"]


def test_blocked_page_falls_back_to_snippet(tmp_path):
    url = "https://paywalled.org/x"
    record = make_extractor({url: FakeResponse(403)}, tmp_path).extract(
        search_result(url))

    assert record["evidence_level"] == "snippet_only"
    assert record["status"] == "failed"
    assert record["blocked"] is True
    assert record["content"] == "tavily snippet"
    assert any("403" in e for e in record["errors"])


def test_timeout_is_recorded(tmp_path):
    url = "https://slow.org/x"
    record = make_extractor({url: requests.Timeout()}, tmp_path).extract(
        search_result(url))
    assert record["evidence_level"] == "snippet_only"
    assert any("Timeout" in e for e in record["errors"])


def test_connection_error_is_recorded(tmp_path):
    url = "https://down.org/x"
    record = make_extractor({url: requests.ConnectionError("dns")},
                            tmp_path).extract(search_result(url))
    assert record["status"] == "failed"
    assert any("Request failed" in e for e in record["errors"])


def test_robots_txt_respected(tmp_path):
    url = "https://private.org/paper"
    routes = {
        "https://private.org/robots.txt": FakeResponse(
            text="User-agent: *\nDisallow: /"),
        url: FakeResponse(text=html_page()),
    }
    extractor = make_extractor(routes, tmp_path)
    record = extractor.extract(search_result(url))

    assert record["evidence_level"] == "snippet_only"
    assert any("robots.txt" in e for e in record["errors"])
    assert url not in extractor.session.requested


def test_unsupported_content_type(tmp_path):
    url = "https://data.org/file.zip"
    routes = {url: FakeResponse(content=b"PK..", content_type="application/zip")}
    record = make_extractor(routes, tmp_path).extract(search_result(url))
    assert any("Unsupported content type" in e for e in record["errors"])


def test_short_meta_description_is_not_an_abstract(tmp_path):
    url = "https://journal.org/paper/3"
    routes = {url: FakeResponse(text=html_page(abstract="Too short."))}
    record = make_extractor(routes, tmp_path).extract(search_result(url))
    assert record["evidence_level"] == "snippet_only"


def test_successful_results_are_cached(tmp_path):
    url = "https://journal.org/paper/1"
    routes = {url: FakeResponse(text=html_page())}
    make_extractor(routes, tmp_path).extract(search_result(url))

    second = make_extractor({}, tmp_path)       # no routes: must use cache
    record = second.extract(search_result(url))
    assert record["from_cache"] is True
    assert record["evidence_level"] == "abstract_only"


def test_extract_many_empty():
    assert ContentExtractor(session=FakeSession({}), cache_dir=None) \
        .extract_many([]) == []

