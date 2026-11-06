"""
Module 5: Content Extractor

Fetches content for each search result, preferring (in order):

  1. full_text     - full paper text from an openly accessible PDF or a
                     full-text HTML page (e.g. arXiv, PMC, MDPI).
  2. abstract_only - abstract + metadata from scholarly meta tags
                     (citation_*, Dublin Core, arXiv abstract block).
  3. page_text     - generic page text that could not be verified as an
                     abstract or full text.
  4. snippet_only  - nothing could be fetched; only the Tavily snippet is
                     available (weakest evidence).

Rules followed:
  * robots.txt is respected; paywalled / blocked pages are not bypassed.
  * an honest User-Agent is sent (no browser spoofing).
  * timeouts, HTTP errors and parse failures are recorded per source.
  * metadata (authors, venue, date, DOI) is only reported when present
    on the page - it is never guessed.
"""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from src.config import CONTENT_CACHE_DIR
from src.pdf_processor import PDFProcessor
from src.section_extractor import SectionExtractor
from src.web_search import normalize_url

USER_AGENT = (
    "ResearchNoveltyDetector/0.1 "
    "(academic literature comparison tool; python-requests)"
)

FULL_TEXT = "full_text"
ABSTRACT_ONLY = "abstract_only"
PAGE_TEXT = "page_text"
SNIPPET_ONLY = "snippet_only"

EVIDENCE_DESCRIPTIONS = {
    FULL_TEXT: "Full text extracted",
    ABSTRACT_ONLY: "Abstract and metadata only",
    PAGE_TEXT: "Unverified page text (not confirmed as abstract/full text)",
    SNIPPET_ONLY: "Search snippet only (weak evidence)",
}

MIN_ABSTRACT_CHARS = 200
MIN_FULLTEXT_CHARS = 6000
MAX_TEXT_CHARS = 200_000


class ContentExtractor:

    def __init__(self, session=None, timeout=15,
                 max_pdf_bytes=25 * 1024 * 1024,
                 cache_dir=CONTENT_CACHE_DIR, use_cache=True,
                 respect_robots=True, max_workers=4):
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.timeout = timeout
        self.max_pdf_bytes = max_pdf_bytes
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache and self.cache_dir is not None
        self.respect_robots = respect_robots
        self.max_workers = max_workers
        self._robots = {}
        self._pdf = PDFProcessor()
        self._sections = SectionExtractor()

    # ------------------------------------------------------------------
    def extract_many(self, search_results):
        """Extract content for many search results (in parallel)."""
        if not search_results:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self.extract, search_results))

    def extract(self, search_result):
        """Extract content for one search result dict."""
        url = search_result.get("url", "")
        key = search_result.get("normalized_url") or normalize_url(url)

        cached = self._load_cache(key)
        if cached:
            cached["from_cache"] = True
            # keep search-time info (query, score) from the current run
            return {**cached, **self._search_fields(search_result)}

        record = self._empty_record(search_result, key)

        try:
            self._extract_into(record, url, key)
        except Exception as error:   # never let one source break the run
            record["errors"].append(f"Unexpected error: {error}")

        self._finalise(record, search_result)
        self._save_cache(key, record)
        return record

    # ------------------------------------------------------------------
    # Core extraction logic
    # ------------------------------------------------------------------
    def _extract_into(self, record, url, key):
        if key.startswith("arxiv:"):
            arxiv_id = key.split(":", 1)[1]
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            self._extract_html(record, abs_url, try_pdf=False)
            self._extract_pdf(record, pdf_url)
            return

        response = self._get(record, url)
        if response is None:
            return

        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" in content_type or response.content[:5] == b"%PDF-":
            self._parse_pdf_bytes(record, response.content, url)
        elif "html" in content_type or "xml" in content_type \
                or not content_type:
            self._parse_html(record, response.text, response.url or url,
                             try_pdf=True)
        else:
            record["errors"].append(
                f"Unsupported content type '{content_type}' at {url}"
            )

    def _extract_html(self, record, url, try_pdf):
        response = self._get(record, url)
        if response is not None:
            self._parse_html(record, response.text, response.url or url,
                             try_pdf=try_pdf)

    def _extract_pdf(self, record, url):
        if record["full_text"]:
            return
        response = self._get(record, url)
        if response is None:
            return
        if response.content[:5] != b"%PDF-":
            record["errors"].append(f"Expected a PDF but got another "
                                    f"content type at {url}")
            return
        self._parse_pdf_bytes(record, response.content, url)

    def _parse_pdf_bytes(self, record, data, url):
        if len(data) > self.max_pdf_bytes:
            record["errors"].append(f"PDF too large to process: {url}")
            return
        try:
            paper = self._pdf.extract_text(pdf_bytes=data,
                                           file_name=url.rsplit("/", 1)[-1])
        except Exception as error:
            record["errors"].append(f"PDF extraction failed ({url}): "
                                    f"{error}")
            return

        text = paper["full_text"]
        if len(text) < MIN_FULLTEXT_CHARS:
            record["errors"].append(f"PDF text too short to be a full "
                                    f"paper ({len(text)} chars): {url}")
            return

        record["full_text"] = text[:MAX_TEXT_CHARS]
        record["full_text_url"] = url
        record["methods"].append("pdf")
        sections = self._sections.extract_sections(text)
        record["sections"] = {k: v for k, v in sections.items() if v}
        if not record["abstract"] and sections.get("abstract"):
            record["abstract"] = sections["abstract"][:3000]
        if not record["title"] and paper.get("title"):
            record["title"] = paper["title"]

    def _parse_html(self, record, html, url, try_pdf):
        soup = BeautifulSoup(html, "html.parser")
        meta = self._read_meta(soup)

        for field in ("title", "authors", "published", "venue", "doi"):
            if meta.get(field) and not record[field]:
                record[field] = meta[field]

        abstract = meta.get("abstract") or self._find_abstract_block(soup)
        if abstract and len(abstract) >= MIN_ABSTRACT_CHARS \
                and not record["abstract"]:
            record["abstract"] = abstract[:3000]
            record["methods"].append("html_abstract")

        # Full-text HTML (e.g. PMC / MDPI / arXiv HTML)
        body_text, looks_full = self._article_text(soup)
        if looks_full and not record["full_text"]:
            record["full_text"] = body_text[:MAX_TEXT_CHARS]
            record["full_text_url"] = url
            record["methods"].append("html_full_text")
            sections = self._sections.extract_sections(
                self._html_to_lines(soup)
            )
            record["sections"] = {k: v for k, v in sections.items() if v}
        elif body_text and not record["page_text"]:
            record["page_text"] = body_text[:20_000]

        if try_pdf and not record["full_text"] and meta.get("pdf_url"):
            self._extract_pdf(record, urljoin(url, meta["pdf_url"]))

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_meta(soup):
        def values(*names):
            found = []
            for name in names:
                for tag in soup.find_all("meta", attrs={"name": name}) + \
                        soup.find_all("meta", attrs={"property": name}):
                    content = (tag.get("content") or "").strip()
                    if content:
                        found.append(content)
            return found

        def first(*names):
            items = values(*names)
            return items[0] if items else None

        title = first("citation_title", "dc.title", "DC.title",
                      "og:title")
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()

        abstract = first("citation_abstract", "dc.description",
                         "DC.description", "description", "og:description")

        return {
            "title": title,
            "authors": values("citation_author", "dc.creator",
                              "DC.creator")[:30] or None,
            "published": first("citation_publication_date", "citation_date",
                               "dc.date", "DC.date",
                               "citation_online_date"),
            "venue": first("citation_journal_title",
                           "citation_conference_title",
                           "citation_publisher"),
            "doi": first("citation_doi", "dc.identifier", "DC.identifier"),
            "pdf_url": first("citation_pdf_url"),
            "abstract": re.sub(r"\s+", " ", abstract).strip()
            if abstract else None
        }

    @staticmethod
    def _find_abstract_block(soup):
        candidates = soup.select(
            "blockquote.abstract, div.abstract, section.abstract, "
            "#abstract, [class*='abstract'], [id*='abstract']"
        )
        for node in candidates:
            text = re.sub(r"\s+", " ", node.get_text(" ")).strip()
            text = re.sub(r"^abstract[:.\s]*", "", text, flags=re.I)
            if len(text) >= MIN_ABSTRACT_CHARS:
                return text
        return None

    @staticmethod
    def _article_text(soup):
        """Return (text, looks_like_full_paper)."""
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "noscript"]):
            tag.decompose()

        container = soup.find("article") or soup.find("main") or soup.body
        if container is None:
            return "", False

        paragraphs = [
            re.sub(r"\s+", " ", p.get_text(" ")).strip()
            for p in container.find_all("p")
        ]
        paragraphs = [p for p in paragraphs if len(p) > 60]
        text = "\n".join(paragraphs)

        headings = " ".join(
            h.get_text(" ").lower()
            for h in container.find_all(["h2", "h3", "h4"])
        )
        has_structure = (
            "introduction" in headings
            and any(word in headings for word in
                    ("method", "approach", "experiment", "result",
                     "conclusion", "discussion"))
        )
        return text, (len(text) >= MIN_FULLTEXT_CHARS and has_structure)

    @staticmethod
    def _html_to_lines(soup):
        lines = []
        for node in soup.find_all(["h1", "h2", "h3", "h4", "p"]):
            text = re.sub(r"\s+", " ", node.get_text(" ")).strip()
            if text:
                lines.append(text)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _get(self, record, url):
        if not self._allowed_by_robots(url):
            record["errors"].append(f"Blocked by robots.txt: {url}")
            record["blocked"] = True
            return None
        try:
            response = self.session.get(url, timeout=self.timeout,
                                        allow_redirects=True)
        except requests.Timeout:
            record["errors"].append(f"Timeout after {self.timeout}s: {url}")
            return None
        except requests.RequestException as error:
            record["errors"].append(f"Request failed ({url}): {error}")
            return None

        if response.status_code in (401, 402, 403, 429, 451):
            record["errors"].append(
                f"Access denied (HTTP {response.status_code}) - page is "
                f"blocked, rate-limited or paywalled: {url}"
            )
            record["blocked"] = True
            return None
        if response.status_code >= 400:
            record["errors"].append(f"HTTP {response.status_code}: {url}")
            return None
        return response

    def _allowed_by_robots(self, url):
        if not self.respect_robots:
            return True
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._robots:
            parser = robotparser.RobotFileParser()
            try:
                response = self.session.get(f"{base}/robots.txt",
                                            timeout=min(self.timeout, 8))
                if response.status_code in (401, 403):
                    parser.disallow_all = True
                elif response.status_code >= 400:
                    parser.allow_all = True
                else:
                    parser.parse(response.text.splitlines())
            except requests.RequestException:
                parser.allow_all = True   # unknown: let the fetch decide
            self._robots[base] = parser
        return self._robots[base].can_fetch(USER_AGENT, url)

    # ------------------------------------------------------------------
    # Record helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _search_fields(result):
        return {
            "query": result.get("query"),
            "search_score": result.get("score"),
            "snippet": result.get("snippet", ""),
        }

    def _empty_record(self, result, key):
        return {
            "url": result.get("url", ""),
            "normalized_url": key,
            "search_title": result.get("title"),
            "title": None,
            "authors": None,
            "published": None,
            "venue": None,
            "doi": None,
            "abstract": None,
            "full_text": None,
            "full_text_url": None,
            "page_text": None,
            "sections": {},
            "methods": [],
            "errors": [],
            "blocked": False,
            "from_cache": False,
            **self._search_fields(result),
        }

    @staticmethod
    def _finalise(record, result):
        if record["full_text"]:
            evidence = FULL_TEXT
            text = record["full_text"]
        elif record["abstract"]:
            evidence = ABSTRACT_ONLY
            text = record["abstract"]
        elif record["page_text"] and len(record["page_text"]) > 500:
            evidence = PAGE_TEXT
            text = record["page_text"]
        else:
            evidence = SNIPPET_ONLY
            text = record["snippet"] or ""

        record["evidence_level"] = evidence
        record["evidence_description"] = EVIDENCE_DESCRIPTIONS[evidence]
        record["content"] = text
        record["char_count"] = len(text)
        record["status"] = "success" if evidence != SNIPPET_ONLY \
            else "failed"
        # Fall back to the search title only when the page had none.
        record["title"] = record["title"] or result.get("title") \
            or record["url"]

    def _cache_path(self, key):
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def _load_cache(self, key):
        if not self.use_cache or not key:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _save_cache(self, key, record):
        # Do not cache failures: the page may be reachable next time.
        if not self.use_cache or not key or record["status"] != "success":
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(key).write_text(json.dumps(record),
                                             encoding="utf-8")
        except OSError:
            pass
