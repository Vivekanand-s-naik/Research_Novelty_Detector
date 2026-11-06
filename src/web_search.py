"""
Module 4: Web Search (Tavily)

* Uses the Tavily Python SDK with search_depth="basic" (1 credit/query).
* Limits number of queries and results per query.
* Restricts results to scholarly domains by default.
* Deduplicates results with normalised URLs (arXiv abs/pdf/versions
  are treated as the same paper).
* Caches responses on disk so re-running the same analysis does not
  spend API credits again.

Search results only contain a short snippet. They are NOT treated as
paper content; the content extractor fetches the actual page later.
"""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.config import (MAX_QUERIES, RESULTS_PER_QUERY, SEARCH_CACHE_DIR,
                        require_env)

ACADEMIC_DOMAINS = [
    "arxiv.org",
    "semanticscholar.org",
    "aclanthology.org",
    "openreview.net",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "link.springer.com",
    "sciencedirect.com",
    "nature.com",
    "mdpi.com",
    "ncbi.nlm.nih.gov",
    "researchgate.net",
    "frontiersin.org",
    "plos.org",
    "biorxiv.org",
    "papers.nips.cc",
    "proceedings.mlr.press",
    "openaccess.thecvf.com",
]

TRACKING_PARAMS = {"ref", "source", "fbclid", "gclid", "via", "sessionid"}
ARXIV_ID = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/(?P<id>\d{4}\.\d{4,5}|[a-z\-]+/\d{7})"
    r"(?:v\d+)?(?:\.pdf)?",
    re.IGNORECASE
)


class SearchError(RuntimeError):
    """Raised when the search provider fails."""


def normalize_url(url):
    """Return a canonical key for a URL, used for deduplication."""
    if not url:
        return ""
    url = url.strip()

    arxiv = ARXIV_ID.search(url)
    if arxiv:
        return f"arxiv:{arxiv.group('id').lower()}"

    parts = urlsplit(url if "://" in url else f"https://{url}")
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host.endswith(":80") or host.endswith(":443"):
        host = host.rsplit(":", 1)[0]

    path = re.sub(r"/+", "/", parts.path).rstrip("/")
    if host.endswith("doi.org"):
        path = path.lower()  # DOIs are case-insensitive

    query = urlencode(sorted(
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_")
        and k.lower() not in TRACKING_PARAMS
    ))
    # scheme is ignored on purpose: http and https are the same page
    return urlunsplit(("", host, path, query, "")).lstrip("/")


class WebSearcher:

    def __init__(self, client=None, api_key=None,
                 max_queries=MAX_QUERIES,
                 results_per_query=RESULTS_PER_QUERY,
                 include_domains=None,
                 cache_dir=SEARCH_CACHE_DIR,
                 use_cache=True):
        """
        client: optional object with a Tavily-like `search()` method
                (used for tests). If omitted, a TavilyClient is created
                lazily using TAVILY_API_KEY.
        """
        self._client = client
        self._api_key = api_key
        self.max_queries = max_queries
        self.results_per_query = max(1, min(results_per_query, 10))
        self.include_domains = (ACADEMIC_DOMAINS if include_domains is None
                                else include_domains)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache and self.cache_dir is not None
        self.api_calls = 0

    @property
    def client(self):
        if self._client is None:
            from tavily import TavilyClient
            api_key = self._api_key or require_env("TAVILY_API_KEY")
            self._client = TavilyClient(api_key=api_key)
        return self._client

    # ------------------------------------------------------------------
    def search(self, queries):
        """
        Run up to `max_queries` searches and return:
            {"results": [...deduplicated...], "errors": [...],
             "api_calls": int, "queries_run": [...]}
        Each result: title, url, normalized_url, snippet, score,
        query (originating), also_found_by (other queries), domain.
        """
        query_strings = []
        for query in queries or []:
            text = query["query"] if isinstance(query, dict) else query
            text = (text or "").strip()
            if text and text not in query_strings:
                query_strings.append(text)
        query_strings = query_strings[:self.max_queries]

        results_by_key = {}
        errors = []

        for query in query_strings:
            try:
                response = self._search_one(query)
            except Exception as error:
                errors.append({"query": query, "error": str(error)})
                continue

            for item in response.get("results", []):
                url = (item.get("url") or "").strip()
                if not url:
                    continue
                key = normalize_url(url)
                score = item.get("score")

                if key in results_by_key:
                    existing = results_by_key[key]
                    if query != existing["query"] and \
                            query not in existing["also_found_by"]:
                        existing["also_found_by"].append(query)
                    if score is not None and \
                            (existing["score"] or 0) < score:
                        existing["score"] = score
                    continue

                results_by_key[key] = {
                    "title": (item.get("title") or "").strip() or None,
                    "url": url,
                    "normalized_url": key,
                    "snippet": (item.get("content") or "").strip(),
                    "score": score,
                    "query": query,
                    "also_found_by": [],
                    "domain": urlsplit(url).netloc.lower().removeprefix(
                        "www.")
                }

        results = sorted(results_by_key.values(),
                         key=lambda r: r["score"] or 0, reverse=True)

        return {
            "results": results,
            "errors": errors,
            "api_calls": self.api_calls,
            "queries_run": query_strings
        }

    # ------------------------------------------------------------------
    def _search_one(self, query):
        params = {
            "query": query,
            "search_depth": "basic",
            "max_results": self.results_per_query,
        }
        if self.include_domains:
            params["include_domains"] = self.include_domains

        cache_file = self._cache_file(params)
        if self.use_cache and cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass  # corrupt cache: fall through to a fresh request

        try:
            response = self.client.search(**params)
        except Exception as error:
            raise SearchError(f"Tavily search failed: {error}") from error
        self.api_calls += 1

        if self.use_cache:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(response, indent=2),
                                      encoding="utf-8")
            except (OSError, TypeError):
                pass
        return response

    def _cache_file(self, params):
        digest = hashlib.sha256(
            json.dumps(params, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        return (self.cache_dir or Path(".")) / f"{digest}.json"
