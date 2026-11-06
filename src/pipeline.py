"""
Pipeline orchestration shared by the CLI (main.py) and Streamlit (app.py).

Stage 1 (local, free):   PDF -> text -> sections -> search queries
Stage 2 (APIs, limited): Tavily search -> content extraction ->
                         embeddings/similarity (local) -> Groq comparison
                         -> report
"""

from datetime import datetime

from src.config import (MAX_QUERIES, RESULTS_PER_QUERY, TOP_K_COMPARE,
                        ConfigError, ensure_directories)
from src.content_extractor import ContentExtractor
from src.paper_comparator import ComparisonError, PaperComparator
from src.pdf_processor import PDFProcessor
from src.query_generator import QueryGenerator
from src.report_generator import ReportGenerator
from src.section_extractor import SectionExtractor
from src.similarity_engine import SimilarityEngine
from src.web_search import WebSearcher


def _noop(message):
    pass


def analyze_paper(pdf_path=None, pdf_bytes=None, file_name=None,
                  query_generator=None, max_queries=MAX_QUERIES):
    """Stage 1: local processing only (no API calls)."""
    paper = PDFProcessor().extract_text(pdf_path=pdf_path,
                                        pdf_bytes=pdf_bytes,
                                        file_name=file_name)
    extraction = SectionExtractor().extract(paper["full_text"])

    generator = query_generator or QueryGenerator(max_queries=max_queries)
    generator.max_queries = max_queries
    queries = generator.generate_queries(extraction["sections"],
                                         title=paper.get("title"))
    return {
        "paper": paper,
        "sections": extraction["sections"],
        "headings": extraction["headings"],
        "detected_headings": extraction["detected_headings"],
        "queries": queries,
    }


def run_research(analysis, queries=None,
                 max_queries=MAX_QUERIES,
                 results_per_query=RESULTS_PER_QUERY,
                 top_k=TOP_K_COMPARE,
                 use_llm=True,
                 tavily_api_key=None,
                 groq_api_key=None,
                 groq_model=None,
                 searcher=None, extractor=None, similarity=None,
                 comparator=None, progress=_noop, save_report=True):
    """Stage 2: search, extract, rank, compare and build the report."""
    ensure_directories()
    queries = queries if queries is not None else analysis["queries"]
    results = dict(analysis)
    results["queries"] = queries
    results["generated_at"] = datetime.now().isoformat(timespec="seconds")
    results["settings"] = {
        "max_queries": max_queries, "results_per_query": results_per_query,
        "top_k": top_k, "use_llm": use_llm,
    }
    results["errors"] = []

    progress(f"Searching Tavily with {min(len(queries), max_queries)} "
             "queries...")
    searcher = searcher or WebSearcher(api_key=tavily_api_key,
                                       max_queries=max_queries,
                                       results_per_query=results_per_query)
    search = searcher.search(queries)
    results["search"] = search
    for err in search.get("errors", []):
        results["errors"].append(f"Search ({err['query']}): {err['error']}")

    progress(f"Extracting content from {len(search['results'])} "
             "sources...")
    extractor = extractor or ContentExtractor()
    records = extractor.extract_many(search["results"])
    results["retrieved"] = records

    progress("Computing embeddings and similarity locally...")
    similarity = similarity or SimilarityEngine()
    ranked = similarity.rank(analysis["paper"], analysis["sections"],
                             records)
    results["ranked"] = ranked

    if use_llm:
        progress("Comparing top papers with Groq (1 request)...")
        comparator = comparator or PaperComparator(
            api_key=groq_api_key, model=groq_model, top_k=top_k
        )
        comparator.top_k = top_k
        try:
            results["comparison"] = comparator.compare(
                analysis["paper"], analysis["sections"], ranked)
        except (ConfigError, ComparisonError) as error:
            results["errors"].append(str(error))
            results["comparison"] = {
                "status": "failed",
                "markdown": f"**LLM comparison failed:** {error}",
                "sources": [], "model": None,
                "warnings": [str(error)], "usage": None,
            }
    else:
        results["comparison"] = None

    progress("Generating report...")
    reporter = ReportGenerator()
    results["report_markdown"] = reporter.build_markdown(results)
    if save_report:
        results["report_paths"] = reporter.save(results,
                                                results["report_markdown"])
    progress("Done.")
    return results

