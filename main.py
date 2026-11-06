"""
Command-line entry point.

Examples:
    python main.py data/uploaded_papers/sample.pdf
    python main.py paper.pdf --max-queries 2 --results-per-query 4
    python main.py paper.pdf --local-only      # no API calls at all
    python main.py paper.pdf --no-llm          # Tavily only, skip Groq
"""

import argparse
import sys

from src.config import (MAX_QUERIES, RESULTS_PER_QUERY, TOP_K_COMPARE,
                        ConfigError)
from src.pdf_processor import PDFProcessingError
from src.pipeline import analyze_paper, run_research


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Research Novelty Detector - compare a paper with "
                    "related work found on the web."
    )
    parser.add_argument("pdf", help="Path to the research paper PDF")
    parser.add_argument("--max-queries", type=int, default=MAX_QUERIES)
    parser.add_argument("--results-per-query", type=int,
                        default=RESULTS_PER_QUERY)
    parser.add_argument("--top-k", type=int, default=TOP_K_COMPARE,
                        help="Number of papers sent to the Groq comparison")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip the Groq comparison")
    parser.add_argument("--local-only", action="store_true",
                        help="Only extract sections and queries (no APIs)")
    args = parser.parse_args(argv)

    # Windows consoles may not support all Unicode characters in papers.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    try:
        print(f"Processing {args.pdf} ...")
        analysis = analyze_paper(pdf_path=args.pdf,
                                 max_queries=args.max_queries)
    except (FileNotFoundError, ValueError, PDFProcessingError) as error:
        print(f"Error: {error}")
        return 1

    paper = analysis["paper"]
    print(f"Title: {paper.get('title')}")
    print(f"Pages: {paper['total_pages']}")
    for warning in paper["warnings"]:
        print(f"Warning: {warning}")

    print("\nSections found:")
    for name, text in analysis["sections"].items():
        source = "; ".join(analysis["headings"].get(name, []))
        status = f"{len(text):,} chars ({source})" if text else "not found"
        print(f"  - {name}: {status}")

    print("\nSearch queries:")
    for query in analysis["queries"]:
        print(f"  [{query['type']}] {query['query']}")

    if args.local_only:
        return 0

    try:
        results = run_research(
            analysis,
            max_queries=args.max_queries,
            results_per_query=args.results_per_query,
            top_k=args.top_k,
            use_llm=not args.no_llm,
            progress=lambda message: print(f"\n>> {message}"),
        )
    except ConfigError as error:
        print(f"Configuration error: {error}")
        return 1

    print("\nSimilarity ranking (topical closeness, NOT novelty):")
    for item in results["ranked"]:
        rank = item["rank"] or "excluded"
        score = item["overall_similarity"]
        score = "n/a" if score is None else f"{score:.2f}"
        print(f"  {rank}. {score}  [{item['evidence_level']}] "
              f"{item['title'][:80]}")

    for error in results["errors"]:
        print(f"Error: {error}")

    paths = results.get("report_paths", {})
    print(f"\nReport saved to: {paths.get('markdown')}")
    print(f"Raw data saved to: {paths.get('json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

