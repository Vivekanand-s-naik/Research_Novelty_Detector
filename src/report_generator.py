"""
Module 9: Report Generator

Builds a structured Markdown report (plus a JSON file with the raw data)
and saves both in the `outputs/` directory.
"""

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.config import OUTPUT_DIR

SECTION_LABELS = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "problem_statement": "Problem Statement",
    "objectives": "Objectives",
    "literature_review": "Literature Review / Related Work",
    "methodology": "Methodology",
    "results": "Results",
    "contributions": "Contributions",
    "conclusion": "Conclusion",
}

DISCLAIMER = (
    "> **Important:** Similarity scores are cosine similarities between "
    "sentence embeddings (all-MiniLM-L6-v2). They show how close two texts "
    "are in topic and wording. They are **not** novelty percentages and do "
    "not prove or disprove originality. The search covered only a small "
    "number of web results, so relevant prior work may be missing."
)

LLM_HEADINGS = {
    "comparison": "paper-by-paper comparison",
    "similarities": "similarities",
    "differences": "differences",
    "gaps": "potential research gaps",
    "unique": "potential unique contributions",
    "limitations": "evidence quality and limitations",
}


def _fmt(score):
    return "n/a" if score is None else f"{score:.2f}"


def _escape_table(text):
    return (text or "").replace("|", "\\|").replace("\n", " ")


def split_llm_sections(markdown):
    """Split the LLM markdown into {key: text} by '## ' headings."""
    parts = {}
    if not markdown:
        return parts
    chunks = re.split(r"^#{1,3}\s+(.+?)\s*$", markdown, flags=re.MULTILINE)
    # chunks = [preamble, heading1, body1, heading2, body2, ...]
    for heading, body in zip(chunks[1::2], chunks[2::2]):
        lower = heading.lower()
        for key, expected in LLM_HEADINGS.items():
            if lower.startswith(expected) or expected in lower:
                parts[key] = body.strip()
                break
    return parts


class ReportGenerator:

    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    def build_markdown(self, results):
        paper = results.get("paper", {})
        sections = results.get("sections", {})
        headings = results.get("headings", {})
        queries = results.get("queries", [])
        search = results.get("search", {})
        ranked = results.get("ranked", [])
        comparison = results.get("comparison") or {}
        generated = results.get("generated_at") or \
            datetime.now().isoformat(timespec="seconds")

        title = paper.get("title") or paper.get("file_name") or "Uploaded paper"
        out = [f"# Research Novelty Analysis Report", "",
               f"**Paper:** {title}  ", f"**Generated:** {generated}", "",
               DISCLAIMER, ""]

        # 1. Uploaded paper details ------------------------------------
        out += ["## 1. Uploaded Paper Details", "",
                f"- **Title (detected):** {title}",
                f"- **File:** {paper.get('file_name', 'n/a')}",
                f"- **Pages:** {paper.get('total_pages', 'n/a')}"]
        for key in ("author", "creationDate"):
            if paper.get("metadata", {}).get(key):
                out.append(f"- **PDF metadata {key}:** "
                           f"{paper['metadata'][key]}")
        for warning in paper.get("warnings", []):
            out.append(f"- ⚠️ {warning}")
        out.append("")

        # 2. Extracted sections ----------------------------------------
        out += ["## 2. Extracted Sections", ""]
        for key, label in SECTION_LABELS.items():
            text = sections.get(key) or ""
            source = "; ".join(headings.get(key, [])) or "no heading"
            if text:
                preview = text[:800] + (" ..." if len(text) > 800 else "")
                out += [f"### {label}",
                        f"*Source heading: {source} · {len(text):,} "
                        f"characters*", "", preview, ""]
            else:
                out += [f"### {label}", "*Not found in the extracted "
                        "text.*", ""]

        # 3. Queries ----------------------------------------------------
        out += ["## 3. Generated Search Queries", ""]
        if queries:
            for i, query in enumerate(queries, 1):
                phrases = ", ".join(query.get("keyphrases", [])) or "—"
                out.append(f"{i}. `{query['query']}` "
                           f"*(type: {query.get('type', 'n/a')}; "
                           f"keyphrases: {phrases})*")
        else:
            out.append("*No queries generated.*")
        out += ["", f"Tavily API calls made in this run: "
                    f"{search.get('api_calls', 0)} "
                    f"(cached queries do not cost credits).", ""]
        for error in search.get("errors", []):
            out.append(f"- ⚠️ Search error for `{error['query']}`: "
                       f"{error['error']}")
        out.append("")

        # 4. Retrieved papers -------------------------------------------
        out += ["## 4. Retrieved Related Papers", ""]
        if ranked:
            out += ["| # | Title | Source | Evidence | Found by query |",
                    "|---|---|---|---|---|"]
            for i, item in enumerate(ranked, 1):
                record = item.get("record", {})
                out.append(
                    f"| {i} | [{_escape_table(item['title'])}]"
                    f"({item['url']}) | "
                    f"{_escape_table(self._domain(item['url']))} | "
                    f"{item.get('evidence_description', '')} | "
                    f"{_escape_table(record.get('query') or '')} |"
                )
        else:
            out.append("*No related papers were retrieved.*")
        out.append("")

        # 5. Similarity ranking -----------------------------------------
        out += ["## 5. Similarity Ranking", "", DISCLAIMER, ""]
        if ranked:
            out += ["| Rank | Title | Overall | Band | Problem | Method | "
                    "Contrib. | Objectives |",
                    "|---|---|---|---|---|---|---|---|"]
            for item in ranked:
                a = item["aspects"]
                rank = item["rank"] if item["rank"] else "excluded"
                out.append(
                    f"| {rank} | {_escape_table(item['title'][:90])} | "
                    f"{_fmt(item['overall_similarity'])} | {item['band']} | "
                    f"{_fmt(a['research_problem']['score'])} | "
                    f"{_fmt(a['methodology']['score'])} | "
                    f"{_fmt(a['contributions']['score'])} | "
                    f"{_fmt(a['objectives']['score'])} |"
                )
            out += ["", "**Score explanations:**", ""]
            for item in ranked:
                out.append(f"- **{item['title'][:90]}** — "
                           f"{item['explanation']} *(Basis: "
                           f"{item['basis']}.)*")
        out.append("")

        # 6-9. LLM comparison -------------------------------------------
        llm = split_llm_sections(comparison.get("markdown", ""))
        status = comparison.get("status")
        not_run = ("*The LLM comparison was not run.*" if not comparison
                   else comparison.get("markdown")
                   if status in ("skipped", "failed")
                   else "*Not provided by the comparison.*")

        out += ["## 6. Paper-by-Paper Comparison", ""]
        if comparison.get("sources"):
            out.append("Sources compared: " + ", ".join(
                f"[{s['id']}] {s['title']}" for s in comparison["sources"]
            ))
            out.append("")
        if status == "completed" and not llm:
            out.append(comparison.get("markdown", ""))
        else:
            out.append(llm.get("comparison", not_run))
        out.append("")

        out += ["## 7. Similarities and Differences", "",
                "### Similarities", "",
                llm.get("similarities", not_run), "",
                "### Differences", "",
                llm.get("differences", not_run), ""]
        out += ["## 8. Potential Research Gaps", "",
                "*Potential gaps are derived only from the retrieved "
                "sources listed below and must be verified with a full "
                "literature review.*", "",
                llm.get("gaps", not_run), ""]
        out += ["## 9. Potential Unique Contributions", "",
                llm.get("unique", not_run), ""]

        # 10. Limitations -----------------------------------------------
        out += ["## 10. Limitations and Evidence Quality", ""]
        counts = Counter(item.get("evidence_level") for item in ranked)
        out += [
            f"- Retrieved sources: {len(ranked)} "
            f"(full text: {counts.get('full_text', 0)}, abstract only: "
            f"{counts.get('abstract_only', 0)}, unverified page text: "
            f"{counts.get('page_text', 0)}, snippet only: "
            f"{counts.get('snippet_only', 0)}).",
            "- Snippet-only sources were not sent to the LLM comparison.",
            "- Section extraction is rule-based; unusual layouts can cause "
            "missing or misassigned sections.",
            "- Only a small number of web searches were made to limit API "
            "usage, so the retrieved set is not a complete literature "
            "review.",
            "- Semantic similarity is not a measure of novelty.",
        ]
        if comparison.get("model"):
            out.append(f"- LLM comparison generated by `{comparison['model']}`"
                       " using only the extracted source text; LLM output "
                       "can still contain errors and must be verified.")
        for warning in comparison.get("warnings", []):
            out.append(f"- ⚠️ {warning}")
        for item in ranked:
            for error in item.get("record", {}).get("errors", [])[:2]:
                out.append(f"- Extraction note for *{item['title'][:70]}*: "
                           f"{error}")
        if llm.get("limitations"):
            out += ["", "**LLM assessment of evidence quality:**", "",
                    llm["limitations"]]
        out.append("")

        # 11. References ------------------------------------------------
        out += ["## 11. Source References", ""]
        ids = {s["url"]: s["id"] for s in comparison.get("sources", [])}
        for i, item in enumerate(ranked, 1):
            record = item.get("record", {})
            tag = f"[{ids[item['url']]}] " if item["url"] in ids else ""
            meta = []
            if record.get("authors"):
                authors = record["authors"]
                meta.append(", ".join(authors[:3]) +
                            (" et al." if len(authors) > 3 else ""))
            if record.get("published"):
                meta.append(str(record["published"]))
            if record.get("venue"):
                meta.append(record["venue"])
            if record.get("doi"):
                meta.append(f"DOI: {record['doi']}")
            meta_text = f" — {'; '.join(meta)}" if meta else ""
            out.append(f"{i}. {tag}{item['title']}{meta_text}. "
                       f"<{item['url']}> *({item['evidence_description']})*")
        if not ranked:
            out.append("*No sources.*")
        out.append("")

        return "\n".join(out)

    @staticmethod
    def _domain(url):
        match = re.match(r"https?://(?:www\.)?([^/]+)", url or "")
        return match.group(1) if match else ""

    # ------------------------------------------------------------------
    def save(self, results, markdown=None):
        """Save Markdown + JSON; return {"markdown": path, "json": path}."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        markdown = markdown or self.build_markdown(results)

        stem = Path(results.get("paper", {}).get("file_name")
                    or "paper").stem
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)[:50] or "paper"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = self.output_dir / f"report_{stem}_{stamp}.md"
        json_path = self.output_dir / f"report_{stem}_{stamp}.json"

        md_path.write_text(markdown, encoding="utf-8")
        json_path.write_text(
            json.dumps(self.to_serialisable(results), indent=2,
                       ensure_ascii=False),
            encoding="utf-8"
        )
        return {"markdown": md_path, "json": json_path}

    @staticmethod
    def to_serialisable(results):
        """Drop bulky fields (full texts) so the JSON stays readable."""
        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()
                        if k not in ("full_text", "pages", "page_text",
                                     "content")}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            if isinstance(value, Path):
                return str(value)
            if hasattr(value, "item"):          # numpy scalars
                return value.item()
            return value
        return clean(results)
