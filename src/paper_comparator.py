"""
Module 8: Paper Comparator (Groq)

The ONLY module that calls an LLM. It makes a single Groq request that
compares the uploaded paper with the top-ranked retrieved papers, using
only the text that was actually extracted from those sources.

Safeguards against unsupported claims:
  * each source is labelled with its evidence level (full text / abstract
    / page text) and the model is told to treat them accordingly;
  * the model must cite sources as [P1], [P2], ... and say
    "Insufficient evidence" when the provided text does not support a
    conclusion;
  * the LLM call is skipped entirely when no source has usable content;
  * citations in the answer are checked against the provided source IDs.
"""

import re

from src.config import TOP_K_COMPARE, get_groq_model, require_env

UPLOADED_BUDGET = {           # characters per uploaded section
    "abstract": 1500,
    "problem_statement": 700,
    "objectives": 500,
    "methodology": 1800,
    "contributions": 900,
    "conclusion": 600,
}
SOURCE_ABSTRACT_CHARS = 1200
SOURCE_PASSAGE_CHARS = 500
USABLE_EVIDENCE = ("full_text", "abstract_only", "page_text")

SYSTEM_PROMPT = """You are a careful research analyst helping a researcher \
position their paper against existing literature.

Rules you must follow:
- Use ONLY the information inside the provided UPLOADED PAPER and SOURCE \
blocks. Do not use outside knowledge about these or other papers.
- Cite every claim about a source with its ID, e.g. [P1]. Never cite IDs \
that were not provided.
- Respect evidence levels: a source with "abstract_only" or "page_text" \
evidence cannot support detailed claims about its methods or results. Say \
so explicitly.
- When the provided text is not enough to decide something, write \
"Insufficient evidence" and explain what is missing.
- Never claim the uploaded paper is guaranteed to be original or novel. \
Use cautious wording such as "potential", "appears", "based on the \
retrieved sources".
- Semantic similarity scores only indicate topical closeness; they are not \
novelty percentages. Do not convert them into novelty claims.
- Do not invent research gaps. A gap must be grounded in something the \
sources state or clearly do not address, and you must name the sources \
you checked."""

OUTPUT_FORMAT = """Write the answer in Markdown using exactly these headings:

## Paper-by-Paper Comparison
For each source ([P1], [P2], ...): research problem overlap, \
methodological similarities, existing contributions of the source, key \
differences from the uploaded paper, and an evidence note.

## Similarities
## Differences
## Potential Research Gaps
## Potential Unique Contributions of the Uploaded Paper
## Evidence Quality and Limitations

Be concise and specific. Use bullet points."""


class ComparisonError(RuntimeError):
    """Raised when the Groq comparison fails."""


class PaperComparator:

    def __init__(self, client=None, api_key=None, model=None,
                 top_k=TOP_K_COMPARE, temperature=0.2,
                 max_completion_tokens=3000):
        """
        client: optional object with a Groq-like
                `chat.completions.create()` (used for tests).
        """
        self._client = client
        self._api_key = api_key
        self.model = model or get_groq_model()
        self.top_k = top_k
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

    @property
    def client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self._api_key or
                                require_env("GROQ_API_KEY"))
        return self._client

    # ------------------------------------------------------------------
    def select_sources(self, ranked_results):
        """Top-k ranked papers with usable content (self-matches excluded)."""
        selected = [
            r for r in ranked_results or []
            if not r.get("likely_same_paper")
            and r.get("evidence_level") in USABLE_EVIDENCE
        ]
        return selected[:self.top_k]

    def compare(self, uploaded_paper, sections, ranked_results):
        """
        Returns a dict:
            status: "completed" | "skipped"
            markdown, sources (list with id/title/url/evidence),
            model, warnings, usage
        """
        sources = self.select_sources(ranked_results)
        source_list = [
            {"id": f"P{i}", "title": s["title"], "url": s["url"],
             "evidence_level": s["evidence_level"],
             "overall_similarity": s["overall_similarity"]}
            for i, s in enumerate(sources, start=1)
        ]

        if not sources:
            return {
                "status": "skipped",
                "markdown": (
                    "**Insufficient evidence.** None of the retrieved "
                    "results provided accessible paper content (full text "
                    "or abstract), so no LLM comparison was performed. "
                    "Search snippets alone are not reliable enough to "
                    "compare research contributions."
                ),
                "sources": [],
                "model": None,
                "warnings": ["No sources with usable content."],
                "usage": None,
            }

        prompt = self.build_prompt(uploaded_paper, sections, sources)
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
        }
        if "gpt-oss" in self.model:
            params["reasoning_effort"] = "low"   # keeps token usage small

        try:
            response = self.client.chat.completions.create(**params)
        except Exception as error:
            raise ComparisonError(
                f"Groq request failed ({self.model}): {error}"
            ) from error

        markdown = (response.choices[0].message.content or "").strip()
        if not markdown:
            raise ComparisonError("Groq returned an empty response.")

        usage = getattr(response, "usage", None)
        usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        } if usage else None

        return {
            "status": "completed",
            "markdown": markdown,
            "sources": source_list,
            "model": self.model,
            "warnings": self.validate_citations(markdown, source_list),
            "usage": usage,
        }

    # ------------------------------------------------------------------
    def build_prompt(self, uploaded_paper, sections, sources):
        sections = sections or {}
        title = (uploaded_paper or {}).get("title") or "Unknown title"

        lines = ["### UPLOADED PAPER", f"Title: {title}"]
        for name, limit in UPLOADED_BUDGET.items():
            text = (sections.get(name) or "").strip()
            label = name.replace("_", " ").title()
            lines.append(f"{label}: {self._clip(text, limit)}" if text
                         else f"{label}: [not found in extracted text]")

        for index, source in enumerate(sources, start=1):
            record = source.get("record", {})
            lines += [
                "",
                f"### SOURCE [P{index}]",
                f"Title: {source['title']}",
                f"URL: {source['url']}",
                f"Evidence level: {source['evidence_level']}",
            ]
            for field in ("authors", "published", "venue"):
                value = record.get(field)
                if value:
                    if isinstance(value, list):
                        value = ", ".join(value[:6]) + \
                            (" et al." if len(value) > 6 else "")
                    lines.append(f"{field.title()}: {value}")
            if source.get("overall_similarity") is not None:
                lines.append(
                    f"Semantic similarity to uploaded abstract: "
                    f"{source['overall_similarity']:.2f} (topical only)"
                )

            abstract = record.get("abstract") or ""
            if abstract:
                lines.append(
                    f"Abstract: {self._clip(abstract, SOURCE_ABSTRACT_CHARS)}"
                )

            seen = set()
            for aspect in source.get("aspects", {}).values():
                passage = aspect.get("evidence_passage")
                if not passage or passage in seen:
                    continue
                if abstract and passage[:120] in abstract:
                    continue
                seen.add(passage)
                lines.append(
                    f"Passage most similar to uploaded "
                    f"{aspect['label'].lower()}: "
                    f"{self._clip(passage, SOURCE_PASSAGE_CHARS)}"
                )

        lines += ["", "### TASK",
                  "Compare the uploaded paper with each source.",
                  OUTPUT_FORMAT]
        return "\n".join(lines)

    @staticmethod
    def validate_citations(markdown, source_list):
        valid = {s["id"] for s in source_list}
        cited = set(re.findall(r"\[(P\d+)\]", markdown))
        warnings = []
        unknown = sorted(cited - valid)
        if unknown:
            warnings.append(
                f"The model cited unknown source IDs {unknown}; treat those "
                "statements as unsupported."
            )
        if not cited:
            warnings.append("The comparison contains no source citations; "
                            "treat its claims with caution.")
        missing = sorted(valid - cited)
        if missing and cited:
            warnings.append(f"Sources never cited in the comparison: "
                            f"{missing}.")
        return warnings

    @staticmethod
    def _clip(text, limit):
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= limit:
            return text
        cut = text[:limit].rsplit(" ", 1)[0]
        return f"{cut} ..."

