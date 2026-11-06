"""
Module 7: Similarity Engine

Ranks retrieved papers by cosine similarity to the uploaded paper and
compares individual aspects (research problem, methodology, contributions,
objectives).

IMPORTANT: cosine similarity of sentence embeddings measures *topical /
semantic closeness of text*. It does not measure originality and must not
be read as a "novelty percentage". A low score only means the retrieved
text talks about different things; a high score means the texts discuss
closely related topics and deserve a careful human/LLM comparison.
"""

import re
from difflib import SequenceMatcher

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.embedding_manager import EmbeddingManager

ASPECTS = {
    "research_problem": {
        "label": "Research problem",
        "sections": ["problem_statement", "abstract"],
        "fallback": ["introduction"],
    },
    "methodology": {
        "label": "Methodology",
        "sections": ["methodology"],
        "fallback": [],
    },
    "contributions": {
        "label": "Contributions",
        "sections": ["contributions"],
        "fallback": ["conclusion"],
    },
    "objectives": {
        "label": "Objectives",
        "sections": ["objectives"],
        "fallback": [],
    },
}

SCORE_BANDS = [
    (0.75, "high", "The texts discuss very closely related topics; "
                   "examine this paper carefully for overlap."),
    (0.55, "moderate", "The texts share a substantial part of their topic "
                       "or terminology."),
    (0.35, "low", "Some shared vocabulary or broad topic, but the focus "
                  "appears different."),
    (-1.0, "minimal", "Little semantic overlap was detected in the "
                      "available text."),
]

SAME_PAPER_TITLE_RATIO = 0.9
SAME_PAPER_SIMILARITY = 0.97


def describe_score(score):
    """Return (band, explanation) for a similarity score."""
    if score is None:
        return "n/a", "Not computed: text was missing on one side."
    for threshold, band, text in SCORE_BANDS:
        if score >= threshold:
            return band, text
    return "minimal", SCORE_BANDS[-1][2]


def _cosine(a, b):
    if a is None or b is None:
        return None
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


def _normalise_title(title):
    return re.sub(r"[^a-z0-9 ]+", " ", (title or "").lower()).split()


def title_similarity(a, b):
    a, b = " ".join(_normalise_title(a)), " ".join(_normalise_title(b))
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


class SimilarityEngine:

    def __init__(self, embedding_manager=None):
        self.embedder = embedding_manager or EmbeddingManager()

    # ------------------------------------------------------------------
    def rank(self, uploaded_paper, sections, retrieved):
        """
        uploaded_paper: dict from PDFProcessor (uses title, full_text)
        sections:       dict from SectionExtractor
        retrieved:      list of records from ContentExtractor

        Returns a list of result dicts sorted by overall similarity.
        """
        sections = sections or {}
        title = (uploaded_paper or {}).get("title") or ""

        summary_text, summary_basis = self._uploaded_summary(
            title, sections, (uploaded_paper or {}).get("full_text", ""))
        uploaded_summary = self.embedder.embed_document(summary_text)

        aspect_vectors = {}
        for name, spec in ASPECTS.items():
            text, used = self._aspect_text(sections, spec)
            aspect_vectors[name] = (
                self.embedder.embed_document(text) if text else None, used
            )

        results = []
        for record in retrieved or []:
            results.append(self._score_record(
                record, title, uploaded_summary, summary_basis,
                aspect_vectors
            ))

        results.sort(key=lambda r: (r["overall_similarity"] is not None,
                                    r["overall_similarity"] or 0),
                     reverse=True)
        rank = 0
        for result in results:
            if not result["likely_same_paper"]:
                rank += 1
                result["rank"] = rank
            else:
                result["rank"] = None
        return results

    # ------------------------------------------------------------------
    def _score_record(self, record, title, uploaded_summary, summary_basis,
                      aspect_vectors):
        retrieved_title = record.get("title") or ""
        abstract = record.get("abstract") or ""
        content = record.get("content") or ""
        evidence = record.get("evidence_level", "snippet_only")

        if abstract:
            target_text = f"{retrieved_title}. {abstract}"
            target_basis = "retrieved title + abstract"
        else:
            target_text = f"{retrieved_title}. {content[:4000]}"
            target_basis = {
                "full_text": "retrieved full text (beginning)",
                "page_text": "retrieved page text (unverified)",
            }.get(evidence, "retrieved search snippet")

        overall = _cosine(uploaded_summary,
                          self.embedder.embed_document(target_text))

        chunks, chunk_matrix = self.embedder.embed_chunks(
            content, chunk_words=120, overlap_words=20, max_chunks=80
        )

        aspects = {}
        for name, (vector, used_sections) in aspect_vectors.items():
            aspects[name] = self._aspect_score(
                name, vector, used_sections, chunks, chunk_matrix, evidence
            )

        t_sim = title_similarity(title, retrieved_title)
        same = bool(t_sim >= SAME_PAPER_TITLE_RATIO or
                    (overall is not None and
                     overall >= SAME_PAPER_SIMILARITY))

        band, explanation = describe_score(overall)
        if same:
            explanation = ("This appears to be the uploaded paper itself "
                           f"(title match {t_sim:.2f}); it is excluded "
                           "from the comparison.")

        return {
            "title": retrieved_title,
            "url": record.get("url"),
            "evidence_level": evidence,
            "evidence_description": record.get("evidence_description"),
            "overall_similarity": overall,
            "band": band,
            "explanation": explanation,
            "basis": f"uploaded {summary_basis} vs {target_basis}",
            "aspects": aspects,
            "likely_same_paper": same,
            "title_similarity": t_sim,
            "record": record,
        }

    def _aspect_score(self, name, vector, used_sections, chunks,
                      chunk_matrix, evidence):
        label = ASPECTS[name]["label"]
        if vector is None:
            return {"label": label, "score": None, "band": "n/a",
                    "explanation": f"No {label.lower()} text was found in "
                                   "the uploaded paper.",
                    "evidence_passage": None, "uploaded_sections": []}
        if chunk_matrix is None:
            return {"label": label, "score": None, "band": "n/a",
                    "explanation": "No retrieved text available.",
                    "evidence_passage": None,
                    "uploaded_sections": used_sections}

        scores = chunk_matrix @ vector     # both are L2-normalised
        best = int(np.argmax(scores))
        score = float(scores[best])
        band, _ = describe_score(score)
        scope = "full text" if evidence == "full_text" else \
            "abstract" if evidence == "abstract_only" else "limited text"
        return {
            "label": label,
            "score": score,
            "band": band,
            "explanation": (
                f"Best-matching passage in the retrieved paper's {scope} "
                f"compared with the uploaded paper's "
                f"{' + '.join(used_sections)}."
            ),
            "evidence_passage": chunks[best][:700],
            "uploaded_sections": used_sections,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _uploaded_summary(title, sections, full_text):
        abstract = sections.get("abstract") or ""
        if abstract:
            return f"{title}. {abstract}".strip(". "), "title + abstract"
        intro = sections.get("introduction") or ""
        if intro:
            return f"{title}. {intro[:3000]}".strip(". "), \
                "title + introduction"
        return f"{title}. {(full_text or '')[:4000]}".strip(". "), \
            "title + beginning of text"

    @staticmethod
    def _aspect_text(sections, spec):
        used = [s for s in spec["sections"] if sections.get(s)]
        if not used:
            used = [s for s in spec["fallback"] if sections.get(s)]
        text = " ".join(sections[s][:6000] for s in used)
        return text.strip(), used

