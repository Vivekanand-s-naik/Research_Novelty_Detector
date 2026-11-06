import numpy as np

from src.embedding_manager import EmbeddingManager, chunk_text, get_model
from src.similarity_engine import (SimilarityEngine, describe_score,
                                   title_similarity)

UPLOADED = {"title": "Graph Neural Networks for Molecule Property Prediction",
            "full_text": "..."}
SECTIONS = {
    "abstract": "graph neural networks predict molecule properties using "
                "geometry and attention",
    "methodology": "equivariant message passing graph network attention "
                   "layers molecules",
    "contributions": "",
    "objectives": "",
    "problem_statement": "",
}


def record(title, content, evidence="abstract_only", abstract=None):
    return {"title": title, "url": f"https://x.org/{title[:5]}",
            "content": content, "abstract": abstract or content,
            "evidence_level": evidence,
            "evidence_description": evidence}


# --- Embedding manager ----------------------------------------------------
def test_embed_empty_text_returns_none(fake_embedder):
    assert fake_embedder.embed_text("") is None
    assert fake_embedder.embed_text("   ") is None
    assert fake_embedder.embed_document("") is None
    assert fake_embedder.embed_chunks("") == ([], None)


def test_embed_texts_mixed(fake_embedder):
    vectors = fake_embedder.embed_texts(["hello world", "", None])
    assert vectors[0] is not None and vectors[1] is None and vectors[2] is None
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0)


def test_document_embedding_is_normalised(fake_embedder):
    vector = fake_embedder.embed_document("word " * 1000)
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_chunk_text():
    assert chunk_text("") == []
    assert chunk_text("a b c") == ["a b c"]
    chunks = chunk_text(" ".join(str(i) for i in range(500)),
                        chunk_words=100, overlap_words=20)
    assert len(chunks) > 4
    assert all(len(c.split()) <= 100 for c in chunks)


def test_model_loaded_once(monkeypatch):
    import src.embedding_manager as em
    loads = []

    class Dummy:
        def __init__(self, name):
            loads.append(name)

    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Dummy)
    monkeypatch.setattr(em, "_MODEL_CACHE", {})
    first = get_model("dummy-model")
    second = get_model("dummy-model")
    assert first is second
    assert loads == ["dummy-model"]
    EmbeddingManager(model_name="dummy-model").model  # reuses cache
    assert loads == ["dummy-model"]


# --- Similarity engine ----------------------------------------------------
def test_ranking_orders_by_similarity(fake_embedder):
    engine = SimilarityEngine(fake_embedder)
    related = record("Molecular GNN", "graph neural networks predict "
                     "molecule properties with attention")
    unrelated = record("Medieval poetry", "a study of medieval french "
                       "poetry and rhyme schemes")
    ranked = engine.rank(UPLOADED, SECTIONS, [unrelated, related])

    assert ranked[0]["title"] == "Molecular GNN"
    assert ranked[0]["rank"] == 1 and ranked[1]["rank"] == 2
    assert ranked[0]["overall_similarity"] > ranked[1]["overall_similarity"]
    assert ranked[0]["explanation"]
    assert "basis" in ranked[0]


def test_aspect_scores_and_missing_sections(fake_embedder):
    engine = SimilarityEngine(fake_embedder)
    ranked = engine.rank(UPLOADED, SECTIONS, [record(
        "Molecular GNN", "equivariant message passing graph network for "
                         "molecules")])
    aspects = ranked[0]["aspects"]

    assert aspects["methodology"]["score"] is not None
    assert aspects["methodology"]["evidence_passage"]
    # no objectives/contributions/conclusion in uploaded paper
    assert aspects["objectives"]["score"] is None
    assert "No objectives" in aspects["objectives"]["explanation"]


def test_same_paper_is_flagged_and_unranked(fake_embedder):
    engine = SimilarityEngine(fake_embedder)
    itself = record("Graph Neural Networks for Molecule Property Prediction",
                    "graph neural networks predict molecule properties")
    other = record("Other GNN", "graph networks for proteins")
    ranked = engine.rank(UPLOADED, SECTIONS, [itself, other])

    flagged = [r for r in ranked if r["likely_same_paper"]]
    assert len(flagged) == 1 and flagged[0]["rank"] is None
    assert [r["rank"] for r in ranked if not r["likely_same_paper"]] == [1]


def test_empty_retrieved_content(fake_embedder):
    engine = SimilarityEngine(fake_embedder)
    ranked = engine.rank(UPLOADED, SECTIONS, [record("", "", abstract="")])
    assert ranked[0]["aspects"]["methodology"]["score"] is None


def test_empty_inputs(fake_embedder):
    engine = SimilarityEngine(fake_embedder)
    assert engine.rank(UPLOADED, SECTIONS, []) == []
    ranked = engine.rank({}, {}, [record("X", "some text")])
    assert ranked[0]["overall_similarity"] is not None or \
        ranked[0]["band"] == "n/a"


def test_describe_score_never_mentions_novelty():
    for score in (None, -0.1, 0.2, 0.4, 0.6, 0.9):
        band, text = describe_score(score)
        assert "novel" not in text.lower()


def test_title_similarity():
    assert title_similarity("A Study of X", "a study of x.") > 0.95
    assert title_similarity("", "anything") == 0.0

