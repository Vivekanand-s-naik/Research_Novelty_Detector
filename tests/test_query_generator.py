from src.query_generator import QueryGenerator, candidate_phrases


class FakeExtractor:
    """Returns pre-defined keyphrases per call, records texts it saw."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, text, top_n=8):
        self.calls.append(text)
        return self.outputs.pop(0) if self.outputs else []


SECTIONS = {
    "abstract": "We study graph neural networks for predicting molecular "
                "properties with geometric information and attention. " * 3,
    "methodology": "An equivariant message passing network with attention "
                   "layers is trained on molecular graphs. " * 3,
    "contributions": "",
}


def test_generates_title_problem_and_method_queries():
    extractor = FakeExtractor([
        [("graph neural networks", 0.7), ("molecular property prediction",
                                          0.6), ("geometric attention", 0.5)],
        [("equivariant message passing", 0.7), ("attention layers", 0.5)],
    ])
    generator = QueryGenerator(keyword_extractor=extractor, max_queries=3)
    queries = generator.generate_queries(
        SECTIONS, title="Geometric Graph Networks for Molecules")

    assert [q["type"] for q in queries] == ["title", "research_problem",
                                           "methodology"]
    assert queries[1]["query"] == ("graph neural networks molecular "
                                   "property prediction geometric attention")
    assert "equivariant message passing" in queries[2]["query"]


def test_max_queries_respected():
    extractor = FakeExtractor([[("graph neural networks", 0.7)],
                               [("message passing", 0.7)]])
    generator = QueryGenerator(keyword_extractor=extractor, max_queries=1)
    queries = generator.generate_queries(SECTIONS, title="A Long Paper Title")
    assert len(queries) == 1


def test_redundant_phrases_and_low_scores_removed():
    extractor = FakeExtractor([[
        ("graph neural network", 0.8),
        ("graph neural networks", 0.79),    # plural duplicate
        ("molecule", 0.1),                  # low relevance
        ("molecular properties", 0.5),
    ]])
    generator = QueryGenerator(keyword_extractor=extractor)
    queries = generator.generate_queries({"abstract": SECTIONS["abstract"]})
    assert queries[0]["keyphrases"] == ["graph neural network",
                                        "molecular properties"]


def test_duplicate_queries_removed():
    same = [("graph neural networks", 0.8), ("molecular properties", 0.6)]
    extractor = FakeExtractor([list(same), list(same)])
    generator = QueryGenerator(keyword_extractor=extractor)
    queries = generator.generate_queries(SECTIONS)
    assert len(queries) == 1


def test_empty_sections_return_no_queries():
    generator = QueryGenerator(keyword_extractor=FakeExtractor([]))
    assert generator.generate_queries({}) == []
    assert generator.generate_queries({"abstract": ""}, title=None) == []


def test_candidate_phrases_are_noun_phrase_like():
    text = ("We train a conditional variational autoencoder (CVAE) to "
            "generate virtual agent behaviors. The CVAE uses motion data.")
    candidates = candidate_phrases(text)
    assert "conditional variational autoencoder" in candidates
    assert "virtual agent behaviors" in candidates
    assert "cvae" in candidates                # acronym unigram kept
    assert "train" not in candidates           # plain verbs dropped
    assert all(len(c.split()) <= 4 for c in candidates)

