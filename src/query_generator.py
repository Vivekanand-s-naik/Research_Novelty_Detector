"""
Module 3: Query Generator

Builds a small number of focused academic search queries locally.

Strategy (no LLM, no API calls):
  1. Title query  - the paper title is usually the best single query for
                    finding closely related work.
  2. Problem query - top keyphrases from abstract / problem statement /
                     introduction, extracted with KeyBERT (MMR for
                     diversity) and ranked by relevance to that text.
  3. Method query  - top keyphrases from methodology / contributions.

Keyphrases inside a query all come from the same section and are ordered
by relevance, so a query describes one coherent aspect instead of being a
bag of unrelated words. Near-duplicate queries are removed.
"""

import re

STOPWORDS = {
    "paper", "study", "work", "approach", "method", "methods", "proposed",
    "propose", "results", "result", "using", "based", "novel", "new",
    "section", "figure", "table", "et", "al", "use", "used", "show",
    "shows", "present", "presents", "we", "our", "this", "fig"
}

# Words that usually end a noun phrase in academic writing (verbs,
# vague adjectives, discourse words). Used only for candidate building.
ACADEMIC_STOPWORDS = {
    "allow", "allows", "allowing", "achieve", "achieves", "achieved",
    "provide", "provides", "provided", "providing", "generate",
    "generates", "generated", "generating", "enable", "enables",
    "enabling", "improve", "improves", "improved", "include", "includes",
    "included", "including", "obtain", "obtained", "perform", "performs",
    "performed", "consist", "consists", "require", "requires", "required",
    "make", "makes", "made", "take", "takes", "taken", "also", "however",
    "thus", "therefore", "furthermore", "moreover", "respectively",
    "various", "different", "several", "many", "much", "specific",
    "specifically", "particular", "particularly", "significant",
    "significantly", "high", "higher", "low", "lower", "good", "better",
    "best", "important", "typically", "usually", "often", "first",
    "second", "third", "finally", "recent", "recently", "existing",
    "previous", "prior", "current", "like", "e", "g", "i", "ie", "eg",
    "etc", "via", "trained", "train", "training", "evaluate", "evaluated",
    "compared", "compare", "demonstrate", "demonstrates", "demonstrated",
    "introduce", "introduces", "introduced", "develop", "developed",
    "leverage", "leverages", "leveraging", "employ", "employs", "employed",
    "utilize", "utilizes", "utilized", "conducted", "conduct", "confirmed",
    "shown", "showed", "given", "known", "able", "aim", "aims", "need",
    "needs", "way", "ways", "number", "set", "sets", "case", "cases",
    "terms", "order", "able", "et", "al", "figure", "fig", "table", "eq",
    "section", "sections", "paper", "work", "study", "approach",
}


def _normalise_phrase(phrase):
    words = re.findall(r"[a-z0-9\-]+", phrase.lower())
    # crude plural folding so "virtual agents" == "virtual agent"
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w
                    for w in words)


def _tokens(text):
    return {w for w in _normalise_phrase(text).split() if w not in STOPWORDS}


def _jaccard(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class KeyBERTExtractor:
    """
    KeyBERT ranking over phrase-like candidates.

    Plain n-grams produce fragments such as "emulate human like". Instead,
    candidates are built RAKE-style: text is split at stopwords, common
    academic verbs and punctuation, which leaves mostly noun phrases
    ("conditional variational autoencoder"). KeyBERT then ranks them
    against a document embedding that covers the whole text (MiniLM alone
    would only read the first ~256 tokens).
    """

    def __init__(self, embedding_model=None):
        from keybert import KeyBERT

        from src.embedding_manager import EmbeddingManager, get_model
        if embedding_model is None:
            embedding_model = get_model()
        self._kw_model = KeyBERT(model=embedding_model)
        self._embedder = EmbeddingManager(model=embedding_model)

    def __call__(self, text, top_n=8):
        candidates = candidate_phrases(text)
        if not candidates:
            return []
        doc_embedding = self._embedder.embed_document(text)
        if doc_embedding is None:
            return []
        return self._kw_model.extract_keywords(
            text,
            candidates=candidates,
            keyphrase_ngram_range=(1, 4),
            stop_words=None,
            use_mmr=True,
            diversity=0.5,
            top_n=top_n,
            doc_embeddings=doc_embedding.reshape(1, -1)
        )


def candidate_phrases(text, max_words=4, max_candidates=300):
    """
    Return candidate keyphrases (lowercase, CountVectorizer-compatible).

    Multi-word phrases of 2..max_words words are kept; single words are
    kept only when they are acronyms in the original text (e.g. "CVAE").
    """
    if not text:
        return []

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop = set(ENGLISH_STOP_WORDS) | STOPWORDS | ACADEMIC_STOPWORDS

    acronyms = {a.lower() for a in re.findall(r"\b[A-Z][A-Z0-9]{1,6}s?\b",
                                              text)}
    counts = {}
    # Split into fragments at punctuation (hyphens are kept inside words)
    for fragment in re.split(r"[^\w\s\-]|\s-\s|\d+(?:\.\d+)?", text):
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]*", fragment)]
        run = []
        for word in words + ["."]:          # "." flushes the last run
            lower = word.lower().strip("-")
            if lower in stop or len(lower) < 2 or word == ".":
                _add_run(run, counts, acronyms, max_words)
                run = []
            else:
                run.append(lower)

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [phrase for phrase, _ in ranked[:max_candidates]]


def _add_run(run, counts, acronyms, max_words):
    if not run:
        return
    # Match sklearn's tokenizer: hyphens split words, 1-char tokens dropped
    tokens = [t for w in run for t in w.split("-") if len(t) > 1]
    if not tokens:
        return
    if len(tokens) > max_words:
        tokens = tokens[-max_words:]   # keep the head of the noun phrase
    if len(tokens) == 1 and tokens[0] not in acronyms:
        return
    phrase = " ".join(tokens)
    counts[phrase] = counts.get(phrase, 0) + 1


class QueryGenerator:

    def __init__(self, keyword_extractor=None, max_queries=3,
                 phrases_per_query=3, max_query_words=14):
        """
        keyword_extractor: callable(text, top_n) -> [(phrase, score), ...]
                           Defaults to KeyBERT (created lazily).
        """
        self._extractor = keyword_extractor
        self.max_queries = max_queries
        self.phrases_per_query = phrases_per_query
        self.max_query_words = max_query_words

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = KeyBERTExtractor()
        return self._extractor

    # ------------------------------------------------------------------
    def generate_queries(self, sections, title=None):
        """
        Return a list of dicts: {"query", "type", "keyphrases"}.
        """
        sections = sections or {}
        candidates = []

        clean_title = self._clean_title(title)
        if clean_title:
            candidates.append({
                "query": clean_title,
                "type": "title",
                "keyphrases": []
            })

        problem_text = self._combine(sections, [
            ("abstract", 2000),
            ("problem_statement", 1500),
            ("objectives", 1000),
            ("introduction", 1500)
        ])
        method_text = self._combine(sections, [
            ("contributions", 1500),
            ("methodology", 5000)
        ])

        for query_type, text in (("research_problem", problem_text),
                                 ("methodology", method_text)):
            query = self._query_from_text(text, query_type)
            if query:
                candidates.append(query)

        # Fallback when nothing was found in sections
        if not candidates:
            all_text = " ".join(v for v in sections.values() if v)
            query = self._query_from_text(all_text[:5000], "general")
            if query:
                candidates.append(query)

        return self._deduplicate(candidates)[:self.max_queries]

    # ------------------------------------------------------------------
    def _query_from_text(self, text, query_type):
        if not text or len(text.split()) < 15:
            return None

        keywords = self.extractor(text, top_n=10)
        phrases = self._select_phrases(keywords)
        if not phrases:
            return None

        words = []
        for phrase in phrases:
            phrase_words = phrase.split()
            if len(words) + len(phrase_words) > self.max_query_words:
                break
            words.extend(phrase_words)

        return {
            "query": " ".join(words),
            "type": query_type,
            "keyphrases": phrases
        }

    def _select_phrases(self, keywords):
        """Pick top phrases that are relevant and not redundant."""
        selected = []
        for phrase, score in keywords:
            phrase = phrase.strip()
            if not phrase or score is None or score < 0.2:
                continue
            tokens = _tokens(phrase)
            if not tokens or all(t.isdigit() for t in tokens):
                continue
            # Skip phrases that mostly repeat an already selected one
            if any(len(tokens & _tokens(s)) / len(tokens) >= 0.5
                   for s in selected):
                continue
            selected.append(phrase)
            if len(selected) >= self.phrases_per_query:
                break
        return selected

    @staticmethod
    def _deduplicate(queries, threshold=0.6):
        unique = []
        for query in queries:
            if all(_jaccard(query["query"], u["query"]) < threshold
                   for u in unique):
                unique.append(query)
        return unique

    @staticmethod
    def _combine(sections, spec):
        parts = []
        for name, limit in spec:
            text = (sections.get(name) or "").strip()
            if text:
                parts.append(text[:limit])
        return " ".join(parts)

    @staticmethod
    def _clean_title(title):
        if not title:
            return None
        title = re.sub(r"\s+", " ", title).strip(" .:-")
        words = title.split()
        if len(words) < 3:
            return None
        return " ".join(words[:20])
