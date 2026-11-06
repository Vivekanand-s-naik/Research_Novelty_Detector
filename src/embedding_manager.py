"""
Module 6: Embedding Manager

Generates sentence embeddings locally with Sentence Transformers
(`all-MiniLM-L6-v2` by default: small, fast, CPU-friendly).

* The model is loaded once per process and reused (see `get_model`).
* Long documents are split into word chunks because MiniLM only reads the
  first ~256 tokens of each input. Chunk embeddings are averaged for a
  document-level vector.
* Empty text returns `None` instead of a misleading zero vector.
"""

import re
import threading

import numpy as np

from src.config import EMBEDDING_MODEL_NAME

_MODEL_CACHE = {}
_LOCK = threading.Lock()


def get_model(model_name=EMBEDDING_MODEL_NAME):
    """Load a SentenceTransformer once and reuse it (thread-safe)."""
    with _LOCK:
        if model_name not in _MODEL_CACHE:
            # Imported lazily so that importing this module stays fast.
            from sentence_transformers import SentenceTransformer
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        return _MODEL_CACHE[model_name]


def chunk_text(text, chunk_words=180, overlap_words=30, max_chunks=60):
    """Split text into overlapping word chunks."""
    if not text or not text.strip():
        return []

    words = re.sub(r"\s+", " ", text).strip().split(" ")
    if len(words) <= chunk_words:
        return [" ".join(words)]

    step = max(1, chunk_words - overlap_words)
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_words]
        if len(chunk) < 20 and chunks:   # skip tiny trailing pieces
            break
        chunks.append(" ".join(chunk))
        if len(chunks) >= max_chunks:
            break
    return chunks


class EmbeddingManager:

    def __init__(self, model=None, model_name=EMBEDDING_MODEL_NAME):
        """
        `model` can be any object with a SentenceTransformer-like
        `encode()` method (useful for tests). If omitted, the shared
        cached model is loaded on first use.
        """
        self._model = model
        self.model_name = model_name

    @property
    def model(self):
        if self._model is None:
            self._model = get_model(self.model_name)
        return self._model

    def embed_texts(self, texts):
        """
        Embed a list of texts. Returns a list with one normalised vector
        per text, or `None` for empty texts.
        """
        results = [None] * len(texts)
        valid = [(i, t.strip()) for i, t in enumerate(texts)
                 if isinstance(t, str) and t.strip()]
        if not valid:
            return results

        vectors = self.model.encode(
            [t for _, t in valid],
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        for (index, _), vector in zip(valid, vectors):
            results[index] = np.asarray(vector, dtype=np.float32)
        return results

    def embed_text(self, text):
        """Embed one short text (or None if empty)."""
        return self.embed_texts([text])[0]

    def embed_chunks(self, text, **chunk_kwargs):
        """Return (chunks, matrix) for a long text; matrix is None if empty."""
        chunks = chunk_text(text, **chunk_kwargs)
        if not chunks:
            return [], None
        vectors = self.embed_texts(chunks)
        return chunks, np.vstack(vectors)

    def embed_document(self, text, **chunk_kwargs):
        """Document-level vector: normalised mean of chunk embeddings."""
        _, matrix = self.embed_chunks(text, **chunk_kwargs)
        if matrix is None:
            return None
        mean = matrix.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else None

