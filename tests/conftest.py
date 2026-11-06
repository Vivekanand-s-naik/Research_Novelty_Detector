import sys
from pathlib import Path

import numpy as np
import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_PDF = ROOT / "data" / "uploaded_papers" / "sample.pdf"


def make_pdf(path, pages):
    """Create a small text PDF with one string per page."""
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        y = 72
        for line in text.splitlines():
            page.insert_text((72, y), line, fontsize=11)
            y += 14
    document.save(str(path))
    document.close()
    return path


class FakeEmbeddingModel:
    """
    Deterministic bag-of-words "embedding" so tests run without
    downloading Sentence Transformers. Texts that share words get
    higher cosine similarity, which is enough to test ranking logic.
    """

    dim = 256

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        vectors = []
        for text in texts:
            vector = np.zeros(self.dim, dtype=np.float32)
            for word in text.lower().split():
                word = "".join(ch for ch in word if ch.isalnum())
                if word:
                    vector[hash(word) % self.dim] += 1.0
            norm = np.linalg.norm(vector)
            if normalize_embeddings and norm > 0:
                vector /= norm
            vectors.append(vector)
        return np.vstack(vectors)


@pytest.fixture
def fake_embedder():
    from src.embedding_manager import EmbeddingManager
    return EmbeddingManager(model=FakeEmbeddingModel())


@pytest.fixture
def sample_pdf():
    if not SAMPLE_PDF.exists():
        pytest.skip("sample.pdf not available")
    return SAMPLE_PDF

