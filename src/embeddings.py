"""Local sentence-transformer embeddings (no API key for vectors)."""

from __future__ import annotations

import numpy as np

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        from src.config import EMBEDDING_MODEL

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return float32 array shape (n, dim), L2-normalized for inner-product ~ cosine."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model()
    vecs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32)
