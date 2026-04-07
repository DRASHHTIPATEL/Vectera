"""FAISS index persisted alongside SQLite chunk rows."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.config import FAISS_META_PATH, FAISS_PATH, ensure_dirs


def _dim() -> int:
    # all-MiniLM-L6-v2 → 384; read from file if exists
    if FAISS_META_PATH.exists():
        meta = json.loads(FAISS_META_PATH.read_text(encoding="utf-8"))
        return int(meta.get("dim", 384))
    return 384


def load_index() -> faiss.Index | None:
    ensure_dirs()
    if not FAISS_PATH.exists():
        return None
    return faiss.read_index(str(FAISS_PATH))


def save_index(index: faiss.Index, dim: int) -> None:
    ensure_dirs()
    faiss.write_index(index, str(FAISS_PATH))
    FAISS_META_PATH.write_text(json.dumps({"dim": dim}), encoding="utf-8")


def build_empty_index(dim: int) -> faiss.IndexFlatIP:
    return faiss.IndexFlatIP(dim)


def append_vectors(vectors: np.ndarray) -> faiss.Index:
    """
    Load or create index, append rows. vectors: (n, dim) float32, row-normalized.
    """
    if vectors.size == 0:
        idx = load_index()
        if idx is None:
            return build_empty_index(_dim())
        return idx
    dim = vectors.shape[1]
    idx = load_index()
    if idx is None:
        idx = build_empty_index(dim)
    elif idx.d != dim:
        raise ValueError(f"FAISS dim mismatch: index {idx.d} vs vectors {dim}")
    idx.add(vectors)
    save_index(idx, dim)
    return idx


def search(query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return distances, indices for single query vector (1, dim)."""
    idx = load_index()
    if idx is None or idx.ntotal == 0:
        return np.array([]), np.array([])
    q = query_vec.astype(np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    dists, inds = idx.search(q, min(k, idx.ntotal))
    return dists[0], inds[0]
