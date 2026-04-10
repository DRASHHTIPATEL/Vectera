"""
Unified persistence: Postgres + pgvector when DATABASE_URL is set, else SQLite + FAISS.
Assessment expects a managed database layer (Snowflake or equivalent); Postgres is the primary path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import use_postgres


def add_document(
    stored_path: str,
    document_name: str,
    company_name: str,
    version: str,
    page_count: int,
    client_label: str = "default",
) -> int:
    if use_postgres():
        from src import postgres_db as db

        return db.add_document(
            stored_path, document_name, company_name, version, page_count, client_label
        )
    from src import database as db

    return db.add_document(
        stored_path, document_name, company_name, version, page_count, client_label
    )


def insert_chunks_after_embed(
    document_id: int,
    rows: list[dict[str, Any]],
    vectors: np.ndarray,
    start_faiss_index: int,
) -> None:
    """Persist chunks and vectors. SQLite uses FAISS + faiss_index; Postgres stores vectors in-row."""
    if use_postgres():
        from src import postgres_db as db

        db.insert_chunks_with_embeddings(document_id, rows, vectors)
        return

    from src import database as db
    from src.faiss_store import append_vectors

    append_vectors(vectors)
    db.insert_chunks(document_id, rows, start_faiss_index)


def next_vector_key() -> int:
    """SQLite: next FAISS slot index. Postgres: unused (returns 0)."""
    if use_postgres():
        return 0
    from src import database as db

    return db.next_faiss_index()


def vector_similarity_search(
    query_embedding: np.ndarray,
    k: int,
    client_label: str | None = None,
) -> list[tuple[float, int]]:
    """
    Return (score, key) sorted by descending relevance.
    key is chunk primary key (Postgres) or faiss_index (SQLite).
    """
    if use_postgres():
        from src import postgres_db as db

        return db.vector_search(query_embedding, k, client_label=client_label)

    from src import database as db
    from src.faiss_store import search

    dists, inds = search(query_embedding, k)
    if inds.size == 0:
        return []
    out = [(float(d), int(i)) for d, i in zip(dists.tolist(), inds.tolist())]
    if client_label is not None and client_label.strip():
        keys = [k for _, k in out]
        meta = db.get_chunks_by_keys(keys)
        doc_ids = {m["document_id"] for m in meta.values()}
        from src.database import DocumentRecord, session_scope
        from sqlalchemy import select

        allowed_docs: set[int] = set()
        with session_scope() as s:
            rows = s.execute(select(DocumentRecord).where(DocumentRecord.id.in_(doc_ids))).scalars().all()
            for r in rows:
                if r.client_label == client_label.strip():
                    allowed_docs.add(int(r.id))
        out = [(s, key) for s, key in out if meta.get(key) and meta[key]["document_id"] in allowed_docs]
    return out


def get_chunks_by_keys(keys: list[int]) -> dict[int, dict[str, Any]]:
    if use_postgres():
        from src import postgres_db as db

        return db.get_chunks_by_ids(keys)
    from src import database as db

    return db.get_chunks_by_keys(keys)


def list_documents(client_label: str | None = None) -> list[dict[str, Any]]:
    if use_postgres():
        from src import postgres_db as db

        return db.list_documents(client_label=client_label)
    from src import database as db

    return db.list_documents(client_label=client_label)


def backend_name() -> str:
    return "postgres+pgvector" if use_postgres() else "sqlite+faiss"
