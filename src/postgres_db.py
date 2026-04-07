"""Postgres + pgvector: document and chunk storage with DB-native similarity search."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Iterable

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.config import DATABASE_URL, EMBEDDING_DIM, ensure_dirs


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    document_name: Mapped[str] = mapped_column(String(512), nullable=False)
    company_name: Mapped[str] = mapped_column(String(256), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    client_label: Mapped[str] = mapped_column(String(128), nullable=False, default="default", index=True)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ChunkRecord(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    document_name: Mapped[str] = mapped_column(String(512), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    company_name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    chart_note: Mapped[str | None] = mapped_column(String(512), nullable=True)


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    ensure_dirs()
    if _engine is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is required for Postgres backend")
        _engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True, future=True)
        with _engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        Base.metadata.create_all(_engine)
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autoflush=False, autocommit=False, future=True
        )
    return _SessionLocal


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def add_document(
    stored_path: str,
    document_name: str,
    company_name: str,
    version: str,
    page_count: int,
    client_label: str = "default",
) -> int:
    with session_scope() as s:
        doc = DocumentRecord(
            stored_path=stored_path,
            document_name=document_name,
            company_name=company_name,
            version=version,
            page_count=page_count,
            client_label=client_label.strip() or "default",
        )
        s.add(doc)
        s.flush()
        return int(doc.id)


def insert_chunks_with_embeddings(
    document_id: int,
    rows: Iterable[dict[str, Any]],
    vectors: np.ndarray,
) -> None:
    rows = list(rows)
    if len(rows) != len(vectors):
        raise ValueError("rows and vectors length mismatch")
    with session_scope() as s:
        for row, vec in zip(rows, vectors):
            emb = vec.astype(float).tolist()
            rec = ChunkRecord(
                document_id=document_id,
                document_name=row["document_name"],
                page_number=row["page_number"],
                company_name=row["company_name"],
                version=row["version"],
                chunk_text=row["chunk_text"],
                embedding=emb,
                chart_note=row.get("chart_note"),
            )
            s.add(rec)


def vector_search(
    query_embedding: np.ndarray,
    k: int,
    client_label: str | None = None,
) -> list[tuple[float, int]]:
    """
    Cosine similarity via pgvector: return (score, chunk_id) with higher score = better match.
    """
    q = query_embedding.astype(float).flatten().tolist()
    dist_col = ChunkRecord.embedding.cosine_distance(q)
    stmt = (
        select(ChunkRecord.id, dist_col.label("dist"))
        .join(DocumentRecord, ChunkRecord.document_id == DocumentRecord.id)
        .order_by(dist_col)
        .limit(k)
    )
    if client_label is not None and client_label.strip():
        stmt = stmt.where(DocumentRecord.client_label == client_label.strip())

    with session_scope() as s:
        rows = s.execute(stmt).all()

    out: list[tuple[float, int]] = []
    for chunk_id, dist in rows:
        d = float(dist) if dist is not None else 1.0
        sim = max(0.0, 1.0 - d)
        out.append((sim, int(chunk_id)))
    return out


def get_chunks_by_ids(ids: list[int]) -> dict[int, dict[str, Any]]:
    if not ids:
        return {}
    with session_scope() as s:
        rows = s.execute(select(ChunkRecord).where(ChunkRecord.id.in_(ids))).scalars().all()
        return {r.id: _chunk_to_dict(r) for r in rows}


def list_documents(client_label: str | None = None) -> list[dict[str, Any]]:
    with session_scope() as s:
        stmt = select(DocumentRecord).order_by(DocumentRecord.id.desc())
        if client_label is not None and client_label.strip():
            stmt = stmt.where(DocumentRecord.client_label == client_label.strip())
        rows = s.execute(stmt).scalars().all()
        return [
            {
                "id": r.id,
                "document_name": r.document_name,
                "company_name": r.company_name,
                "version": r.version,
                "page_count": r.page_count,
                "client_label": r.client_label,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


def _chunk_to_dict(r: ChunkRecord) -> dict[str, Any]:
    return {
        "id": r.id,
        "document_id": r.document_id,
        "document_name": r.document_name,
        "page_number": r.page_number,
        "company_name": r.company_name,
        "version": r.version,
        "chunk_text": r.chunk_text,
        "faiss_index": r.id,
        "chart_note": r.chart_note,
    }
