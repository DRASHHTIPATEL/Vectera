"""SQLite persistence for chunks and document metadata."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Iterable

from sqlalchemy import DateTime, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.config import DB_PATH, ensure_dirs


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
    faiss_index: Mapped[int] = mapped_column(Integer, unique=True, nullable=False, index=True)
    chart_note: Mapped[str | None] = mapped_column(String(512), nullable=True)


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    ensure_dirs()
    if _engine is None:
        _engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
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
            client_label=(client_label or "default").strip() or "default",
        )
        s.add(doc)
        s.flush()
        return int(doc.id)


def insert_chunks(
    document_id: int,
    rows: Iterable[dict[str, Any]],
    start_faiss_index: int,
) -> list[int]:
    """Insert chunk rows; assigns faiss_index sequentially from start_faiss_index."""
    ids: list[int] = []
    idx = start_faiss_index
    with session_scope() as s:
        for row in rows:
            rec = ChunkRecord(
                document_id=document_id,
                document_name=row["document_name"],
                page_number=row["page_number"],
                company_name=row["company_name"],
                version=row["version"],
                chunk_text=row["chunk_text"],
                faiss_index=idx,
                chart_note=row.get("chart_note"),
            )
            s.add(rec)
            s.flush()
            ids.append(int(rec.id))
            idx += 1
    return ids


def get_chunk_by_faiss_index(faiss_idx: int) -> dict[str, Any] | None:
    with session_scope() as s:
        r = s.execute(select(ChunkRecord).where(ChunkRecord.faiss_index == faiss_idx)).scalar_one_or_none()
        if r is None:
            return None
        return _chunk_to_dict(r)


def get_chunks_by_faiss_indices(indices: list[int]) -> dict[int, dict[str, Any]]:
    if not indices:
        return {}
    with session_scope() as s:
        rows = s.execute(
            select(ChunkRecord).where(ChunkRecord.faiss_index.in_(indices))
        ).scalars().all()
        return {r.faiss_index: _chunk_to_dict(r) for r in rows}


def get_chunks_by_keys(keys: list[int]) -> dict[int, dict[str, Any]]:
    """SQLite backend keys are faiss_index values."""
    return get_chunks_by_faiss_indices(keys)


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


def next_faiss_index() -> int:
    with session_scope() as s:
        m = s.execute(select(func.max(ChunkRecord.faiss_index))).scalar()
        return int(m) + 1 if m is not None else 0


def _chunk_to_dict(r: ChunkRecord) -> dict[str, Any]:
    return {
        "id": r.id,
        "document_id": r.document_id,
        "document_name": r.document_name,
        "page_number": r.page_number,
        "company_name": r.company_name,
        "version": r.version,
        "chunk_text": r.chunk_text,
        "faiss_index": r.faiss_index,
        "chart_note": r.chart_note,
    }
