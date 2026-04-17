"""SQLite persistence for chunks and document metadata."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import re
from typing import Any, Generator, Iterable

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, func, or_, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.config import DB_PATH, ensure_dirs
from src.metrics_heuristic import rank_metric_query_matches


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    document_name: Mapped[str] = mapped_column(String(512), nullable=False)
    company_name: Mapped[str] = mapped_column(String(256), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    document_year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    document_month: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
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
    document_year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    document_month: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    faiss_index: Mapped[int] = mapped_column(Integer, unique=True, nullable=False, index=True)
    chart_note: Mapped[str | None] = mapped_column(String(512), nullable=True)
    is_structured: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)
    structured_type: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    source_type: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    ocr_low_confidence: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)


class ExtractedMetricRecord(Base):
    __tablename__ = "extracted_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    document_name: Mapped[str] = mapped_column(String(512), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    company_name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    document_year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    document_month: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    metric_name: Mapped[str] = mapped_column(String(256), nullable=False)
    value: Mapped[str] = mapped_column(String(256), nullable=False)
    normalized_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[str] = mapped_column(String(16), nullable=False, default="medium")
    source_type: Mapped[str] = mapped_column(String(16), nullable=False, default="text")


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    ensure_dirs()
    if _engine is None:
        _engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
        Base.metadata.create_all(_engine)
        _ensure_compat_columns(_engine)
    return _engine


def _ensure_compat_columns(engine) -> None:
    with engine.connect() as conn:
        doc_cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(documents)").fetchall()}
        if "document_year" not in doc_cols:
            conn.exec_driver_sql("ALTER TABLE documents ADD COLUMN document_year INTEGER")
        if "document_month" not in doc_cols:
            conn.exec_driver_sql("ALTER TABLE documents ADD COLUMN document_month INTEGER")

        chunk_cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(chunks)").fetchall()}
        if "document_year" not in chunk_cols:
            conn.exec_driver_sql("ALTER TABLE chunks ADD COLUMN document_year INTEGER")
        if "document_month" not in chunk_cols:
            conn.exec_driver_sql("ALTER TABLE chunks ADD COLUMN document_month INTEGER")
        if "is_structured" not in chunk_cols:
            conn.exec_driver_sql("ALTER TABLE chunks ADD COLUMN is_structured INTEGER NOT NULL DEFAULT 0")
        if "structured_type" not in chunk_cols:
            conn.exec_driver_sql("ALTER TABLE chunks ADD COLUMN structured_type VARCHAR(32)")
        if "source_type" not in chunk_cols:
            conn.exec_driver_sql("ALTER TABLE chunks ADD COLUMN source_type VARCHAR(16)")
        if "ocr_low_confidence" not in chunk_cols:
            conn.exec_driver_sql("ALTER TABLE chunks ADD COLUMN ocr_low_confidence INTEGER NOT NULL DEFAULT 0")

        em_cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(extracted_metrics)").fetchall()}
        if em_cols and "normalized_value" not in em_cols:
            conn.exec_driver_sql("ALTER TABLE extracted_metrics ADD COLUMN normalized_value REAL")
        conn.commit()


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
    document_year: int | None = None,
    document_month: int | None = None,
) -> int:
    with session_scope() as s:
        doc = DocumentRecord(
            stored_path=stored_path,
            document_name=document_name,
            company_name=company_name,
            version=version,
            page_count=page_count,
            client_label=(client_label or "default").strip() or "default",
            document_year=document_year,
            document_month=document_month,
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
                document_year=row.get("document_year"),
                document_month=row.get("document_month"),
                chunk_text=row["chunk_text"],
                faiss_index=idx,
                chart_note=row.get("chart_note"),
                is_structured=1 if row.get("is_structured") else 0,
                structured_type=row.get("structured_type"),
                source_type=row.get("source_type"),
                ocr_low_confidence=1 if row.get("ocr_low_confidence") else 0,
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
                "document_year": r.document_year,
                "document_month": r.document_month,
                "page_count": r.page_count,
                "client_label": r.client_label,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


def get_document_stored_path(document_name: str, client_label: str | None = None) -> str | None:
    """Filesystem path saved at ingest (for rendering PDF pages in the UI)."""
    with session_scope() as s:
        stmt = select(DocumentRecord.stored_path).where(DocumentRecord.document_name == document_name)
        if client_label is not None and client_label.strip():
            stmt = stmt.where(DocumentRecord.client_label == client_label.strip())
        stmt = stmt.order_by(DocumentRecord.id.desc()).limit(1)
        return s.execute(stmt).scalar_one_or_none()


def list_chart_chunks(client_label: str | None = None, limit: int = 1200) -> list[dict[str, Any]]:
    """Return chart/OCR-like chunks for optional chart-finder mode in the UI."""
    with session_scope() as s:
        stmt = (
            select(ChunkRecord)
            .join(DocumentRecord, ChunkRecord.document_id == DocumentRecord.id)
            .where(
                or_(
                    ChunkRecord.structured_type == "chart",
                    ChunkRecord.source_type == "ocr",
                    ChunkRecord.chart_note.is_not(None),
                    ChunkRecord.ocr_low_confidence == 1,
                )
            )
            .order_by(
                ChunkRecord.document_year.desc(),
                ChunkRecord.document_month.desc(),
                ChunkRecord.id.desc(),
            )
            .limit(limit)
        )
        if client_label is not None and client_label.strip():
            stmt = stmt.where(DocumentRecord.client_label == client_label.strip())
        rows = s.execute(stmt).scalars().all()
        return [_chunk_to_dict(r) for r in rows]


def has_document_version(document_name: str, version: str, client_label: str | None = None) -> bool:
    with session_scope() as s:
        stmt = select(func.count(DocumentRecord.id)).where(
            DocumentRecord.document_name == document_name,
            DocumentRecord.version == version,
        )
        if client_label and client_label.strip():
            stmt = stmt.where(DocumentRecord.client_label == client_label.strip())
        c = s.execute(stmt).scalar() or 0
        return int(c) > 0


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
        "document_year": r.document_year,
        "document_month": r.document_month,
        "chunk_text": r.chunk_text,
        "faiss_index": r.faiss_index,
        "chart_note": r.chart_note,
        "is_structured": bool(r.is_structured),
        "structured_type": r.structured_type,
        "source_type": r.source_type,
        "ocr_low_confidence": bool(r.ocr_low_confidence),
    }


def insert_extracted_metrics(document_id: int, document_name: str, rows: Iterable[dict[str, Any]]) -> None:
    if not rows:
        return
    with session_scope() as s:
        for row in rows:
            s.add(
                ExtractedMetricRecord(
                    document_id=document_id,
                    document_name=document_name,
                    page_number=int(row["page_number"]),
                    company_name=row["company_name"],
                    version=row["version"],
                    document_year=row.get("document_year"),
                    document_month=row.get("document_month"),
                    metric_name=row["metric_name"],
                    value=row["value"],
                    normalized_value=row.get("normalized_value"),
                    unit=row.get("unit"),
                    confidence=row.get("confidence") or "medium",
                    source_type=row.get("source_type") or "text",
                )
            )


def _metric_row_to_dict(r: ExtractedMetricRecord, client_label: str) -> dict[str, Any]:
    return {
        "id": r.id,
        "document_id": r.document_id,
        "document_name": r.document_name,
        "page_number": r.page_number,
        "company_name": r.company_name,
        "version": r.version,
        "document_year": r.document_year,
        "document_month": r.document_month,
        "metric_name": r.metric_name,
        "value": r.value,
        "original_value": r.value,
        "normalized_value": r.normalized_value,
        "unit": r.unit,
        "confidence": r.confidence,
        "source_type": r.source_type,
        "client_label": client_label,
    }


def match_metrics_for_query(
    query: str,
    client_label: str | None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Token-match query words against metric names; prefer confidence, source, recency; scope by client."""
    q_raw = (query or "").lower()
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "what",
        "which",
        "how",
        "when",
        "where",
        "did",
        "does",
        "for",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "per",
    }
    tokens = [t for t in re.split(r"\W+", q_raw) if len(t) > 2 and t not in stop]

    with session_scope() as s:
        stmt = (
            select(ExtractedMetricRecord, DocumentRecord.client_label)
            .join(DocumentRecord, ExtractedMetricRecord.document_id == DocumentRecord.id)
            .order_by(ExtractedMetricRecord.id.desc())
        )
        if client_label and client_label.strip():
            stmt = stmt.where(DocumentRecord.client_label == client_label.strip())
        rows = list(s.execute(stmt).all())
        ranked = rank_metric_query_matches(rows, tokens, limit)
        return [_metric_row_to_dict(em, clab) for em, clab in ranked]


def list_metrics_for_client(client_label: str | None, limit: int = 500) -> list[dict[str, Any]]:
    with session_scope() as s:
        stmt = (
            select(ExtractedMetricRecord, DocumentRecord.client_label)
            .join(DocumentRecord, ExtractedMetricRecord.document_id == DocumentRecord.id)
            .order_by(
                ExtractedMetricRecord.company_name,
                ExtractedMetricRecord.document_year.desc(),
                ExtractedMetricRecord.document_month.desc(),
                ExtractedMetricRecord.id.desc(),
            )
            .limit(limit)
        )
        if client_label and client_label.strip():
            stmt = stmt.where(DocumentRecord.client_label == client_label.strip())
        rows = s.execute(stmt).all()
        return [_metric_row_to_dict(em, clab) for em, clab in rows]
