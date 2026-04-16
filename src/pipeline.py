"""End-to-end: PDF → chunks → database (Postgres+pgvector or SQLite+FAISS)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.chunking import pages_to_chunks
from src.config import UPLOADS_DIR, ensure_dirs, use_postgres
from src.embeddings import embed_texts
from src.ingestion import PageContent, extract_pdf_pages, infer_metadata_from_filename, save_upload
from src.metrics_heuristic import dedupe_extracted_metrics, extract_metrics_from_text
from src.persistence import (
    add_document,
    has_document_version,
    insert_chunks_after_embed,
    insert_extracted_metrics,
    next_vector_key,
)

LOGGER = logging.getLogger(__name__)


def _metrics_from_pages(pages: list[PageContent], company_name: str, version: str, year: int | None, month: int | None) -> list[dict]:
    rows: list[dict] = []
    for p in pages:
        st = "ocr" if p.ocr_low_confidence else ("table" if p.structured_type == "table" else "text")
        rows.extend(
            extract_metrics_from_text(
                p.text,
                page_number=p.page_number,
                company_name=company_name,
                version=version,
                document_year=year,
                document_month=month,
                source_type=st,
                ocr_low_confidence=p.ocr_low_confidence,
            )
        )
    return dedupe_extracted_metrics(rows)


def _safe_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)
    return base[:200] if len(base) > 200 else base


def ingest_pdf_path(
    pdf_path: Path,
    company_name: str | None = None,
    version: str | None = None,
    client_label: str = "default",
) -> dict:
    """
    Ingest a PDF from disk (offline batch). Same pipeline as ingest_pdf(); copies into UPLOADS_DIR.
    """
    pdf_path = pdf_path.resolve()
    file_bytes = pdf_path.read_bytes()
    return ingest_pdf(file_bytes, pdf_path.name, company_name, version, client_label)


def ingest_pdf(
    file_bytes: bytes,
    original_filename: str,
    company_name: str | None = None,
    version: str | None = None,
    client_label: str = "default",
) -> dict:
    """
    Persist PDF, extract pages, chunk, embed, store in configured backend.
    company_name and version are user-provided metadata for multi-version support.
    client_label scopes data for optional multi-tenant demos (see README).
    """
    ensure_dirs()
    safe = _safe_filename(original_filename)
    path = save_upload(file_bytes, UPLOADS_DIR, safe)
    document_name = safe
    cl = (client_label or "default").strip() or "default"

    inferred = infer_metadata_from_filename(original_filename)
    resolved_company = (company_name or "").strip() or inferred.company_name or "Unknown"
    resolved_version = (version or "").strip() or inferred.version_label or "unknown"
    LOGGER.info(
        "ingest metadata %s => company=%s version=%s year=%s month=%s",
        original_filename,
        resolved_company,
        resolved_version,
        inferred.document_year,
        inferred.document_month,
    )
    if has_document_version(document_name, resolved_version, client_label=cl):
        LOGGER.warning(
            "duplicate version ingest detected: document=%s version=%s client=%s",
            document_name,
            resolved_version,
            cl,
        )

    pages = extract_pdf_pages(file_bytes, original_filename)
    chunks = pages_to_chunks(
        pages,
        document_name,
        resolved_company,
        resolved_version,
        document_year=inferred.document_year,
        document_month=inferred.document_month,
    )
    if not chunks:
        doc_id = add_document(
            str(path),
            document_name,
            resolved_company,
            resolved_version,
            len(pages),
            cl,
            document_year=inferred.document_year,
            document_month=inferred.document_month,
        )
        insert_extracted_metrics(
            doc_id,
            document_name,
            _metrics_from_pages(
                pages,
                resolved_company,
                resolved_version,
                inferred.document_year,
                inferred.document_month,
            ),
        )
        return {
            "document_id": doc_id,
            "document_name": document_name,
            "chunks": 0,
            "pages": len(pages),
            "message": "No text chunks produced (empty or unscanned PDF).",
        }

    start_idx = next_vector_key()
    texts = [c["chunk_text"] for c in chunks]
    vectors = embed_texts(texts)

    doc_id = add_document(
        str(path),
        document_name,
        resolved_company,
        resolved_version,
        len(pages),
        cl,
        document_year=inferred.document_year,
        document_month=inferred.document_month,
    )
    insert_extracted_metrics(
        doc_id,
        document_name,
        _metrics_from_pages(
            pages,
            resolved_company,
            resolved_version,
            inferred.document_year,
            inferred.document_month,
        ),
    )
    insert_chunks_after_embed(doc_id, chunks, vectors, start_idx)

    backend = "Postgres (pgvector)" if use_postgres() else "SQLite + FAISS"
    return {
        "document_id": doc_id,
        "document_name": document_name,
        "chunks": len(chunks),
        "pages": len(pages),
        "message": f"Ingested {len(chunks)} chunks from {len(pages)} pages ({backend}).",
    }
