"""End-to-end: PDF → chunks → database (Postgres+pgvector or SQLite+FAISS)."""

from __future__ import annotations

import re
from pathlib import Path

from src.chunking import pages_to_chunks
from src.config import UPLOADS_DIR, ensure_dirs, use_postgres
from src.embeddings import embed_texts
from src.ingestion import extract_pdf_pages, save_upload
from src.persistence import add_document, insert_chunks_after_embed, next_vector_key


def _safe_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)
    return base[:200] if len(base) > 200 else base


def ingest_pdf_path(
    pdf_path: Path,
    company_name: str,
    version: str,
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
    company_name: str,
    version: str,
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

    pages = extract_pdf_pages(file_bytes, original_filename)
    chunks = pages_to_chunks(pages, document_name, company_name.strip(), version.strip())
    if not chunks:
        doc_id = add_document(str(path), document_name, company_name, version, len(pages), cl)
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

    doc_id = add_document(str(path), document_name, company_name, version, len(pages), cl)
    insert_chunks_after_embed(doc_id, chunks, vectors, start_idx)

    backend = "Postgres (pgvector)" if use_postgres() else "SQLite + FAISS"
    return {
        "document_id": doc_id,
        "document_name": document_name,
        "chunks": len(chunks),
        "pages": len(pages),
        "message": f"Ingested {len(chunks)} chunks from {len(pages)} pages ({backend}).",
    }
