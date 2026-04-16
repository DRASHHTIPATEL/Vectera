#!/usr/bin/env python3
"""
Offline ingestion: PDFs from a directory → extract → chunk → embed → persist (SQLite+FAISS or Postgres).

Run from repo root:
  python ingest.py ./data
  python ingest.py ./pdfs --company "Acme Corp" --version Q3-2024 --client default --recursive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ensure_dirs, use_postgres
from src.pipeline import ingest_pdf_path


def _collect_pdfs(directory: Path, recursive: bool) -> list[Path]:
    directory = directory.resolve()
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")
    if recursive:
        paths = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    else:
        paths = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    return sorted(paths, key=lambda p: p.name.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDFs offline for Vectera RAG (chunk + embed + store).")
    parser.add_argument(
        "directory",
        type=Path,
        help="Folder containing .pdf files (e.g. ./data)",
    )
    parser.add_argument(
        "--company",
        default=None,
        help="Optional override for company_name on all PDFs in this run. If omitted, inferred from filename.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Optional override for version label on all PDFs. If omitted, inferred from filename.",
    )
    parser.add_argument("--client", default="default", help="Client / workspace label (default: default).")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also index PDFs in subdirectories.",
    )
    args = parser.parse_args()

    ensure_dirs()
    pdfs = _collect_pdfs(args.directory, args.recursive)
    if not pdfs:
        print(f"No PDF files found in {args.directory.resolve()}", file=sys.stderr)
        raise SystemExit(1)

    backend = "Postgres (pgvector)" if use_postgres() else "SQLite + FAISS"
    print(f"Backend: {backend}")
    print(f"Found {len(pdfs)} PDF(s).")

    cl = (args.client or "default").strip() or "default"
    ver = (args.version or "").strip() or None

    ok = 0
    for p in pdfs:
        company = (args.company or "").strip() or None
        print(f"  → {p.name} (company={company!r}, version={ver!r}, client={cl!r})")
        try:
            result = ingest_pdf_path(p, company, ver, client_label=cl)
            print(f"     {result['message']}")
            ok += 1
        except Exception as e:
            print(f"     ERROR: {e}", file=sys.stderr)

    print(f"Done. Successfully processed {ok}/{len(pdfs)} file(s).")
    if not use_postgres():
        from src.config import FAISS_PATH

        print(f"Vector index: {FAISS_PATH} (persisted for query-time retrieval.)")


if __name__ == "__main__":
    main()
