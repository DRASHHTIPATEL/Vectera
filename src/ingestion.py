"""PDF ingestion: per-page text, optional table extraction, chart heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass
class PageContent:
    page_number: int  # 1-based
    text: str
    tables_markdown: str
    chart_warning: str | None


def extract_pdf_pages(file_obj: BinaryIO | bytes, filename: str) -> list[PageContent]:
    """
    Extract text per page using pdfplumber (layout-aware) and append table summaries.
    If a page is image-heavy / low text, set chart_warning for downstream messaging.
    """
    import io

    import pdfplumber

    if isinstance(file_obj, bytes):
        file_obj = io.BytesIO(file_obj)

    pages_out: list[PageContent] = []
    with pdfplumber.open(file_obj) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()

            table_parts: list[str] = []
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for ti, table in enumerate(tables):
                if not table:
                    continue
                lines = []
                for row in table:
                    cells = [str(c).strip() if c is not None else "" for c in row]
                    if any(cells):
                        lines.append(" | ".join(cells))
                if lines:
                    table_parts.append(f"[Table {ti + 1} on page {i}]\n" + "\n".join(lines))

            tables_md = "\n\n".join(table_parts)
            combined = text
            if tables_md:
                combined = (combined + "\n\n" + tables_md).strip() if combined else tables_md

            chart_warning = None
            if len(text) < 80 and not table_parts:
                imgs = getattr(page, "images", None) or []
                if imgs:
                    chart_warning = (
                        "Chart data could not be fully extracted — page appears image-heavy."
                    )
                elif not combined:
                    chart_warning = "Chart data could not be fully extracted — little or no text on this page."

            pages_out.append(
                PageContent(
                    page_number=i,
                    text=combined or "",
                    tables_markdown=tables_md,
                    chart_warning=chart_warning,
                )
            )
    return pages_out


def save_upload(file_bytes: bytes, dest_dir: Path, safe_name: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / safe_name
    path.write_bytes(file_bytes)
    return path
