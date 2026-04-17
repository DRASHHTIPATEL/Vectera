"""Render PDF pages to PNG bytes for UI preview (Streamlit)."""

from __future__ import annotations

import io
from pathlib import Path


def render_pdf_page_png_bytes(
    pdf_path: Path,
    page_number: int,
    *,
    resolution: int = 144,
) -> bytes | None:
    """
    Rasterize a single page (1-based page_number) to PNG bytes.
    Uses pdfplumber (already a project dependency).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        return None
    try:
        import pdfplumber
    except Exception:
        return None
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                return None
            page = pdf.pages[page_number - 1]
            im = page.to_image(resolution=resolution)
            buf = io.BytesIO()
            im.original.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        return None
