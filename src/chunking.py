"""
Structure-first chunking: split on paragraphs and sentence boundaries,
merge small units, overlap between adjacent chunks — not a single fixed token window.
"""

from __future__ import annotations

import re
from typing import TypedDict

from src.config import CHUNK_MIN_CHARS, CHUNK_OVERLAP_CHARS, CHUNK_TARGET_CHARS
from src.ingestion import PageContent


class ChunkPayload(TypedDict, total=False):
    document_name: str
    page_number: int
    company_name: str
    version: str
    chunk_text: str
    chart_note: str | None


def _split_into_units(text: str) -> list[str]:
    """Prefer paragraph breaks, then lines, then sentences."""
    text = re.sub(r"\r\n", "\n", text).strip()
    if not text:
        return []
    parts = re.split(r"\n\s*\n+", text)
    units: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) <= CHUNK_TARGET_CHARS:
            units.append(p)
        else:
            units.extend(_split_long_block(p))
    return units


def _split_long_block(block: str) -> list[str]:
    """Split long paragraphs on sentence-ish boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", block)
    out: list[str] = []
    buf = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) + 1 <= CHUNK_TARGET_CHARS:
            buf = f"{buf} {s}".strip()
        else:
            if buf:
                out.append(buf)
            if len(s) > CHUNK_TARGET_CHARS:
                for i in range(0, len(s), CHUNK_TARGET_CHARS - CHUNK_OVERLAP_CHARS):
                    out.append(s[i : i + CHUNK_TARGET_CHARS])
                buf = ""
            else:
                buf = s
    if buf:
        out.append(buf)
    return out


def _merge_small_units(units: list[str]) -> list[str]:
    if not units:
        return []
    merged: list[str] = []
    cur = units[0]
    for u in units[1:]:
        if len(cur) < CHUNK_MIN_CHARS and len(cur) + len(u) + 2 <= CHUNK_TARGET_CHARS * 1.2:
            cur = f"{cur}\n\n{u}"
        else:
            merged.append(cur)
            cur = u
    merged.append(cur)
    return merged


def _overlap_split(big: str) -> list[str]:
    """If still over target, window with overlap."""
    if len(big) <= CHUNK_TARGET_CHARS * 1.15:
        return [big]
    chunks: list[str] = []
    start = 0
    n = len(big)
    while start < n:
        end = min(start + CHUNK_TARGET_CHARS, n)
        piece = big[start:end]
        chunks.append(piece.strip())
        if end >= n:
            break
        start = max(0, end - CHUNK_OVERLAP_CHARS)
    return [c for c in chunks if c]


def page_to_chunks(
    page: PageContent,
    document_name: str,
    company_name: str,
    version: str,
) -> list[ChunkPayload]:
    raw = page.text.strip()
    if not raw:
        if page.chart_warning:
            return [
                {
                    "document_name": document_name,
                    "page_number": page.page_number,
                    "company_name": company_name,
                    "version": version,
                    "chunk_text": f"[Page {page.page_number} — no extractable text] {page.chart_warning}",
                    "chart_note": page.chart_warning,
                }
            ]
        return []

    units = _merge_small_units(_split_into_units(raw))
    final_chunks: list[str] = []
    for u in units:
        final_chunks.extend(_overlap_split(u))

    chart_note = page.chart_warning
    out: list[ChunkPayload] = []
    for i, ct in enumerate(final_chunks):
        note = chart_note if i == 0 else None
        out.append(
            {
                "document_name": document_name,
                "page_number": page.page_number,
                "company_name": company_name,
                "version": version,
                "chunk_text": ct,
                "chart_note": note,
            }
        )
    return out


def pages_to_chunks(
    pages: list[PageContent],
    document_name: str,
    company_name: str,
    version: str,
) -> list[ChunkPayload]:
    all_chunks: list[ChunkPayload] = []
    for p in pages:
        all_chunks.extend(page_to_chunks(p, document_name, company_name, version))
    return all_chunks
