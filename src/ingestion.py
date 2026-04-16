"""PDF ingestion: per-page text, metadata inference, optional structured/OCR extraction."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from pathlib import Path
from typing import BinaryIO

LOGGER = logging.getLogger(__name__)

# Visual / chart-style pages: OCR when pdf text is thin but raster content likely.
_CHART_HINT_WORDS = frozenset(
    (
        "figure",
        "chart",
        "graph",
        "exhibit",
        "appendix",
        "graphic",
        "illustrative",
        "nareit",
        "source:",
        "y/y",
        "yoy",
    )
)
_OCR_RESOLUTION = 300


MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
SHORT_MONTH = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dec",
}


@dataclass
class DocumentMetadata:
    company_name: str
    document_year: int | None
    document_month: int | None
    version_label: str


@dataclass
class PageContent:
    page_number: int  # 1-based
    text: str
    tables_markdown: str
    chart_warning: str | None
    structured_type: str | None = None
    ocr_low_confidence: bool = False


def infer_metadata_from_filename(filename: str) -> DocumentMetadata:
    stem = Path(filename).stem
    cleaned = re.sub(r"[_\-]+", " ", stem)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    lower = cleaned.lower()

    month = None
    year = None
    md = re.search(r"\b(\d{1,2})[./-](\d{1,2})[./-](20\d{2})\b", lower)
    if md:
        month = int(md.group(1))
        year = int(md.group(3))
    m = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"[\s._-]*(20\d{2}|\d{2})\b",
        lower,
    )
    if m and year is None:
        month = MONTH_MAP.get(m.group(1)[:3], MONTH_MAP.get(m.group(1), None))
        yy = int(m.group(2))
        year = yy if yy >= 2000 else 2000 + yy
    else:
        y = re.search(r"\b(20\d{2})\b", lower)
        if y and year is None:
            year = int(y.group(1))

    if year and month:
        version_label = f"{SHORT_MONTH.get(month, 'unk')}-{year}"
    elif year:
        version_label = str(year)
    else:
        version_label = "unknown"

    tokens = re.split(r"[\s._-]+", cleaned)
    stop = {
        "investor",
        "presentation",
        "deck",
        "session",
        "roadshow",
        "company",
        "update",
        "appendix",
        "report",
        "merger",
        "resize",
    }
    head: list[str] = []
    for t in tokens:
        lt = t.lower()
        if (
            lt in stop
            or re.fullmatch(r"q[1-4]", lt)
            or re.fullmatch(r"20\d{2}", lt)
            or re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}", lt)
            or lt in MONTH_MAP
        ):
            break
        head.append(t)
    company_name = " ".join(head).strip() or cleaned
    if company_name.islower():
        company_name = company_name.title()

    return DocumentMetadata(
        company_name=company_name.strip(),
        document_year=year,
        document_month=month,
        version_label=version_label,
    )


def _normalize_numeric_noise(text: str) -> str:
    text = re.sub(r"(\d)\s+\.\s+(\d)", r"\1.\2", text)
    text = re.sub(r"(\d)\s+%", r"\1%", text)
    text = re.sub(r"\$\s+(\d)", r"$\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _format_table(table: list[list[str | None]], page_number: int, idx: int) -> str:
    lines: list[str] = []
    for row in table:
        cells = [str(c).strip() if c is not None else "" for c in row]
        cells = [_normalize_numeric_noise(c) for c in cells if c and c.strip()]
        if cells:
            lines.append(" | ".join(cells))
    if not lines:
        return ""
    return f"[STRUCTURED_TABLE] [Table {idx + 1} on page {page_number}]\n" + "\n".join(lines)


def _extract_tables_fallback(bytes_data: bytes, page_number: int) -> list[list[list[str | None]]]:
    tables: list[list[list[str | None]]] = []
    try:
        import camelot  # type: ignore
        import io
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tf:
            tf.write(bytes_data)
            tf.flush()
            out = camelot.read_pdf(tf.name, pages=str(page_number), flavor="stream")
            for t in out:
                vals = t.df.values.tolist()
                tables.append([[str(c) for c in row] for row in vals])
    except Exception:
        pass
    if tables:
        return tables
    try:
        import tabula  # type: ignore
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tf:
            tf.write(bytes_data)
            tf.flush()
            out = tabula.read_pdf(tf.name, pages=page_number, multiple_tables=True, pandas_options={"dtype": str})
            for df in out or []:
                vals = df.fillna("").astype(str).values.tolist()
                tables.append(vals)
    except Exception:
        pass
    return tables


def _page_suggests_chart_or_figure(text: str, n_images: int) -> bool:
    """Heuristic: likely slide with chart/figure when text is short and hints present."""
    if n_images == 0:
        return False
    if len(text) < 72:
        return True
    t = text.lower()
    if len(text) < 240 and any(w in t for w in _CHART_HINT_WORDS):
        return True
    return False


def _merge_ocr_passes(chunks: list[str]) -> str:
    """Deduplicate lines from multiple Tesseract configs; preserve rough order."""
    seen: set[str] = set()
    out_lines: list[str] = []
    for ch in chunks:
        for line in ch.splitlines():
            s = line.strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out_lines.append(s)
    return "\n".join(out_lines)


def _scan_numeric_tokens_for_chart_ocr(txt: str) -> str:
    """Append a compact scan line so retrieval/embeddings see extracted digits."""
    nums: list[str] = []
    for m in re.finditer(r"\b\d{1,3}(?:\.\d+)?\s*%", txt):
        nums.append(m.group(0).replace(" ", ""))
    for m in re.finditer(r"\$[\d,]+(?:\.\d+)?(?:\s*[BMK])?", txt):
        nums.append(m.group(0).replace(" ", ""))
    if not nums:
        return txt
    uniq: list[str] = []
    seen: set[str] = set()
    for n in nums:
        if n.lower() in seen:
            continue
        seen.add(n.lower())
        uniq.append(n)
        if len(uniq) >= 16:
            break
    return txt + "\n[CHART_OCR_NUMBERS_SCAN] " + ", ".join(uniq)


def _try_ocr_text(page: object) -> str:
    """
    Chart/figure pages: higher-res raster OCR with preprocessing + two PSM passes.
    Best-effort only; still marked low-confidence downstream.
    """
    try:
        import pytesseract  # type: ignore
        from PIL import ImageEnhance, ImageOps  # noqa: F401

        img = page.to_image(resolution=_OCR_RESOLUTION).original
        try:
            if img.mode not in ("L", "1"):
                img = img.convert("L")
            img = ImageOps.autocontrast(img, cutoff=2)
            img = ImageEnhance.Contrast(img).enhance(1.12)
        except Exception:
            pass

        chunks: list[str] = []
        for cfg in ("--oem 3 --psm 6", "--oem 3 --psm 11"):
            try:
                t = pytesseract.image_to_string(img, config=cfg) or ""
                t = t.strip()
                if t:
                    chunks.append(t)
            except Exception:
                continue
        if not chunks:
            return ""
        merged = _merge_ocr_passes(chunks)
        merged = _normalize_numeric_noise(merged)
        if not merged.strip():
            return ""
        merged = _scan_numeric_tokens_for_chart_ocr(merged)
        return f"[CHART_OR_FIGURE_OCR]\n[OCR_EXTRACTED]\n{merged}"
    except Exception:
        return ""
    return ""


def extract_pdf_pages(file_obj: BinaryIO | bytes, filename: str) -> list[PageContent]:
    """
    Extract text per page using pdfplumber (layout-aware) and append table summaries.
    If a page is image-heavy / low text, set chart_warning for downstream messaging.
    """
    import io

    import pdfplumber

    bytes_data: bytes
    if isinstance(file_obj, bytes):
        bytes_data = file_obj
        file_obj = io.BytesIO(file_obj)
    else:
        bytes_data = file_obj.read()
        file_obj = io.BytesIO(bytes_data)

    pages_out: list[PageContent] = []
    with pdfplumber.open(file_obj) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            text = _normalize_numeric_noise(text)

            table_parts: list[str] = []
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            if not tables:
                tables = _extract_tables_fallback(bytes_data, i)
            for ti, table in enumerate(tables):
                if not table:
                    continue
                tbl = _format_table(table, i, ti)
                if tbl:
                    table_parts.append(tbl)

            tables_md = "\n\n".join(table_parts)
            combined = text
            if tables_md:
                combined = (combined + "\n\n" + tables_md).strip() if combined else tables_md

            chart_warning = None
            structured_type: str | None = "table" if table_parts else None
            ocr_low = False
            imgs = getattr(page, "images", None) or []
            n_img = len(imgs)
            want_chart_ocr = (
                not table_parts
                and n_img > 0
                and (len(text) < 80 or _page_suggests_chart_or_figure(text, n_img))
            )
            if want_chart_ocr:
                ocr_text = _try_ocr_text(page)
                if ocr_text:
                    combined = (
                        (combined + "\n\n" + ocr_text + "\n[OCR_LOW_CONFIDENCE]").strip()
                        if combined
                        else (ocr_text + "\n[OCR_LOW_CONFIDENCE]")
                    )
                    structured_type = "chart"
                    ocr_low = True
                    LOGGER.info(
                        "Chart/figure OCR (low-confidence) for %s page %s (images=%s)",
                        filename,
                        i,
                        n_img,
                    )
                chart_warning = (
                    "Chart/figure content is only partially captured — values may be incomplete or misread. "
                    "Verify against the original PDF."
                )
            elif len(text) < 80 and not table_parts:
                if imgs:
                    chart_warning = (
                        "Chart data could not be fully extracted — page appears image-heavy "
                        "(OCR unavailable or produced no text)."
                    )
                elif not combined:
                    chart_warning = "Chart data could not be fully extracted — little or no text on this page."
                    LOGGER.info("Skipping low-text page %s in %s", i, filename)

            pages_out.append(
                PageContent(
                    page_number=i,
                    text=combined or "",
                    tables_markdown=tables_md,
                    chart_warning=chart_warning,
                    structured_type=structured_type,
                    ocr_low_confidence=ocr_low,
                )
            )
    return pages_out


def save_upload(file_bytes: bytes, dest_dir: Path, safe_name: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / safe_name
    path.write_bytes(file_bytes)
    return path
