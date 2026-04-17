"""PDF ingestion: per-page text, metadata inference, optional structured/OCR extraction."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import shutil
from pathlib import Path
from typing import BinaryIO

from src.config import CHART_OCR_ENABLED

LOGGER = logging.getLogger(__name__)
_TESSERACT_WARNED = False

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
_MAX_IMAGE_REGIONS_OCR = 6


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

    qfy = re.search(r"\bq([1-4])[-_\s]?(20\d{2})\b", lower)
    if qfy:
        year = int(qfy.group(2))

    md = re.search(r"\b(\d{1,2})[./-](\d{1,2})[./-](20\d{2})\b", lower)
    if md and year is None:
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

    # Extra passes when primary patterns miss (e.g. filenames with investor-day-2025, appendix dates).
    if year is None:
        for _pat in (
            r"investor[-_\s]?day[-_\s]?(20\d{2})",
            r"roadshow[-_\s]?(20\d{2})",
            r"(20\d{2})[-_\s]?appendix",
            r"appendix[-_\s]?(20\d{2})",
        ):
            m2 = re.search(_pat, lower)
            if m2:
                year = int(m2.group(m2.lastindex))
                break
    if year is None:
        years_found = sorted({int(x) for x in re.findall(r"\b(20\d{2})\b", lower)})
        if years_found:
            year = years_found[-1]
    if month is None:
        mshort = re.search(
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-_\.](\d{2})\b(?!\d)",
            lower,
        )
        if mshort:
            mo = MONTH_MAP.get(mshort.group(1)[:3])
            yy = int(mshort.group(2))
            full_y = yy + (2000 if yy < 50 else 1900)
            if full_y >= 2000 and full_y <= 2035:
                month = mo
                if year is None:
                    year = full_y

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


def _should_run_chart_ocr(text: str, n_images: int, table_parts: bool) -> bool:
    """
    When to rasterize the page and run Tesseract.

    Many decks draw charts as vectors (no `page.images`); full-page render still paints them,
    so we also trigger on chart keywords + moderate text length without requiring embedded images.
    """
    if table_parts:
        return False
    if n_images > 0 and (len(text) < 80 or _page_suggests_chart_or_figure(text, n_images)):
        return True
    t = text.lower()
    if len(text) < 280 and any(w in t for w in _CHART_HINT_WORDS):
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


def _infer_chart_series_line(txt: str) -> str:
    """
    Infer a simple label->value mapping for 2-category bar charts from OCR text.
    Example target: [CHART_OCR_SERIES] US=17.7%; EGP=24.2%
    """
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return ""

    label_line = ""
    for ln in lines:
        toks = re.findall(r"\b[A-Z]{2,6}\b", ln)
        if 2 <= len(toks) <= 4:
            label_line = ln
            break
    if not label_line:
        return ""
    labels = re.findall(r"\b[A-Z]{2,6}\b", label_line)
    if len(labels) < 2:
        return ""
    labels = labels[:2]

    vals: list[float] = []
    for m in re.finditer(r"\b(\d{1,3}(?:\.\d+)?)\s*%", txt):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        # Filter likely axis ticks (round values like 0/5/10/.../100)
        if abs(v - round(v)) < 1e-9 and (int(round(v)) % 5 == 0):
            continue
        if 0.0 <= v <= 100.0:
            vals.append(v)
    # Fallback if everything was filtered out.
    if len(vals) < 2:
        for m in re.finditer(r"\b(\d{1,3}(?:\.\d+)?)\s*%", txt):
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            if 0.0 <= v <= 100.0:
                vals.append(v)
    if len(vals) < 2:
        return ""

    uniq_vals: list[float] = []
    for v in vals:
        if any(abs(v - u) < 1e-6 for u in uniq_vals):
            continue
        uniq_vals.append(v)
        if len(uniq_vals) >= len(labels):
            break
    if len(uniq_vals) < len(labels):
        return ""

    if len(labels) == 2 and len(uniq_vals) == 2:
        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

        def _loose_token_pat(tok: str) -> str:
            chars = [re.escape(ch) for ch in tok if ch.isalnum()]
            if not chars:
                return re.escape(tok)
            return r"\s*".join(chars)

        l0 = _norm(labels[0])
        l1 = _norm(labels[1])
        low_txt = _norm(txt)
        p0 = _loose_token_pat(l0)
        p1 = _loose_token_pat(l1)
        # If OCR text states one category is greater than the other, map accordingly.
        if re.search(rf"\b{p1}\b.*\bgreater than\b.*\b{p0}\b", low_txt):
            uniq_vals = [min(uniq_vals), max(uniq_vals)]
        elif re.search(rf"\b{p0}\b.*\bgreater than\b.*\b{p1}\b", low_txt):
            uniq_vals = [max(uniq_vals), min(uniq_vals)]

    pairs = [f"{lab}={val:g}%" for lab, val in zip(labels, uniq_vals)]
    return "[CHART_OCR_SERIES] " + "; ".join(pairs)


def _ensure_tesseract_logged_once() -> bool:
    global _TESSERACT_WARNED
    ok = bool(shutil.which("tesseract"))
    if not ok and not _TESSERACT_WARNED:
        LOGGER.warning(
            "Tesseract not found on PATH — chart/figure OCR is skipped during ingest. "
            "Install the engine (e.g. `brew install tesseract` on macOS) and re-run ingest."
        )
        _TESSERACT_WARNED = True
    return ok


def _pdf_image_bbox_to_pil_crop(img_dict: dict, page: object, pil_full: object):
    """Map pdfplumber image dict x0/top/x1/bottom to a PIL crop of the rendered page."""
    try:
        pw, ph = float(page.width), float(page.height)
        iw, ih = pil_full.size
        sx = iw / pw
        sy = ih / ph
        x0, x1 = float(img_dict["x0"]), float(img_dict["x1"])
        top, bottom = float(img_dict["top"]), float(img_dict["bottom"])
        left = int(max(0, min(x0, x1) * sx))
        right = int(min(iw, max(x0, x1) * sx))
        upper = int(max(0, min(top, bottom) * sy))
        lower = int(min(ih, max(top, bottom) * sy))
        if right - left < 20 or lower - upper < 20:
            return None
        return pil_full.crop((left, upper, right, lower))
    except Exception:
        return None


def _ocr_pil_image(img: object) -> str:
    """Tesseract on one PIL image: preprocess + multi-PSM merge (best-effort)."""
    try:
        import pytesseract  # type: ignore
        from PIL import ImageEnhance, ImageOps
    except Exception:
        return ""

    try:
        if getattr(img, "mode", None) not in ("L", "1"):
            img = img.convert("L")
        img = ImageOps.autocontrast(img, cutoff=2)
        img = ImageEnhance.Contrast(img).enhance(1.12)
    except Exception:
        pass

    chunks: list[str] = []
    for cfg in ("--oem 3 --psm 6", "--oem 3 --psm 11", "--oem 3 --psm 4"):
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
    return merged.strip()


def _try_ocr_text(page: object) -> str:
    """
    Chart/figure pages: full-page raster OCR plus optional per-embedded-image crops.
    Marked low-confidence downstream; requires Tesseract on PATH when enabled.
    """
    if not CHART_OCR_ENABLED:
        return ""
    if not _ensure_tesseract_logged_once():
        return ""
    try:
        import pytesseract  # noqa: F401
    except Exception:
        return ""

    try:
        pil_img = page.to_image(resolution=_OCR_RESOLUTION).original
    except Exception:
        return ""

    parts: list[str] = []
    full_txt = _ocr_pil_image(pil_img)
    if full_txt:
        parts.append(full_txt)

    for img_meta in (getattr(page, "images", None) or [])[:_MAX_IMAGE_REGIONS_OCR]:
        crop = _pdf_image_bbox_to_pil_crop(img_meta, page, pil_img)
        if crop is None:
            continue
        region_txt = _ocr_pil_image(crop)
        if region_txt:
            parts.append(region_txt)

    if not parts:
        return ""
    merged = _merge_ocr_passes(parts)
    if not merged.strip():
        return ""
    merged = _scan_numeric_tokens_for_chart_ocr(merged)
    series_line = _infer_chart_series_line(merged)
    if series_line:
        merged = merged + "\n" + series_line
    return f"[CHART_OR_FIGURE_OCR]\n[OCR_EXTRACTED]\n{merged}"


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
            want_chart_ocr = _should_run_chart_ocr(text, n_img, bool(table_parts))
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
