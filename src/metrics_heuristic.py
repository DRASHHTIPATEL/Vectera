"""
Deterministic financial-style metric extraction from text/tables (no LLM, no APIs).
Stores original display strings plus normalized_value for analytics (ratios, USD base units).
"""

from __future__ import annotations

import re
from typing import Any

_CONF_RANK = {"high": 3, "medium": 2, "low": 1}
_SOURCE_RANK = {"table": 3, "text": 2, "ocr": 1}


def standardize_confidence(source_type: str, ocr_low_confidence: bool) -> str:
    """high = structured table; medium = text; low = OCR."""
    if ocr_low_confidence or source_type == "ocr":
        return "low"
    if source_type == "table":
        return "high"
    return "medium"


def _parse_percent_to_ratio(s: str) -> float | None:
    m = re.match(r"^\s*(\d{1,3}(?:\.\d+)?)\s*%\s*$", s.strip())
    if m:
        return float(m.group(1)) / 100.0
    return None


def _parse_currency_to_base_usd(s: str) -> float | None:
    t = s.strip().replace(",", "")
    m = re.match(r"^\s*\$?\s*([\d.]+)\s*([BMK])?\s*$", t, re.I)
    if not m:
        return None
    n = float(m.group(1))
    suf = (m.group(2) or "").upper()
    mult = {"B": 1e9, "M": 1e6, "K": 1e3}.get(suf, 1.0)
    return n * mult


def _parse_plain_usd_or_per_share(s: str) -> float | None:
    t = s.strip().replace(",", "")
    m = re.match(r"^\s*\$?\s*([\d.]+)\s*$", t)
    if not m:
        return None
    return float(m.group(1))


def normalize_metric_value(original_value: str, unit_hint: str | None) -> tuple[float | None, str | None]:
    """
    Returns (normalized_value, canonical_unit).
    Percent display → ratio 0–1 (e.g. 84.7% → 0.847). Currency → USD float. unit_hint from rules.
    """
    raw = (original_value or "").strip()
    uh = (unit_hint or "").lower()

    if uh == "percent" or raw.endswith("%"):
        r = _parse_percent_to_ratio(raw if raw.endswith("%") else f"{raw}%")
        if r is not None:
            return r, "ratio"

    if uh == "usd" or raw.startswith("$"):
        u = _parse_currency_to_base_usd(raw)
        if u is not None:
            return u, "usd"

    if uh == "per_share":
        u = _parse_plain_usd_or_per_share(raw.replace("$", "").strip())
        if u is not None:
            return u, "usd_per_share"

    if uh == "percent":
        r = _parse_percent_to_ratio(raw if "%" in raw else raw + "%")
        if r is not None:
            return r, "ratio"

    return None, None


def _apply_normalization_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Mutates row: original_value, normalized_value (float), unit canonical."""
    orig = str(row.get("value", "")).strip()
    row["original_value"] = orig
    nv, can_unit = normalize_metric_value(orig, row.get("unit"))
    row["normalized_value"] = nv
    if can_unit:
        row["unit"] = can_unit
    return row


def _dedupe_same_pass(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Across regex passes: same metric+value+page once."""
    seen: set[tuple[str, str, int]] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        key = (r.get("metric_name", ""), r.get("value", ""), int(r.get("page_number") or 0))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def dedupe_extracted_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    One row per (company_name, metric_name, version): prefer higher confidence,
    table > text > OCR, then parsed numeric, then shorter/cleaner original string.
    """
    if not rows:
        return []
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in rows:
        co = (r.get("company_name") or "").strip()
        mn = (r.get("metric_name") or "").strip().lower()
        ver = (r.get("version") or "").strip()
        buckets.setdefault((co, mn, ver), []).append(r)

    out: list[dict[str, Any]] = []
    for group in buckets.values():

        def sort_key(x: dict[str, Any]) -> tuple[int, int, int, int]:
            conf = _CONF_RANK.get(x.get("confidence") or "medium", 2)
            src = _SOURCE_RANK.get(x.get("source_type") or "text", 2)
            nv = x.get("normalized_value")
            has_nv = 1 if nv is not None else 0
            clean = -len(str(x.get("original_value") or x.get("value") or ""))
            return (conf, src, has_nv, clean)

        best = max(group, key=sort_key)
        out.append(best)
    return out


def extract_metrics_from_text(
    text: str,
    *,
    page_number: int,
    company_name: str,
    version: str,
    document_year: int | None,
    document_month: int | None,
    source_type: str,
    ocr_low_confidence: bool = False,
) -> list[dict[str, Any]]:
    base_conf = standardize_confidence(source_type, ocr_low_confidence)
    t = text or ""
    lower = t.lower()
    out: list[dict[str, Any]] = []

    def _row(
        metric_name: str,
        display_value: str,
        unit_hint: str,
        confidence_override: str | None = None,
    ) -> dict[str, Any]:
        conf = confidence_override or base_conf
        row: dict[str, Any] = {
            "metric_name": metric_name,
            "value": display_value,
            "page_number": page_number,
            "company_name": company_name,
            "version": version,
            "document_year": document_year,
            "document_month": document_month,
            "source_type": source_type,
            "confidence": conf,
            "unit": unit_hint,
        }
        return _apply_normalization_fields(row)

    for m in re.finditer(
        r"(occupancy|leased|utilization)[^\n]{0,50}?[:.]?\s*(\d{1,3}(?:\.\d+)?)\s*%",
        lower,
        re.I,
    ):
        out.append(_row(m.group(1).title(), f"{m.group(2)}%", "percent"))

    for m in re.finditer(
        r"(total\s+)?(revenue|sales|rent)[^\n]{0,60}?[:.]?\s*(\$[\d,]+(?:\.\d+)?(?:\s*[BMK])?)",
        lower,
        re.I,
    ):
        out.append(_row((m.group(2) or "revenue").title(), m.group(3), "usd"))

    for m in re.finditer(
        r"(\d{1,3}(?:\.\d+)?)\s*%(?:[^\n]{0,20})?(yoy|y/y|year[- ]over[- ]year|growth)",
        lower,
        re.I,
    ):
        out.append(
            _row(
                "Growth",
                f"{m.group(1)}%",
                "percent",
                confidence_override="medium" if base_conf == "high" else base_conf,
            )
        )

    for label, pat in (
        ("NOI", r"noi[^\n]{0,40}?[:.]?\s*(\$[\d,]+(?:\.\d+)?(?:\s*[BMK])?)"),
        ("FFO", r"ffo[^\n]{0,20}?per\s+share[^\n]{0,10}?[:.]?\s*\$?\s*(\d+(?:\.\d+)?)"),
        ("AFFO", r"affo[^\n]{0,20}?per\s+share[^\n]{0,10}?[:.]?\s*\$?\s*(\d+(?:\.\d+)?)"),
    ):
        for m in re.finditer(pat, lower, re.I):
            val = m.group(1)
            if val.startswith("$"):
                out.append(_row(label, val, "usd"))
            else:
                out.append(
                    _row(
                        label,
                        f"${val}",
                        "per_share",
                        confidence_override="medium" if base_conf == "high" else base_conf,
                    )
                )

    out = _dedupe_same_pass(out)
    return out


def metric_orm_sort_key(
    em: Any,
    *,
    tokens: list[str],
    max_recency_units: int,
) -> tuple[float, int, int, float]:
    """Sort key (higher is better) for DB metric rows: relevance, confidence, source, recency."""
    mn = (getattr(em, "metric_name", None) or "").lower()
    val = (getattr(em, "value", None) or "").lower()
    ws = 0.0
    for t in tokens:
        if t in mn or t in val:
            ws += 2.0
    conf = _CONF_RANK.get(getattr(em, "confidence", None) or "medium", 2)
    src = _SOURCE_RANK.get(getattr(em, "source_type", None) or "text", 2)
    y = getattr(em, "document_year", None) or 0
    mo = getattr(em, "document_month", None) or 0
    rec_u = y * 12 + mo
    rec = float(rec_u) / float(max(max_recency_units, 1))
    return (ws, conf, src, rec)


def rank_metric_query_matches(
    rows_orm: list[tuple[Any, str]],
    tokens: list[str],
    limit: int,
) -> list[tuple[Any, str]]:
    """Filter by token overlap (when tokens non-empty); rank by relevance × confidence × source × recency."""
    if not rows_orm:
        return []
    max_rec = max(
        ((getattr(em, "document_year", None) or 0) * 12 + (getattr(em, "document_month", None) or 0))
        for em, _ in rows_orm
    )
    max_rec = max(max_rec, 1)
    scored: list[tuple[tuple[float, int, int, float], Any, str]] = []
    for em, clab in rows_orm:
        sk = metric_orm_sort_key(em, tokens=tokens, max_recency_units=max_rec)
        if tokens and sk[0] == 0.0:
            continue
        scored.append((sk, em, clab))
    scored.sort(key=lambda x: x[0], reverse=True)
    seen: set[tuple[str, str, str]] = set()
    out: list[tuple[Any, str]] = []
    for _, em, clab in scored:
        key = (getattr(em, "metric_name", "") or "", getattr(em, "value", "") or "", getattr(em, "version", "") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append((em, clab))
        if len(out) >= limit:
            break
    if not out and rows_orm and not tokens:
        for em, clab in rows_orm[:limit]:
            out.append((em, clab))
    return out[:limit]


def format_metric_chunk_row(row: dict[str, Any]) -> str:
    """Single retrieval chunk text for LLM grounding."""
    conf = row.get("confidence", "unknown")
    st = row.get("source_type", "text")
    pg = row.get("page_number", "?")
    orig = row.get("original_value") or row.get("value", "")
    nv = row.get("normalized_value")
    nv_part = f" | normalized={nv}" if nv is not None else ""
    return (
        f"[STRUCTURED_METRIC] {row.get('metric_name','')}: {orig}{nv_part} "
        f"(page {pg}, source={st}, confidence={conf})"
    )
