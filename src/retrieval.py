"""
Retrieve top similar chunks, then diversify so answers are not dominated by one PDF.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re

from src.config import (
    MAX_CHUNKS_PER_DOCUMENT_IN_BATCH,
    RETRIEVAL_CANDIDATES,
    RETRIEVAL_FINAL_MAX,
    RETRIEVAL_FINAL_MIN,
)
from src.embeddings import embed_texts
from src.metrics_heuristic import format_metric_chunk_row
from src.persistence import get_chunks_by_keys, match_metrics_for_query, vector_similarity_search

RECENCY_BOOST_MAX = 0.022
SYNTH_CONF_SCORE = {"high": 0.9935, "medium": 0.9925, "low": 0.991}

TREND_HINTS = (
    "trend",
    "over time",
    "changed",
    "change",
    "historical",
    "across versions",
    "year over year",
    "yoy",
    "timeline",
)

MONTH_LOOKUP = {
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


def _extract_query_date(query: str) -> tuple[int | None, int | None]:
    q = query.lower()
    m = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(20\d{2})\b",
        q,
    )
    if m:
        month = MONTH_LOOKUP.get(m.group(1), MONTH_LOOKUP.get(m.group(1)[:3]))
        return int(m.group(2)), month
    y = re.search(r"\b(20\d{2})\b", q)
    if y:
        return int(y.group(1)), None
    return None, None


def _is_trend_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in TREND_HINTS)


def classify_query_intent(query: str) -> dict[str, bool]:
    """Rule-based intent for retrieval (no ML)."""
    ql = query.lower()
    ty, _ = _extract_query_date(query)
    trend = _is_trend_query(query)
    comparison = any(
        p in ql
        for p in (
            "compare",
            "comparison",
            " vs ",
            " versus ",
            "compared to",
            "compared with",
            " vs. ",
            "relative to",
            "against ",
            "between ",
        )
    )
    latest_intent = (
        any(p in ql for p in ("latest", "most recent", "newest", "current"))
        and ty is None
        and not trend
    )
    return {
        "temporal_strict": ty is not None,
        "trend": trend,
        "comparison": comparison,
        "latest_intent": latest_intent,
        "direct_metric": _is_numeric_or_metric_query(query),
    }


def _is_numeric_or_metric_query(query: str) -> bool:
    q = query.lower()
    keys = (
        "occupancy",
        "revenue",
        "sales",
        "rent",
        "ffo",
        "noi",
        "affo",
        "npi",
        "growth",
        "yoy",
        "margin",
        "ebitda",
        "cap rate",
        "how much",
        "what is",
        "what was",
        "what were",
        "percentage",
    )
    if any(k in q for k in keys):
        return True
    if re.search(r"\b\d{1,3}(?:\.\d+)?\s*%|\$[\d,]+|million|billion|\bbps\b", q):
        return True
    return False


def _synthetic_metric_chunks(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i, row in enumerate(rows):
        conf = row.get("confidence") or "medium"
        low_ocr = conf == "low" or row.get("source_type") == "ocr"
        disp = row.get("original_value") or row.get("value")
        out.append(
            {
                "document_name": row["document_name"],
                "page_number": row["page_number"],
                "company_name": row["company_name"],
                "version": row["version"],
                "document_year": row.get("document_year"),
                "document_month": row.get("document_month"),
                "chunk_text": format_metric_chunk_row(row),
                "chart_note": None,
                "is_structured": True,
                "structured_type": "metric",
                "source_type": row.get("source_type") or "text",
                "ocr_low_confidence": bool(low_ocr),
                "faiss_index": -1 - i,
                "from_stored_metrics": True,
                "confidence": conf,
                "score": float(SYNTH_CONF_SCORE.get(conf, 0.9925)),
                "metric_display_name": row.get("metric_name"),
                "metric_display_value": disp,
                "normalized_value": row.get("normalized_value"),
            }
        )
    return out


def _chunk_passes_temporal_filter(
    meta: dict,
    *,
    query_year: int | None,
    query_month: int | None,
    trend_mode: bool,
    target_versions_by_company: dict[str, tuple[int | None, int | None]],
) -> bool:
    """
    Ingest often leaves document_year/month NULL; do not drop those chunks when the query
    mentions a year — only exclude when the chunk *has* a date that conflicts.

    If the query does not specify a year, apply "latest version" narrowing when not in trend mode
    (null-safe: unknown chunk dates are kept).
    """
    if query_year is not None:
        cy = meta.get("document_year")
        if cy is not None and cy != query_year:
            return False
        if query_month is not None:
            cm = meta.get("document_month")
            if cm is not None and cm != query_month:
                return False
        return True

    if trend_mode:
        return True
    target = target_versions_by_company.get(meta["company_name"])
    if not target:
        return True
    ty, tm = target
    cy = meta.get("document_year")
    cm = meta.get("document_month")
    if ty is not None and cy is not None and cy != ty:
        return False
    if tm is not None and cm is not None and cm != tm:
        return False
    return True


_LEX_STOP = frozenset(
    {
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "does",
        "did",
        "that",
        "this",
        "with",
        "from",
        "into",
        "your",
        "their",
        "there",
        "about",
        "between",
        "compare",
        "comparison",
        "presentation",
        "investor",
        "deck",
        "according",
    }
)


def _lexical_boost(query: str, meta: dict | None) -> float:
    """
    Light hybrid signal on top of dense retrieval: agenda/title pages, token overlap,
    and cross-company confusion (Digital Realty vs Realty Income).
    """
    if meta is None:
        return 0.0
    ql = query.lower()
    text = (meta.get("chunk_text") or "").lower()
    company = (meta.get("company_name") or "").lower()
    docn = (meta.get("document_name") or "").lower()
    page = int(meta.get("page_number") or 999)

    b = 0.0
    # Disambiguate very different REITs that share "Realty" in the name
    if "digital realty" in ql and "realty income" in company:
        b -= 0.14
    if "realty income" in ql and "digital realty" in company:
        b -= 0.14

    if company and len(company) > 3 and company in ql:
        b += 0.045
    elif _matches_company_hint(query, meta.get("company_name") or ""):
        b += 0.028

    if "road ahead" in ql and "road ahead" in text:
        b += 0.065

    if page <= 3 and "investor day" in text and ("bxp" in ql or "bxp" in docn):
        b += 0.06

    if any(
        s in ql
        for s in (
            "present",
            "scheduled",
            "agenda",
            "investor day",
            "who is",
            "who's",
            "road ahead",
        )
    ):
        if any(
            x in text
            for x in (
                "10:00",
                "10:05",
                "10:45",
                "agenda",
                "chairman",
                "ceo",
                "welcome",
                "introduction",
            )
        ):
            b += 0.08

    if any(s in ql for s in ("title", "deck about", "called", "cover page", "cover of")):
        if any(x in text for x in ("the impact of", "brick and mortar", "investor presentation")):
            b += 0.042
        if page <= 2:
            b += 0.025

    q_tokens = [t for t in re.split(r"\W+", ql) if len(t) > 3 and t not in _LEX_STOP]
    overlap = sum(1 for t in q_tokens[:14] if t in text)
    b += min(0.055, overlap * 0.004)

    return b


def _rerank_with_lexical(
    query: str,
    order: list[tuple[float, int]],
    by_key: dict[int, dict],
) -> list[tuple[float, int]]:
    boosted: list[tuple[float, int]] = []
    for sim, ck in order:
        ck = int(ck)
        adj = float(sim) + _lexical_boost(query, by_key.get(ck))
        boosted.append((adj, ck))
    boosted.sort(key=lambda x: -x[0])
    return boosted


def _comparison_needs_more_diversity(
    metas: list[dict],
) -> tuple[bool, set[tuple[int | None, int | None]], set[str | None]]:
    """True if every chunk is the same period+file (common when only newest deck ranks high)."""
    if len(metas) < 2:
        return True, set(), set()
    periods = {(m.get("document_year"), m.get("document_month")) for m in metas}
    periods.discard((None, None))
    names = {m.get("document_name") for m in metas}
    if len(names) >= 2:
        return False, periods, names
    if len(periods) >= 2:
        return False, periods, names
    # Same file, same (year, month) or all-null dates: try to widen.
    return True, periods, names


def _supplement_comparison_coverage(
    selected: list[tuple[float, int]],
    order: list[tuple[float, int]],
    by_key: dict[int, dict],
    *,
    k_final: int,
    company_hinted: set[str],
) -> list[tuple[float, int]]:
    """Ensure at least two distinct decks or (year, month) periods when possible."""
    metas = [by_key[k] for _, k in selected if k in by_key]
    need_more, periods, doc_names = _comparison_needs_more_diversity(metas)
    if not need_more:
        return selected

    selected_keys = {k for _, k in selected}
    old_years = {m.get("document_year") for m in metas if m.get("document_year") is not None}

    def _consider(meta: dict) -> bool:
        if company_hinted and meta.get("company_name") not in company_hinted:
            return False
        y, mo = meta.get("document_year"), meta.get("document_month")
        dn = meta.get("document_name")
        if dn and dn not in doc_names:
            return True
        if y is not None and y not in old_years:
            return True
        if y is not None and (y, mo) not in periods and (y, mo) != (None, None):
            return True
        return False

    extra: tuple[float, int] | None = None
    for score, ck in order:
        if ck in selected_keys:
            continue
        meta = by_key.get(ck)
        if meta is None or not _consider(meta):
            continue
        extra = (float(score), ck)
        break

    if extra is None:
        return selected

    new_list = list(selected)
    if len(new_list) < k_final:
        new_list.append(extra)
        return new_list[:k_final]

    doc_ct = Counter(m.get("document_name") for m in metas if m.get("document_name"))
    dom_doc = doc_ct.most_common(1)[0][0] if doc_ct else None
    drop_pool = [
        i
        for i, (_, k) in enumerate(selected)
        if dom_doc is None or by_key.get(k, {}).get("document_name") == dom_doc
    ]
    if not drop_pool:
        drop_pool = list(range(len(selected)))
    drop_i = min(drop_pool, key=lambda i: selected[i][0])
    new_list = [selected[i] for i in range(len(selected)) if i != drop_i] + [extra]
    return new_list[:k_final]


def _matches_company_hint(query: str, company_name: str) -> bool:
    q = query.lower()
    cn = company_name.lower().strip()
    if not cn:
        return False
    # Avoid cross-REIT confusion: "Digital Realty" vs "Realty Income" share "realty"
    # (company_name is sometimes truncated, e.g. "Realty Incom").
    if "digital realty" in q and "realty" in cn and "digital" not in cn:
        return False
    if "realty income" in q and "digital realty" in cn:
        return False
    if cn in q:
        return True
    tokens = [t for t in re.split(r"\W+", cn) if len(t) > 2]
    return any(t in q for t in tokens[:2])


def diversified_retrieve(
    query: str,
    final_k: int | None = None,
    client_label: str | None = None,
) -> list[dict]:
    """
    1) Vector search (pgvector or FAISS) for top candidates.
    2) Greedy rerank: cap chunks per document_name; optional cross-company swap.
    """
    k_final = final_k or RETRIEVAL_FINAL_MAX
    k_final = max(RETRIEVAL_FINAL_MIN, min(RETRIEVAL_FINAL_MAX, k_final))

    intent = classify_query_intent(query)
    cand_limit = min(56, max(32, RETRIEVAL_CANDIDATES * 2)) if intent["comparison"] else RETRIEVAL_CANDIDATES

    qv = embed_texts([query])
    if qv.shape[0] == 0:
        return []

    order = vector_similarity_search(qv, cand_limit, client_label=client_label)
    if not order:
        return []

    chunk_keys = [int(k) for _, k in order]
    by_key = get_chunks_by_keys(chunk_keys)
    if not by_key:
        return []

    order = _rerank_with_lexical(query, order, by_key)

    query_year, query_month = _extract_query_date(query)
    trend_mode = intent["trend"]
    if intent["comparison"]:
        # Do not pin to a single month/year; "Mar 2026 vs Dec 2025" would otherwise drop one side.
        query_year, query_month = None, None
    company_hinted = {m["company_name"] for m in by_key.values() if _matches_company_hint(query, m["company_name"])}

    max_recency_units = 1
    for m in by_key.values():
        max_recency_units = max(
            max_recency_units,
            (m.get("document_year") or 0) * 12 + (m.get("document_month") or 0),
        )

    def _recency_boost(meta: dict) -> float:
        u = (meta.get("document_year") or 0) * 12 + (meta.get("document_month") or 0)
        return RECENCY_BOOST_MAX * (float(u) / float(max_recency_units))

    struct_boost = 0.04 if intent["comparison"] else 0.03

    target_versions_by_company: dict[str, tuple[int | None, int | None]] = {}
    if not trend_mode and not intent["comparison"]:
        company_dates: dict[str, list[tuple[int | None, int | None]]] = defaultdict(list)
        for m in by_key.values():
            if company_hinted and m["company_name"] not in company_hinted:
                continue
            company_dates[m["company_name"]].append((m.get("document_year"), m.get("document_month")))
        for co, pairs in company_dates.items():
            valid = [(y or 0, mo or 0) for y, mo in pairs]
            if valid:
                best = max(valid)
                target_versions_by_company[co] = (best[0] or None, best[1] or None)

    per_doc: dict[str, int] = defaultdict(int)
    selected: list[tuple[float, int]] = []

    for score, ck in order:
        ck = int(ck)
        meta = by_key.get(ck)
        if meta is None:
            continue
        if company_hinted and meta["company_name"] not in company_hinted:
            continue
        if not _chunk_passes_temporal_filter(
            meta,
            query_year=query_year,
            query_month=query_month,
            trend_mode=trend_mode,
            target_versions_by_company=target_versions_by_company,
        ):
            continue
        doc_key = meta["document_name"]
        if per_doc[doc_key] >= MAX_CHUNKS_PER_DOCUMENT_IN_BATCH:
            continue
        score = float(score) + _recency_boost(meta)
        if meta.get("is_structured"):
            score = score + struct_boost
        selected.append((score, ck))
        per_doc[doc_key] += 1
        if len(selected) >= k_final:
            break

    if len(selected) < RETRIEVAL_FINAL_MIN:
        for score, ck in order:
            ck = int(ck)
            if any(ck == s[1] for s in selected):
                continue
            meta = by_key.get(ck)
            if meta is None:
                continue
            if company_hinted and meta["company_name"] not in company_hinted:
                continue
            if not _chunk_passes_temporal_filter(
                meta,
                query_year=query_year,
                query_month=query_month,
                trend_mode=trend_mode,
                target_versions_by_company=target_versions_by_company,
            ):
                continue
            score = float(score) + _recency_boost(meta)
            if meta.get("is_structured"):
                score = score + struct_boost
            selected.append((score, ck))
            if len(selected) >= RETRIEVAL_FINAL_MIN:
                break

    if intent["comparison"]:
        for _ in range(4):
            prev = selected
            selected = _supplement_comparison_coverage(
                selected,
                order,
                by_key,
                k_final=k_final,
                company_hinted=company_hinted,
            )
            if selected == prev:
                break

    if len(selected) >= 4 and trend_mode:
        metas = [by_key[s[1]] for s in selected if s[1] in by_key]
        companies = {m["company_name"] for m in metas}
        if len(companies) == 1:
            only_co = next(iter(companies))
            selected_keys = {s[1] for s in selected}
            for score, ck in order:
                ck = int(ck)
                if ck in selected_keys:
                    continue
                meta = by_key.get(ck)
                if meta is None or meta["company_name"] == only_co:
                    continue
                weakest = min(selected, key=lambda x: x[0])
                if weakest[0] < score:
                    selected = [s for s in selected if s[1] != weakest[1]]
                    selected.append((score, ck))
                break

    results: list[dict] = []
    for score, ck in selected:
        m = by_key.get(ck)
        if m is None:
            continue
        row = dict(m)
        row["score"] = float(score)
        results.append(row)

    if _is_numeric_or_metric_query(query) or intent["comparison"]:
        cl = client_label if (client_label and str(client_label).strip()) else None
        mlim = min(8 if intent["comparison"] else 6, k_final)
        mrows = match_metrics_for_query(query, cl, limit=mlim)
        synth = _synthetic_metric_chunks(mrows)
        take_vec = max(0, k_final - len(synth))
        results = synth + results[:take_vec]

    return results
