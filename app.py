"""
Streamlit UI: query pre-indexed investment PDFs — retrieval + grounded answers with citations.

Documents are indexed offline via ingest.py; this app does not chunk or embed documents.
"""

from __future__ import annotations

import hashlib
import re
import sys
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import DATABASE_URL, OPENAI_BASE_URL, OPENAI_MODEL, USE_OLLAMA, ensure_dirs, use_postgres
from src.persistence import (
    backend_name,
    get_document_stored_path,
    list_chart_chunks,
    list_documents,
    list_metrics_for_client,
    match_metrics_for_query,
)
from src.pdf_render import render_pdf_page_png_bytes
from src.rag import answer_question
from src.retrieval import diversified_retrieve

st.set_page_config(page_title="Vectera — Investment Doc RAG", layout="wide")
ensure_dirs()

st.title("Vectera")
st.caption(
    "Ask questions over **pre-indexed** PDFs — answers use retrieved text only, with sources. "
    "Index documents offline: `python ingest.py ./your_pdf_folder`."
)


def _extract_metric_pairs(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in re.finditer(r"([A-Za-z][A-Za-z \-/()]{2,40})[:\-]\s*([$]?\d[\d,]*(?:\.\d+)?%?)", text):
        key = re.sub(r"\s+", " ", m.group(1)).strip()
        val = m.group(2).strip()
        if key and val:
            out.append((key, val))
    return out


def _collect_structured_metrics(chunks: list[dict]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    metrics: list[tuple[str, str]] = []
    for c in chunks:
        if c.get("from_stored_metrics") and c.get("metric_display_name") and c.get("metric_display_value"):
            kv = (str(c["metric_display_name"])[:48], str(c["metric_display_value"]))
            if kv not in seen:
                seen.add(kv)
                metrics.append(kv)
            continue
        if not c.get("is_structured"):
            continue
        for kv in _extract_metric_pairs(c.get("chunk_text", "")):
            if kv in seen:
                continue
            seen.add(kv)
            metrics.append(kv)
    return metrics[:9]


def _pct_or_float(s: str) -> float | None:
    t = (s or "").strip().replace(",", "")
    m = re.match(r"^(\d+(?:\.\d+)?)\s*%$", t)
    if m:
        return float(m.group(1))
    m = re.match(r"^\$?([\d.]+)\s*([BMK])?$", t)
    if m:
        n = float(m.group(1))
        mult = {"B": 1e9, "M": 1e6, "K": 1e3}.get(m.group(2) or "", 1)
        return n * mult
    return None


def _chunk_is_chart_like(c: dict) -> bool:
    st = (c.get("structured_type") or "").lower()
    src = (c.get("source_type") or "").lower()
    if st == "chart" or src == "ocr":
        return True
    if c.get("chart_note"):
        return True
    ct = c.get("chunk_text") or ""
    return "[CHART_OR_FIGURE_OCR]" in ct or "[OCR_EXTRACTED]" in ct


def _ocr_scan_numbers_for_bar_chart(text: str) -> dict[str, float]:
    """Parse chart OCR hints; prefer explicit series, avoid axis-tick bars."""
    sm = re.search(r"\[CHART_OCR_SERIES\]\s*([^\n]+)", text)
    if sm:
        out_series: dict[str, float] = {}
        for item in [p.strip() for p in sm.group(1).split(";") if p.strip()]:
            mm = re.match(r"([A-Za-z0-9 _-]{1,24})\s*=\s*([\d.]+)\s*%?", item)
            if not mm:
                continue
            try:
                out_series[mm.group(1).strip()] = float(mm.group(2))
            except ValueError:
                continue
        if len(out_series) >= 2:
            return out_series

    m = re.search(r"\[CHART_OCR_NUMBERS_SCAN\]\s*([^\n]+)", text)
    if not m:
        return {}
    parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
    out: dict[str, float] = {}
    for i, p in enumerate(parts[:24]):
        v = _pct_or_float(p)
        if v is None:
            try:
                clean = re.sub(r"[^\d.\-]", "", p)
                if clean:
                    v = float(clean)
            except ValueError:
                continue
        if v is None:
            continue
        # Ignore likely axis ticks when we have richer values.
        if abs(v - round(v)) < 1e-9 and (int(round(v)) % 5 == 0):
            continue
        label = p if len(p) <= 28 else p[:25] + "…"
        out[f"{i + 1}. {label}"] = float(v)
    if out:
        return out
    # Fallback: keep behavior if everything got filtered.
    for i, p in enumerate(parts[:24]):
        v = _pct_or_float(p)
        if v is None:
            continue
        label = p if len(p) <= 28 else p[:25] + "…"
        out[f"{i + 1}. {label}"] = float(v)
    return out


def _group_chart_chunks(chunks: list[dict]) -> "OrderedDict[tuple[str, int], list[dict]]":
    groups: OrderedDict[tuple[str, int], list[dict]] = OrderedDict()
    for c in chunks:
        if not _chunk_is_chart_like(c):
            continue
        key = (c["document_name"], int(c["page_number"]))
        if key not in groups:
            groups[key] = []
        groups[key].append(c)
    return groups


_CHART_QUERY_STOP = {
    "the",
    "a",
    "an",
    "what",
    "which",
    "show",
    "display",
    "from",
    "this",
    "that",
    "with",
    "chart",
    "graph",
    "figure",
    "slide",
    "pdf",
    "page",
}


def _query_tokens_for_chart_finder(query: str) -> list[str]:
    toks = [t for t in re.split(r"\W+", (query or "").lower()) if len(t) > 2]
    return [t for t in toks if t not in _CHART_QUERY_STOP]


def _is_chart_request(query: str) -> bool:
    ql = (query or "").lower()
    if any(k in ql for k in ("chart", "graph", "figure", "plot", "bar", "line chart", "slide")):
        return True
    # Common analyst phrasing that still means "show chart-like evidence".
    if ("show" in ql or "display" in ql) and ("trend" in ql or "over time" in ql):
        return True
    return False


def _chart_match_score(query: str, chunk: dict) -> float:
    ql = (query or "").lower()
    text = " ".join(
        [
            str(chunk.get("chunk_text") or "").lower(),
            str(chunk.get("document_name") or "").lower(),
            str(chunk.get("company_name") or "").lower(),
            str(chunk.get("version") or "").lower(),
        ]
    )
    score = 0.0
    if chunk.get("structured_type") == "chart":
        score += 0.6
    if chunk.get("source_type") == "ocr" or chunk.get("ocr_low_confidence"):
        score += 0.25
    if "[chart_ocr_series]" in text:
        score += 0.5
    if chunk.get("chart_note"):
        score += 0.15

    tokens = _query_tokens_for_chart_finder(query)
    for t in tokens[:14]:
        if t in text:
            score += 0.09
    if _is_chart_request(ql):
        score += 0.06
    return score


def _find_chart_chunks_for_query(query: str, client_label: str | None, limit_pages: int = 6) -> list[dict]:
    rows = list_chart_chunks(client_label=client_label, limit=1500)
    if not rows:
        return []
    by_page: dict[tuple[str, int], tuple[float, dict]] = {}
    for r in rows:
        key = (str(r.get("document_name") or ""), int(r.get("page_number") or 0))
        if not key[0] or key[1] <= 0:
            continue
        s = _chart_match_score(query, r)
        prev = by_page.get(key)
        if prev is None or s > prev[0]:
            by_page[key] = (s, r)
    ranked = sorted(by_page.values(), key=lambda x: x[0], reverse=True)
    picked: list[dict] = []
    for s, r in ranked:
        if s <= 0 and _is_chart_request(query):
            # Keep recent chart pages for broad "show chart" asks.
            pass
        elif s <= 0:
            continue
        picked.append(r)
        if len(picked) >= limit_pages:
            break
    return picked


def _render_charts_and_ocr_panel(chunks: list[dict], client_label: str | None) -> None:
    groups = _group_chart_chunks(chunks)
    if not groups:
        return

    st.subheader("Charts & OCR — retrieved excerpts")
    st.caption(
        "**Left:** page rendered from the indexed PDF. **Right:** text stored for this chunk (includes OCR when used). "
        "Numbers are approximate; verify material figures in the source file."
    )

    for (doc_name, page_num), group in groups.items():
        merged = "\n\n---\n\n".join((x.get("chunk_text") or "").strip() for x in group if (x.get("chunk_text") or "").strip())
        note = next((x.get("chart_note") for x in group if x.get("chart_note")), None)
        ver = group[0].get("version") or "—"
        uid = hashlib.sha256(f"{doc_name}:{page_num}".encode()).hexdigest()[:16]

        st.markdown(f"##### {doc_name} · page {page_num} · `{ver}`")

        stored = get_document_stored_path(doc_name, client_label)
        png_bytes: bytes | None = None
        if stored:
            pth = Path(stored)
            if not pth.is_file():
                alt = ROOT / stored
                pth = alt if alt.is_file() else pth
            if pth.is_file():
                png_bytes = render_pdf_page_png_bytes(pth, page_num)

        c1, c2 = st.columns([1, 1])
        with c1:
            if png_bytes:
                st.image(BytesIO(png_bytes), caption=f"Rendered page {page_num}", use_container_width=True)
            else:
                st.caption("Page preview unavailable (missing file on disk or render error).")
        with c2:
            if note:
                st.info(note)
            st.text_area(
                "Chunk text / OCR",
                value=merged[:12000] + ("…" if len(merged) > 12000 else ""),
                height=360,
                key=f"ocr_txt_{uid}",
            )

        nums = _ocr_scan_numbers_for_bar_chart(merged)
        if len(nums) >= 2:
            st.caption("OCR numeric scan (ingestion heuristic — not axis labels)")
            st.bar_chart(nums)
        elif len(nums) == 1:
            only_k, only_v = next(iter(nums.items()))
            st.metric(label="Single OCR token (from scan line)", value=f"{only_v:g}", help=only_k)

        st.divider()


_faiss_ntotal: int | None = None
if not use_postgres():
    from src.faiss_store import load_index

    _idx = load_index()
    _faiss_ntotal = int(_idx.ntotal) if _idx is not None else 0

with st.sidebar:
    st.header("Index status")
    st.caption(
        "Ingestion runs **outside** this app. From the repo root:\n\n`python ingest.py ./data`"
    )
    client = st.text_input(
        "Client / workspace (filter)",
        value="default",
        help="Optional scope: only documents with this client_label are listed (and used for retrieval "
        "when non-empty). Clear to list all clients (Postgres only).",
    )

    st.divider()
    st.subheader("Storage")
    st.caption(f"**Backend:** `{backend_name()}`")
    if not use_postgres():
        st.caption(
            f"**FAISS vectors on disk:** `{_faiss_ntotal if _faiss_ntotal is not None else '—'}` "
            "(loaded at startup; not rebuilt here.)"
        )
    if USE_OLLAMA:
        st.caption(f"**LLM:** Ollama — model `{OPENAI_MODEL}` @ `{OPENAI_BASE_URL}`")
    else:
        st.caption(f"**LLM:** OpenAI-compatible API — `{OPENAI_MODEL}`")
    if DATABASE_URL:
        st.caption("Connected via `DATABASE_URL` (Postgres + pgvector). Embeddings live in the database.")
    else:
        st.caption("**SQLite** + **FAISS** under `data/` — index built by `ingest.py`, reused every session.")

    st.divider()
    st.subheader("Indexed documents")
    cl_filter = client.strip() if client and client.strip() else None
    docs = list_documents(client_label=cl_filter)
    if not docs:
        st.info(
            "No documents indexed yet. Run `python ingest.py ./path/to/pdfs` then refresh this page."
        )
    else:
        for d in docs[:25]:
            clab = d.get("client_label", "")
            st.write(
                f"**{d['document_name']}** — {d['company_name']} / `{d['version']}` "
                f"({d['page_count']} pg)" + (f" · client `{clab}`" if clab else "")
            )

st.subheader("Ask a question")
q = st.text_area(
    "Question",
    placeholder="What were the main risk factors called out?",
    height=100,
)
col_a, col_b = st.columns([1, 4])
with col_a:
    n_sources = st.slider("Chunks to retrieve", min_value=5, max_value=8, value=6)
with col_b:
    chart_finder_mode = st.checkbox(
        "Chart Finder mode (prioritize chart-page display)",
        value=False,
        help="When enabled, the app also searches chart/OCR chunks directly and displays top matching chart pages.",
    )
ask = st.button("Answer", type="primary")

client_for_retrieval = client.strip() if client and client.strip() else None

if ask and q.strip():
    with st.spinner("Retrieving and reasoning…"):
        chunks = diversified_retrieve(q.strip(), final_k=n_sources, client_label=client_for_retrieval)
        chart_chunks: list[dict] = []
        if chart_finder_mode or _is_chart_request(q.strip()):
            chart_chunks = _find_chart_chunks_for_query(
                q.strip(),
                client_for_retrieval,
                limit_pages=max(4, n_sources),
            )
        answer_chunks = chart_chunks if chart_chunks else chunks
        ans = answer_question(q.strip(), answer_chunks)
        query_metric_hits = match_metrics_for_query(q.strip(), client_for_retrieval, limit=12)

    st.caption(
        "Heuristic metrics and OCR are **best-effort**. Values tagged **low** confidence or **ocr** "
        "should be verified against the source PDF."
    )
    st.markdown(ans)

    panel_chunks = answer_chunks
    _render_charts_and_ocr_panel(panel_chunks, client_for_retrieval)
    if (chart_finder_mode or _is_chart_request(q.strip())) and not _group_chart_chunks(panel_chunks):
        st.info("No chart-like pages matched this query in the indexed corpus.")

    structured_metrics = _collect_structured_metrics(answer_chunks)
    if structured_metrics:
        st.subheader("Key metrics (retrieval)")
        cols = st.columns(min(3, len(structured_metrics)))
        for i, (k, v) in enumerate(structured_metrics[:6]):
            cols[i % len(cols)].metric(k[:36], v)

    if query_metric_hits:
        st.subheader("Structured metrics (this question)")
        subcols = st.columns(min(3, len(query_metric_hits)))
        for i, row in enumerate(query_metric_hits[:9]):
            conf = row.get("confidence", "medium")
            stype = row.get("source_type", "text")
            nv = row.get("normalized_value")
            help_txt = (
                f"Confidence: {conf} (high=table, medium=text, low=OCR) · source: {stype} · "
                f"page {row.get('page_number')}"
                + (f" · normalized={nv}" if nv is not None else "")
            )
            subcols[i % len(subcols)].metric(
                str(row.get("metric_name", ""))[:32],
                str(row.get("value", "")),
                help=help_txt,
            )

    all_client_metrics = list_metrics_for_client(client_for_retrieval, limit=400)
    names = sorted({m["metric_name"] for m in all_client_metrics})
    if len(names) >= 1 and len(all_client_metrics) >= 2:
        pick = st.selectbox("Compare metric across versions", options=[""] + names, key="cmp_metric")
        if pick:
            sub = [m for m in all_client_metrics if m["metric_name"] == pick]
            sub.sort(key=lambda r: ((r.get("document_year") or 0), (r.get("document_month") or 0), r.get("version") or ""))
            st.dataframe(
                [
                    {
                        "company": r["company_name"],
                        "version": r["version"],
                        "value": r["value"],
                        "normalized": r.get("normalized_value"),
                        "unit": r.get("unit"),
                        "confidence": r["confidence"],
                        "source": r["source_type"],
                        "page": r["page_number"],
                    }
                    for r in sub
                ],
                use_container_width=True,
            )
            series: dict[str, float] = {}
            for r in sub:
                v = r.get("normalized_value")
                if v is None:
                    v = _pct_or_float(str(r.get("value", "")))
                if v is None:
                    continue
                label = r.get("version") or "?"
                series[label] = float(v)
            if len(series) >= 2:
                st.caption("Values by version (numeric parse of stored strings — verify against the PDF).")
                st.bar_chart(series)

    versions = sorted(
        {(c.get("version"), c.get("document_year"), c.get("document_month")) for c in chunks},
        key=lambda x: ((x[1] or 0), (x[2] or 0), x[0] or ""),
    )
    if len(versions) > 1:
        st.subheader("Version coverage (retrieved chunks)")
        rows = []
        for v, y, m in versions:
            n = sum(1 for c in chunks if c.get("version") == v)
            rows.append({"version": v, "year": y, "month": m, "chunks": n})
        st.dataframe(rows, use_container_width=True)
        st.bar_chart(data={r["version"]: r["chunks"] for r in rows}, use_container_width=True)

    with st.expander("Retrieved context — what the model actually saw"):
        for c in chunks:
            src = c.get("source_type") or "—"
            conf = c.get("confidence") or ("low" if c.get("ocr_low_confidence") else "—")
            tag = f" · `{src}`"
            if c.get("ocr_low_confidence") or src == "ocr":
                tag += " · **OCR / verify**"
            st.markdown(
                f"**{c['document_name']}** · page {c['page_number']} · "
                f"{c['company_name']} · `{c['version']}` · score `{c.get('score', 0):.3f}`"
                + (" · structured" if c.get("is_structured") else "")
                + tag
                + (f" · conf `{conf}`" if conf != "—" else "")
            )
            st.text(c["chunk_text"][:4000] + ("…" if len(c["chunk_text"]) > 4000 else ""))
            st.divider()

elif ask:
    st.warning("Enter a question first.")
