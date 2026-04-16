"""
Streamlit UI: query pre-indexed investment PDFs — retrieval + grounded answers with citations.

Documents are indexed offline via ingest.py; this app does not chunk or embed documents.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import DATABASE_URL, OPENAI_BASE_URL, OPENAI_MODEL, USE_OLLAMA, ensure_dirs, use_postgres
from src.persistence import backend_name, list_documents, list_metrics_for_client, match_metrics_for_query
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
ask = st.button("Answer", type="primary")

client_for_retrieval = client.strip() if client and client.strip() else None

if ask and q.strip():
    with st.spinner("Retrieving and reasoning…"):
        chunks = diversified_retrieve(q.strip(), final_k=n_sources, client_label=client_for_retrieval)
        ans = answer_question(q.strip(), chunks)
        query_metric_hits = match_metrics_for_query(q.strip(), client_for_retrieval, limit=12)

    st.caption(
        "Heuristic metrics and OCR are **best-effort**. Values tagged **low** confidence or **ocr** "
        "should be verified against the source PDF."
    )
    st.markdown(ans)

    structured_metrics = _collect_structured_metrics(chunks)
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
