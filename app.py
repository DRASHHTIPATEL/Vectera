"""
Streamlit UI: query pre-indexed investment PDFs — retrieval + grounded answers with citations.

Documents are indexed offline via ingest.py; this app does not chunk or embed documents.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import DATABASE_URL, OPENAI_BASE_URL, OPENAI_MODEL, USE_OLLAMA, ensure_dirs, use_postgres
from src.persistence import backend_name, list_documents
from src.rag import answer_question
from src.retrieval import diversified_retrieve

st.set_page_config(page_title="Vectera — Investment Doc RAG", layout="wide")
ensure_dirs()

st.title("Vectera")
st.caption(
    "Ask questions over **pre-indexed** PDFs — answers use retrieved text only, with sources. "
    "Index documents offline: `python ingest.py ./your_pdf_folder`."
)

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

    st.markdown(ans)

    with st.expander("Retrieved context — what the model actually saw"):
        for c in chunks:
            st.markdown(
                f"**{c['document_name']}** · page {c['page_number']} · "
                f"{c['company_name']} · `{c['version']}` · score `{c.get('score', 0):.3f}`"
            )
            st.text(c["chunk_text"][:4000] + ("…" if len(c["chunk_text"]) > 4000 else ""))
            st.divider()

elif ask:
    st.warning("Enter a question first.")
