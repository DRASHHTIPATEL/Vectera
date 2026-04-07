"""
Streamlit UI: upload investment PDFs (with company + version + client metadata), ask questions,
view grounded answers with citations.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import DATABASE_URL, OPENAI_BASE_URL, OPENAI_MODEL, USE_OLLAMA, ensure_dirs
from src.persistence import backend_name, list_documents
from src.pipeline import ingest_pdf
from src.rag import answer_question
from src.retrieval import diversified_retrieve

st.set_page_config(page_title="Vectera — Investment Doc RAG", layout="wide")
ensure_dirs()

st.title("Vectera")
st.caption("Ask questions over your PDFs — answers use retrieved text only, with sources. "
           "Set company + version when you upload so multi-version decks stay comparable.")

with st.sidebar:
    st.header("Ingest documents")
    client = st.text_input(
        "Client / workspace",
        value="default",
        help="Optional scope for multi-client demos. Retrieval and the document list respect this when set. "
        "Clear to see all clients (Postgres only).",
    )
    company = st.text_input("Company name", placeholder="e.g. Acme Corp", help="Used to group filings and versions.")
    version = st.text_input(
        "Document version / label",
        value="v1",
        help='e.g. "2024-10-K", "Q3-2024", "v2". Stored on every chunk for version-aware answers.',
    )
    files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Index uploaded PDFs", type="primary") and files:
        progress = st.progress(0.0, text="Starting…")
        cl = (client or "default").strip() or "default"
        for i, f in enumerate(files):
            data = f.getvalue()
            try:
                result = ingest_pdf(data, f.name, company or "Unknown", version or "v1", client_label=cl)
                st.success(f"{f.name}: {result['message']}")
            except Exception as e:
                st.error(f"{f.name}: {e}")
            progress.progress((i + 1) / len(files), text=f"Done {i + 1}/{len(files)}")
        progress.empty()

    st.divider()
    st.subheader("Storage")
    st.caption(f"**Backend:** `{backend_name()}`")
    if USE_OLLAMA:
        st.caption(f"**LLM:** Ollama — model `{OPENAI_MODEL}` @ `{OPENAI_BASE_URL}`")
    else:
        st.caption(f"**LLM:** OpenAI-compatible API — `{OPENAI_MODEL}`")
    if DATABASE_URL:
        st.caption("Connected via `DATABASE_URL` (Postgres + pgvector).")
    else:
        st.caption("Running without **DATABASE_URL** — **SQLite** + **FAISS** on disk. "
                   "That’s normal for local dev; use Postgres + `DATABASE_URL` when you want the full DB path.")

    st.divider()
    st.subheader("Indexed documents")
    cl_filter = client.strip() if client and client.strip() else None
    docs = list_documents(client_label=cl_filter)
    if not docs:
        st.info("No documents yet.")
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
