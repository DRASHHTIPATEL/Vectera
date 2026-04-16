# Speaker script — Vectera RAG (what to say on call / recording)

Read naturally; pause on **bold** for emphasis. ~8–12 minutes with live app.

---

## 1. Hook (20 sec)

“I built a **Retrieval-Augmented Generation** pipeline over **investment PDFs**. Users ask in plain English; the system **retrieves** relevant chunks from an **indexed** corpus, then an **LLM** writes an answer that stays **grounded** in those chunks—with **citations**, **version labels**, and a dedicated **Conflicts** section so we don’t silently merge disagreeing numbers across decks.”

---

## 2. What the app does (45 sec)

- **Offline:** PDFs go through **extract → chunk → embed → save** (`python ingest.py ./folder`). Metadata includes **company**, **version**, optional **client**.
- **Online:** **Streamlit** only **embeds the question**, **searches** the vector store, and calls the **LLM**. No re-ingestion on every click.
- **Trust:** Expand **Retrieved context** to see **exact snippets** and scores—not just the final paragraph.

---

## 3. Changes I made (main story — ~2 min)

Say this as *your* work:

1. **Split indexing from the UI**  
   Earlier you could upload inside Streamlit; I moved that to **`ingest.py`**. That matches **production thinking**: batch workers build the index; the **app tier** only serves queries.

2. **Clarified where GPT / LLM is used**  
   **Chunking** and **document embeddings** are **deterministic code** plus **local sentence-transformers**—no cloud API for vectors. The **LLM** runs only in **`rag.py`** for the **final answer**. I documented that in the **README** table so nobody assumes “GPT indexed everything.”

3. **Free local path**  
   If there’s **no `OPENAI_API_KEY`**, config **defaults to Ollama** on the machine. **README** and **`.env.example`** spell out: Ollama + local embeddings + SQLite/FAISS = **zero API cost** for a laptop demo.

4. **Scalability story in the repo**  
   Added **`scripts/pgvector_hnsw.sql`** and README sections on **Postgres + pgvector**, **HNSW**, **worker/queue ingest**, and **object storage** for PDFs at larger scale—not full production, but **clear direction**.

5. **Hardening**  
   **`rag.py`** catches **connection errors** so a dead Ollama doesn’t crash Streamlit with a raw traceback. **Sources** in the prompt are tightened so citations read like **real references**, not internal debug strings.

6. **Small hygiene**  
   **`.gitignore`** excludes local tooling paths and build artifacts; **`data/`** stays local—index never committed.

---

## 4. Main features — brief / requirements (~2 min)

| They asked for | How I hit it |
|----------------|--------------|
| **Python** | Whole stack in Python. |
| **Database layer** (Snowflake *or* equivalent) | **Postgres + pgvector** when `DATABASE_URL` is set; **SQLite + FAISS** for local—same code paths via **`persistence.py`**. |
| **PDF → text** | **`pdfplumber`** per page; tables flattened to text when possible. |
| **Chunking** | Structure-first: paragraphs, sentence-ish splits, **overlap** so cuts aren’t arbitrary. |
| **Embedding + retrieval** | **Local** embeddings; **vector search** + **per-document cap** + optional **cross-company** swap in **`retrieval.py`**. |
| **LLM answers** | OpenAI-compatible client; **Ollama** for free local. |
| **Citations** | Prompt enforces **Sources** + inline traceability; UI shows **Retrieved context**. |
| **Non-CLI UI** | **Streamlit** — question, chunk slider, Answer, expander. |
| **Version awareness** | **Version** stored on every chunk at ingest; prompt forces attribution by version. |
| **Conflicts** | Prompt: **don’t blend** conflicting facts; **Conflicts** section. |
| **Charts / tables** | Tables extracted when **`pdfplumber`** works; **chart_note** when page is image-heavy; prompt says don’t invent chart data. |
| **Multi-client (optional)** | **`client_label`** on documents + sidebar filter—lightweight, not full RBAC. |

**One line:** “Core RAG loop plus the **messy real-world** bits: versions, conflicts, and honest limits on charts.”

---

## 5. Scalability (~1 min)

**Today:** Single machine, **SQLite + FAISS**, **`ingest.py`** from CLI—fine for the take-home and ~10 decks.

**How this *pattern* scales:**

- **Index once, query many** — heavy work is **not** in the user request path.
- **Postgres + pgvector** for metadata + vectors at real volume.
- **ANN index** (e.g. **HNSW** — we ship **`pgvector_hnsw.sql`**) so search isn’t scanning every row.
- **Workers / queues** running the same ingest pipeline; **object storage** for PDF bytes; **stateless** Streamlit or API replicas in front.

**Sound bite:** “I’m not claiming this repo is Netflix scale—I’m showing **separation of concerns** and a **documented** path to managed DB + batch ingest.”

---

## 6. Explainability (~30 sec)

“Grounding isn’t faith-based: every answer is supposed to tie to **Sources**, and you can open **Retrieved context** to match claims to **chunk text**. Versions and conflict handling are **prompt-level** guardrails on top of that.”

---

## 7. Live demo — say this while clicking (~2 min)

1. **GitHub:** Open **README** → point at **two-phase** flow and **LLM vs indexing** table.
2. **Sidebar:** “These docs were indexed with **`ingest.py`**; company and version labels came from my CLI flags.”
3. **Ask:** e.g. *“Compare Digital Realty **2025-12** vs **2026-03** on bookings or demand—cite sources.”*
4. **Expand Retrieved context:** “Here’s what the model actually saw.”
5. **Optional second ask:** *“Any tension between Public Storage **company-update-mar-26** and **merger-mar-2026**?”* — highlight **Conflicts** if populated.

---

## 8. Close (15 sec)

“Limitations are in the README—no OCR, generic embeddings, SQLite path isn’t multi-tenant production. With more time I’d add **hybrid BM25 + vector** and a small **eval set**. Happy to go deeper on any layer.”

---

## Cheat sheet — one breath each

- **Change:** Ingest **offline** (`ingest.py`); app **query-only**.  
- **Feature:** RAG + **citations** + **versions** + **conflicts** + **chart honesty**.  
- **Scale:** Two-phase + **Postgres/pgvector** + **HNSW script** + worker story.  
- **Proof:** **Retrieved context** expander + **Sources** in the answer.
