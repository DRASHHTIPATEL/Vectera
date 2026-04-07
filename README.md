# Vectera — technical assessment (RAG over investment PDFs)

I built this for the **Vectera.ai RAG technical assessment**: a working **Streamlit** app that ingests PDFs, chunks and embeds them, stores everything in a **real database layer**, retrieves diversified context, and uses an **LLM** only for answers that stay **grounded in retrieved text** with **citations**. I also wired in **version labels**, **conflict-aware prompting**, and honest handling of **charts/tables**.

This is intentionally **not** production infrastructure — it is meant to show how I think about ambiguity, retrieval, and tradeoffs.

---

## Deliverables checklist

| Deliverable | Where |
|-------------|--------|
| Working UI (not CLI) | `app.py` (Streamlit) |
| Code repository | Your Git remote (GitHub, etc.) — push this folder |
| README (this file) | `README.md` |
| Demo (5–10 min) | `DEMO.md` + record or live session |
| Submission handoff | `SUBMISSION.md` (checklist: repo link, env, what to show) |
| Push to GitHub | `PUSH_TO_GITHUB.md` (commands — requires your account) |

---

## Assessment brief → implementation (traceability)

| Requirement | How it’s covered |
|-------------|------------------|
| Python | Whole project |
| Database (Snowflake or equivalent) | **Postgres + pgvector** when `DATABASE_URL` is set (`src/postgres_db.py`, `docker-compose.yml`). **SQLite + FAISS** fallback without Docker (`src/database.py`, `src/faiss_store.py`). |
| PDF → text ingestion | `src/ingestion.py` |
| Chunking | `src/chunking.py` |
| Embeddings + retrieval | `src/embeddings.py`, `src/retrieval.py`, `src/persistence.py` |
| LLM answers + citations | `src/rag.py` (system prompt + `QUESTION` / `CONTEXT` user message) |
| Streamlit UI (not CLI-only) | `app.py` |
| Version awareness | Version label at upload → stored per chunk → prompt + UI |
| Conflicts / cross-document | Diversified retrieval + prompt rules in `src/rag.py` |
| Charts / tables / limitations | Table extract + chart note heuristic in `src/ingestion.py`; prompt rules 6–7 in `src/rag.py` |
| Optional multi-client | `client_label` / “Client / workspace” in UI + DB filter (Postgres) |
| README narrative | This file |
| Demo | `DEMO.md` |

**LLM prompt (grounding rules):** defined as `SYSTEM_PROMPT` in `src/rag.py` (includes citation format, conflict handling, and rule 7 on not blaming chart extraction unless stated in context).

---

## Official brief checklist (Vectera.ai — full pass/fail)

Below is the assessment text mapped to this repo. **Snowflake** is listed as preferred in the brief; I use **PostgreSQL + pgvector** as an **equivalent database service** (Supabase/RDS/Neon-compatible) with `docker-compose.yml`, and document how this maps to Snowflake. **SQLite + FAISS** is a documented local fallback when `DATABASE_URL` is unset.

### Required technologies

| Brief | Met? | Where |
|-------|------|--------|
| **Python** | Yes | All application code |
| **Database: Snowflake (preferred) OR equivalent (Postgres, Supabase, …)** | Yes | **Postgres + pgvector:** `src/postgres_db.py`, `docker-compose.yml`, `DATABASE_URL`. **Equivalent** to the brief. **SQLite + FAISS:** `src/database.py`, `src/faiss_store.py` — dev fallback only. |
| **Store and retrieve document data using this database layer** | Yes | Chunks + metadata in DB; vectors in Postgres (`embedding` column) or FAISS (SQLite mode). Retrieval via `src/persistence.py`. |

### Functional requirements (core)

| Brief | Met? | Where |
|-------|------|--------|
| Document ingestion (PDF → text) | Yes | `src/ingestion.py` |
| Chunking | Yes | `src/chunking.py` |
| Embedding + retrieval | Yes | `src/embeddings.py`, `src/retrieval.py`, `src/persistence.py` |
| LLM-based answer generation | Yes | `src/rag.py` (OpenAI-compatible API; **Ollama** supported via `.env`) |
| **Citations (required)** | Yes | `SYSTEM_PROMPT` in `src/rag.py` forces **Sources** + inline citations; UI shows answer + **Retrieved context** |
| Responses reference source documents and/or sections | Yes | Document name, page, version in context headers and **Sources** block |

### Application requirement

| Brief | Met? | Where |
|-------|------|--------|
| **Working UI — not CLI-only** | Yes | **Streamlit** `app.py` (recommended path from brief) |

### Key capabilities to demonstrate

| Brief | Met? | How |
|-------|------|-----|
| **1. Version awareness** — avoid blindly mixing conflicting values across versions | Yes | User-set **version** label on ingest → stored per chunk → prompt rules 3–4 in `src/rag.py` |
| **2. Cross-document reasoning & conflicts** — attribution, don’t merge conflicting facts | Yes | Diversified retrieval (`src/retrieval.py`) + prompt rules 2–3, **Conflicts** section in output |
| **3. Charts, tables, structured content** — extract tables OR explain limitations | Yes | `pdfplumber` tables in `src/ingestion.py`; chart/image heuristic + prompt rules 6–7 in `src/rag.py` |

### README required write-up (brief)

| Brief asks for | Section in this file |
|----------------|----------------------|
| System architecture (high-level) | [System architecture](#system-architecture-high-level) |
| How you used the database (Snowflake or equivalent) | [How I used the database](#how-i-used-the-database-snowflake-vs-equivalent) |
| Chunking strategy | [Chunking strategy](#chunking-strategy) |
| Retrieval approach | [Retrieval approach](#retrieval-approach) |
| Versioning | [Version awareness](#version-awareness) |
| Conflicting information | [Conflicting information](#conflicting-information) |
| Charts/tables | [Charts, tables, and structured content](#charts-tables-and-structured-content) |
| Known limitations | [Known limitations](#known-limitations) |
| What you would improve with more time | [What I would improve with more time](#what-i-would-improve-with-more-time) |

### Deliverables (brief)

| Brief | Met? | Where |
|-------|------|--------|
| Working application — ask questions, answers **with citations** | Yes | `app.py` |
| Code repository — setup, **configure database**, environment, **how to run** | Yes | This README + `.env.example` + `PUSH_TO_GITHUB.md` |
| README (important) | Yes | This file |
| Demo (5–10 min), example queries | Yes | `DEMO.md` |

### Example questions (from brief — try in the UI)

- “What is [Company X]’s key strategy?”
- “How has [metric] changed across document versions?”
- “What drives demand according to these materials?”
- “Are there conflicting data points across documents?”
- “Summarize key trends shown in the documents”

### Evaluation criteria (how the design responds)

| Criterion | Addressed by |
|-----------|----------------|
| **Retrieval quality / grounding** | Diversified retrieval + audit expander in UI + strict `SYSTEM_PROMPT` |
| **Complexity: versions, conflicts, chart limits** | Sections above + `src/rag.py`, `src/ingestion.py` |
| **System design / tradeoffs / scale** | Architecture narrative, Postgres vs SQLite, Snowflake mapping, limitations & improvements |
| **Code quality** | Modular `src/` layout, `persistence` abstraction |
| **Communication** | This README + `DEMO.md` |

### Optional (brief)

| Brief | Met? | Where |
|-------|------|--------|
| Documents may belong to different clients — how you’d approach access control | Yes | [Multi-client documents](#multi-client-documents-optional-consideration) — `client_label` + future RLS/OIDC narrative |

### Out of scope (brief says not expected)

Perfect chart extraction, complete conflict resolution, production-grade infra — **explicitly** not claimed; limitations are stated.

---

## System architecture (high-level)

1. **Ingestion** (`src/ingestion.py`) — PDF → text per page with `pdfplumber`; tables flattened to text when extraction succeeds; weak text + images triggers a **chart limitation** note.
2. **Chunking** (`src/chunking.py`) — structure-first splits (paragraphs → sentences), merge small fragments, overlap windows — **not** a single fixed token grinder.
3. **Embeddings** (`src/embeddings.py`) — local `sentence-transformers` so indexing does not require an embedding API.
4. **Persistence** (`src/persistence.py`) — **unified API** over two backends:
   - **Postgres + pgvector** when `DATABASE_URL` is set (this is the path that matches the assessment’s “Snowflake or equivalent” requirement).
   - **SQLite + FAISS** when `DATABASE_URL` is unset — handy for a laptop demo with zero Docker.
5. **Retrieval** (`src/retrieval.py`) — top‑N similarity search, then **per-document caps** and a small **cross-company swap** so one deck cannot dominate.
6. **Reasoning** (`src/rag.py`) — OpenAI-compatible chat API with strict instructions: **context-only**, **no merging conflicts**, **version attribution**, fixed response sections (**Answer / Key Points / Conflicts / Sources**).
7. **UI** (`app.py`) — upload, optional **client/workspace** scope, questions, citations, expandable retrieved chunks.

```mermaid
flowchart LR
  PDF[PDF upload] --> ING[Ingest + tables]
  ING --> CH[Chunk + overlap]
  CH --> EMB[Local embeddings]
  EMB --> DB[(Postgres + pgvector\nor SQLite + FAISS)]
  Q[Question] --> RET[Diversified retrieval]
  DB --> RET
  RET --> LLM[LLM grounded answer]
```

---

## How I used the database (Snowflake vs equivalent)

The brief allows **Snowflake (preferred)** or an **equivalent service**. I implemented **PostgreSQL with the `pgvector` extension** because:

- It satisfies “store and retrieve document data using this database layer” including **vector similarity** in the same store.
- It runs locally via **Docker**, or on **Supabase / RDS / Neon** with the same `DATABASE_URL` — typical for a take-home.

**Snowflake mapping (what I would do with more time and an account):** I would load raw PDFs to a **stage**, parse text in a **Snowpark** Python UDF or external worker, store chunks in a table with **VARIANT** metadata (company, version, client), and use **Cortex Search / vector features** (or an external vector index) for similarity — keeping the same **retrieval + diversification + prompt** shell. The application code would swap the `persistence` implementation for Snowflake’s APIs while keeping the Streamlit UX.

**Schema (Postgres):** tables `documents` and `chunks`; `chunks.embedding` is `vector(384)` by default (must match `EMBEDDING_MODEL` / `EMBEDDING_DIM`).

---

## Chunking strategy

- Split on **blank-line paragraphs** first, then **sentence-like** boundaries for oversized blocks.
- **Merge** undersized pieces so the retriever does not drown in micro-chunks.
- **Overlap** only when a block still exceeds the target window — preserves continuity without blindly overlapping every boundary.

**Tradeoff:** this is cheaper and more deterministic than embedding-based segmentation; it can still split mid-thought on poorly formatted PDFs.

---

## Retrieval approach

- Pull **more candidates** than I show the LLM (`RETRIEVAL_CANDIDATES` in `src/config.py`).
- Greedy select with **`MAX_CHUNKS_PER_DOCUMENT_IN_BATCH`** so answers draw from **multiple PDFs** when the corpus allows.
- Optional **cross-company swap** if every top hit is one company but another company exists in the candidate list.

**Tradeoff:** diversity rules can slightly reduce raw similarity — I prefer that over single-source answers for multi-document questions.

---

## Version awareness

I do **not** infer “which fiscal year” from the PDF automatically (that breaks on messy decks). At upload time you set a **version label** (e.g. `Q3-2024`, `v2`). That value is stored on **every chunk** and passed into the LLM context so it can say **“According to [version] …”** and compare versions without silently merging numbers.

---

## Conflicting information

- **Retrieval** tries to surface multiple relevant chunks (and versions).
- **Prompting** forbids blending contradictory facts; the model must separate **“according to document A … / document B …”** and list conflicts explicitly.

I still treat the LLM as **untrustworthy without audit**: the UI shows **retrieved context** so you can verify grounding.

---

## Charts, tables, and structured content

- **Tables:** `pdfplumber` `extract_tables()` → rendered as text in the chunk when it works.
- **Charts:** often **no reliable numeric extraction** from vector graphics. I emit an explicit **limitation note** on those pages so the system does not hallucinate series data.

---

## Multi-client documents (optional consideration)

I added a **`client_label`** (shown as **Client / workspace** in the UI). In **Postgres**, retrieval and listing can filter by this field — a minimal stand-in for tenant scope. **How I would approach real access control:** JWT/OIDC identity → resolve `tenant_id` → enforce **row-level security** in Postgres (or equivalent in Snowflake), encrypt PDFs at rest, audit query logs, and never mix tenant indexes in one unscoped FAISS file (today’s SQLite fallback is single-tenant unless you split files).

---

## Known limitations

- No **OCR** for scanned decks.
- General-purpose embeddings, not finance-tuned.
- **SQLite + FAISS** path has no server-side RLS; use **Postgres** for assessment-style deployment.
- Changing **embedding model / dimension** requires **re-ingesting** (dimension must match `EMBEDDING_DIM` / DB vector type).

---

## What I would improve with more time

- **Hybrid retrieval** (BM25 + vectors) for tickers, CIKs, and exact phrases.
- **OCR** pipeline for image-only pages.
- **Eval harness** (golden questions + citation overlap metrics).
- **Snowflake**-native persistence behind the same `persistence` interface.
- **IVFFLAT / HNSW** index tuning on `chunks.embedding` for large corpora.

---

## Environment setup

**Python 3.10+** recommended.

### Free LLM (no paid API key)

I support **[Ollama](https://ollama.com)** (runs locally, no cloud bill): install Ollama, run `ollama pull llama3.2`, then set **`USE_OLLAMA=1`** in `.env`. The code calls Ollama’s OpenAI-compatible API at `http://127.0.0.1:11434/v1`. **Embeddings are already local**; only the answer step needs this LLM.

You can also point **`OPENAI_BASE_URL`** at other OpenAI-compatible hosts (some offer free tiers or credits), with **`OPENAI_API_KEY`** as required by that provider.

```bash
cd Vectera
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` (see `.env.example` for all options):

- **`USE_OLLAMA=1`** — local **Ollama** (free); run `ollama serve` and `ollama pull llama3.2`.  
- **`OPENAI_API_KEY`** — use with **`USE_OLLAMA=0`** for OpenAI or another OpenAI-compatible cloud API.
- **`DATABASE_URL`** — **recommended for the assessment** after `docker compose up -d`:

  `postgresql://vectera:vectera@localhost:5432/vectera`

Leave `DATABASE_URL` unset to use **`data/vectera.db` + `data/faiss.index`** (no Docker).

### Postgres + pgvector (recommended)

```bash
docker compose up -d
# wait until healthy, then:
streamlit run app.py
```

**Supabase:** paste the project’s **connection string** as `DATABASE_URL` (add `?sslmode=require` if required). Enable **pgvector** in the database extensions if it is not already on.

### If you upgraded from an older SQLite-only checkout

Delete `data/vectera.db` once so SQLAlchemy can recreate tables with the new `client_label` column (or run your own `ALTER TABLE`).

---

## Run the app

```bash
streamlit run app.py
```

With **`USE_OLLAMA=1`** in `.env`, ensure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull llama3.2`). Alternatively:

```bash
chmod +x scripts/run.sh && ./scripts/run.sh
```

Upload PDFs with **Company**, **Version**, and optional **Client / workspace**, then ask questions. Use **`DEMO.md`** for a 5–10 minute walkthrough script. Use **`SUBMISSION.md`** before you send the repo link + demo.

---

## Repository layout

```
README.md              # this file — architecture & assessment narrative
.env.example           # copy to .env; Ollama / API / DATABASE_URL options
PUSH_TO_GITHUB.md      # git remote + push steps
app.py                 # Streamlit UI
docker-compose.yml     # Postgres + pgvector
scripts/run.sh         # Ollama + venv + Streamlit helper
DEMO.md                # demo script
SUBMISSION.md          # submission checklist
src/
  config.py            # env + tuning knobs
  ingestion.py         # PDF + tables + chart note
  chunking.py
  embeddings.py
  persistence.py       # routes to Postgres or SQLite
  postgres_db.py       # SQLAlchemy + pgvector
  database.py          # SQLite + chunk rows
  faiss_store.py       # used only in SQLite mode
  retrieval.py
  rag.py               # LLM system prompt + grounded answering
  pipeline.py
```

---

## First-person closing

I optimized for **clarity and a believable assessment story**: a real **database-backed** store with **pgvector**, a **Streamlit** UI, and prompts that encode **version** and **conflict** behavior instead of pretending the model will magically behave. The parts I would scale next are **hybrid search**, **Snowflake wiring**, and **tenant isolation** — not more clever chunk sizes.
