# Demo script (5–10 minutes)

Use for a **recorded** walkthrough or **live** review.

## Prerequisites (pick what matches your `.env`)

- **LLM:** Either **Ollama** (`USE_OLLAMA=1`, `ollama serve`, `ollama pull llama3.2`) **or** a cloud key (`OPENAI_API_KEY`, `USE_OLLAMA=0`).
- **Database (recommended for the assessment narrative):** `docker compose up -d`, set `DATABASE_URL` in `.env` for **Postgres + pgvector**.  
  If you skip Docker, say explicitly that you’re on **SQLite + FAISS** for the demo machine.

## 1. Setup (~1 min)

- Show `README.md` architecture (mermaid diagram) and **how data is stored** (Postgres+pgvector vs SQLite+FAISS).
- Start the app: `streamlit run app.py` or `./scripts/run.sh`.
- Point out sidebar: **Client / workspace**, **Company**, **Version** — version labels drive **version-aware** answers.

## 2. Ingest (~2 min)

- Upload **two PDFs** for the same company with **different version labels** (e.g. `2023-deck` vs `2024-deck`).
- Upload at least one PDF for a **second company** (or third-party report) so **multi-source** retrieval is visible.
- Confirm **Indexed documents** updates; note **backend** line (Postgres vs SQLite).

## 3. Version-aware question (~2 min)

Example:

- *“How has [metric or strategy] changed across document versions for [Company X]?”*

**Highlight:** attributions to **version labels** and citations **document + page**.

## 4. Cross-document / conflicts (~2 min)

Example:

- *“Are there conflicting data points across these materials on [topic]?”*

**Highlight:** model should **not** merge conflicting facts; **Conflicts** section + **Sources**.

## 5. Charts / tables (~1 min)

Pick a chart-heavy slide if available:

- *“What does the material say about [topic] on that slide?”*

**Highlight:** if context includes a **chart extraction** note, the answer stays honest; if the topic is simply missing, it should **not** blame charts unless that note appears (see `src/rag.py` rule 7).

## 6. Audit trail (~1 min)

- Expand **Retrieved context** — show **chunks**, scores, metadata.

## 7. Optional: client scope (~1 min)

- **Client / workspace** `client-a` → ingest; switch to `client-b` → ingest — show scoped listing (Postgres filters in DB; SQLite filters in app).

---

## Example questions (from the brief)

- What is [Company X]’s key strategy?
- How has [metric] changed across document versions?
- What drives demand according to these materials?
- Are there conflicting data points across documents?
- Summarize key trends shown in the documents
