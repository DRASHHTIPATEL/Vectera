# Before you submit

Quick list so nothing gets forgotten.

**What to send**
- Link to the **GitHub** (or similar) repo — local commits are already in the project; you still need to **push** (see `PUSH_TO_GITHUB.md`; I can’t log into your account).
- This **README**.
- A **5–10 min demo** — recorded or live is fine. Script ideas are in `DEMO.md`.

**What to say in the demo (one minute of context)**
- How you run the LLM: **Ollama** vs API key.
- Where data lives: **Postgres + pgvector** if you used Docker + `DATABASE_URL`, or **SQLite + FAISS** if you didn’t — both are valid; just don’t imply Postgres if you only ran SQLite.

**Sanity check before recording**
- `cp .env.example .env` if needed, fill in what you use.
- Ingest at least two versions for one company + one other doc.
- Ask one **version** question and one **conflict / cross-doc** question.
- Open **Retrieved context** once.

**If someone only reads three files:** `README.md`, `app.py`, `src/rag.py`.
