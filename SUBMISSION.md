# Submission checklist (Vectera.ai technical assessment)

Use this when handing in the take-home.

## Required deliverables

| Item | Status |
|------|--------|
| **Git repository** (GitHub or similar) | Push this project; add the remote URL in your email or form. |
| **README** | `README.md` — architecture, database, chunking, retrieval, versions, conflicts, charts, limitations, improvements. |
| **Working app (not CLI)** | `streamlit run app.py` or `./scripts/run.sh` |
| **Demo (5–10 min)** | Follow `DEMO.md`; record (Loom, QuickTime, etc.) or do a live session. |

## Environment you should mention in the demo

1. **LLM:** `USE_OLLAMA=1` + Ollama, **or** `OPENAI_API_KEY` + cloud model.  
2. **Database (recommended for the brief):** `DATABASE_URL` → Postgres + pgvector (`docker compose up -d`).  
   If Docker is unavailable, state that you used **SQLite + FAISS** for local dev (supported fallback).

## Quick verify before recording

```bash
cp .env.example .env   # if needed
# edit .env (Ollama and/or DATABASE_URL)
docker compose up -d     # optional Postgres
ollama serve             # if using Ollama (or use scripts/run.sh)
streamlit run app.py
```

- Upload ≥2 PDFs with **different version labels** for one company.  
- Ask at least one **version** question and one **conflict / cross-document** question.  
- Expand **Retrieved context** once to show auditability.

## Files reviewers may open first

- `README.md` — design narrative  
- `app.py` — UI  
- `src/persistence.py` — DB routing  
- `src/rag.py` — system prompt + grounding rules  
