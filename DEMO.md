# Demo (about 5–10 minutes)

Here’s how I’d walk someone through this — recorded or live.

Before you hit record: pick your LLM path (**Ollama** with `USE_OLLAMA=1`, or a cloud key with `USE_OLLAMA=0`). If you have Docker, `docker compose up -d` + `DATABASE_URL` in `.env` is a nice story for Postgres; if not, just say you’re on **SQLite + FAISS** locally — that’s honest and the app supports it.

**1 — Setup (~1 min)**  
Open the README, scroll to the mermaid diagram, explain in one sentence: PDFs → chunks → embeddings → DB → retrieve → LLM. Start the app (`streamlit run app.py` or `./scripts/run.sh`). Point at the sidebar: **Client**, **Company**, **Version** — the version field matters for “compare across versions” questions.

**2 — Ingest (~2 min)**  
Upload two PDFs for the **same company** with **different version labels** (e.g. `2023-deck` vs `2024-deck`). Add another PDF from a **different company** or a third-party report so retrieval isn’t one document only. Check the indexed list in the sidebar.

**3 — Version question (~2 min)**  
Ask something like: *“How has [metric or strategy] changed across document versions for [Company X]?”*  
Call out that answers should tie back to **version labels** and **document + page** in Sources.

**4 — Conflicts (~2 min)**  
Ask: *“Are there conflicting data points across these materials on [topic]?”*  
Emphasize you’re **not** looking for the model to average two numbers — Conflicts section + separate attributions.

**5 — Charts / tables (~1 min)**  
If you have a chart-heavy slide: ask what the materials say. If the retrieved chunk has a limitation note, the answer should stay honest. If the topic just isn’t in the text, it shouldn’t invent chart data (rule 7 in `src/rag.py`).

**6 — Audit (~1 min)**  
Expand **Retrieved context** and show the actual chunks and scores — that’s the grounding story.

**7 — Optional (~1 min)**  
Switch **Client / workspace** between two values and ingest — scoped listing is clearer on Postgres; SQLite still filters in the app.

Example prompts from the brief (copy-paste friendly):

- What is [Company X]’s key strategy?
- How has [metric] changed across document versions?
- What drives demand according to these materials?
- Are there conflicting data points across documents?
- Summarize key trends shown in the documents
