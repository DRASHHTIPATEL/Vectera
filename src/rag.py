"""LLM answering with strict grounding, citations, conflict and version handling."""

from __future__ import annotations

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL


SYSTEM_PROMPT = """You are an analyst assistant for investment documents.

Rules (must follow all):
1) Answer ONLY using the provided CONTEXT blocks. If the answer is not in context, say you cannot find it in the uploaded documents.
2) Do NOT invent facts, numbers, or sources. Do NOT use outside knowledge to fill gaps.
3) If two context excerpts disagree (different numbers, dates, or claims), do NOT merge them into one blended fact. Present each version separately (e.g., "According to …" for each).
4) When multiple VERSION labels exist for the same company, explicitly attribute statements to the version (e.g., "According to Version 2024-Q1 …").
5) Every factual claim in your answer must be traceable to a source line below. Use inline citations like [DocumentName p.Page].
6) If context mentions that chart/table data could not be fully extracted, reflect that honestly; do not guess chart values.
7) Do NOT blame "chart extraction" or "image-heavy pages" unless that exact limitation appears in the CONTEXT for the relevant excerpt. If the topic is simply missing from the documents, say the documents do not address it.

Output format (exact headings):

Answer:
<your answer>

Key Points:
- <bullet>
- <bullet>

Conflicts (if any):
- <bullet, or "None apparent from the provided context.">

Sources:
- <Document name> (Page <n>) [Version: <version> if applicable]
"""


def _format_context(chunks: list[dict]) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        header = (
            f"[{i}] document={c['document_name']} | page={c['page_number']} | "
            f"company={c['company_name']} | version={c['version']}"
        )
        body = c["chunk_text"].strip()
        if c.get("chart_note"):
            body += f"\n(Note: {c['chart_note']})"
        lines.append(header + "\n" + body)
    return "\n\n---\n\n".join(lines)


def answer_question(question: str, chunks: list[dict]) -> str:
    if not OPENAI_API_KEY:
        return (
            "Answer:\n"
            "I cannot call the language model. Do one of the following:\n"
            "- **Free (local):** Install [Ollama](https://ollama.com), run `ollama pull llama3.2`, "
            "start the Ollama app, then set `USE_OLLAMA=1` in `.env` (no paid API key).\n"
            "- **Cloud:** Set `OPENAI_API_KEY` in `.env` (see `.env.example`).\n\n"
            "Key Points:\n"
            "- Embeddings already run locally; only the answer step needs an LLM.\n\n"
            "Conflicts (if any):\n"
            "- N/A\n\n"
            "Sources:\n"
            + "\n".join(
                f"- {c['document_name']} (Page {c['page_number']}) [Version: {c['version']}]"
                for c in chunks
            )
        )

    if not chunks:
        return (
            "Answer:\nNo relevant chunks were retrieved. Ingest documents first.\n\n"
            "Key Points:\n- Upload and index PDFs before asking questions.\n\n"
            "Conflicts (if any):\n- N/A\n\n"
            "Sources:\n- (none)"
        )

    ctx = _format_context(chunks)
    user_msg = f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}"

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    # Ollama ignores the key but the client requires a non-empty string.
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    return text
