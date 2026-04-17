"""LLM answering with strict grounding, citations, conflict and version handling."""

from __future__ import annotations

import re

from openai import APIConnectionError, APITimeoutError, OpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, USE_OLLAMA


SYSTEM_PROMPT = """You are an analyst assistant for investment documents.

Rules (must follow all):
1) Answer ONLY using the provided CONTEXT blocks. If the answer is not in context, say you cannot find it in the uploaded documents.
2) Do NOT invent facts, numbers, or sources. Do NOT use outside knowledge to fill gaps.
3) If two context excerpts disagree (different numbers, dates, or claims), do NOT merge them into one blended fact. Present each version separately (e.g., "According to …" for each).
4) When multiple VERSION labels exist for the same company, explicitly attribute statements to the version (e.g., "According to Version 2024-Q1 …").
4b) If the question compares **two or more time periods** (e.g. different months/years or decks), answer **for each period separately** using only the CONTEXT lines that match that period or file. If context for a named period is missing, say it is not in the retrieved excerpts.
5) Every factual claim in your answer must be traceable to a source line below. Use inline citations like [DocumentName p.Page].
6) If context mentions that chart/table data could not be fully extracted, reflect that honestly; do not guess chart values.
7) Do NOT blame "chart extraction" or "image-heavy pages" unless that exact limitation appears in the CONTEXT for the relevant excerpt. If the topic is simply missing from the documents, say the documents do not address it.
8) In **Sources**, every line MUST look like a human citation: `- Filename.pdf (Page N) [Version: label]`. Do NOT echo CONTEXT headers or use `document=…|page=…|company=…` key-value style.
9) For chart OCR excerpts: treat y-axis tick marks (e.g. 0%, 5%, 10%, 15%, ...) as scale only, not data values. Prefer explicit bar labels or `[CHART_OCR_SERIES]` when available.

Output format (exact headings):

Answer:
<your answer>

Key Points:
- <bullet>
- <bullet>

Conflicts (if any):
- <bullet, or "None apparent from the provided context.">

Sources:
- <document_filename.pdf> (Page <n>) [Version: <version>]
- (repeat per source; same format only)
"""


def _fallback_sources(chunks: list[dict]) -> list[str]:
    out: list[str] = []
    seen: set[tuple[str, int, str]] = set()
    for c in chunks:
        doc = str(c.get("document_name") or "").strip()
        page = int(c.get("page_number") or 0)
        ver = str(c.get("version") or "unknown").strip() or "unknown"
        if not doc or page <= 0:
            continue
        key = (doc, page, ver)
        if key in seen:
            continue
        seen.add(key)
        out.append(f"- {doc} (Page {page}) [Version: {ver}]")
    return out or ["- (none)"]


def _normalize_source_line(line: str) -> str | None:
    line = line.strip().lstrip("-").strip()
    if not line:
        return None

    # Already in desired style.
    m_human = re.search(r"(.+?)\s*\(Page\s*(\d+)\)\s*\[Version:\s*([^\]]+)\]", line, flags=re.I)
    if m_human:
        doc = m_human.group(1).strip()
        page = int(m_human.group(2))
        ver = m_human.group(3).strip()
        return f"- {doc} (Page {page}) [Version: {ver}]"

    # Parse key=value style lines produced by some models.
    doc = None
    page = None
    ver = None
    m_doc = re.search(r"document=([^|]+)", line, flags=re.I)
    if m_doc:
        doc = m_doc.group(1).strip()
    m_page = re.search(r"page=(\d+)", line, flags=re.I)
    if m_page:
        page = int(m_page.group(1))
    m_ver = re.search(r"version=([^|\]]+)", line, flags=re.I)
    if m_ver:
        ver = m_ver.group(1).strip()

    if doc and page:
        ver = ver or "unknown"
        return f"- {doc} (Page {page}) [Version: {ver}]"
    return None


def _extract_chart_series_from_chunks(chunks: list[dict]) -> dict[str, float]:
    """
    Parse explicit chart series hints from OCR chunks, e.g.
    [CHART_OCR_SERIES] US=17.7%; EGP=24.2%
    """
    series: dict[str, float] = {}
    for c in chunks:
        txt = str(c.get("chunk_text") or "")
        m = re.search(r"\[CHART_OCR_SERIES\]\s*([^\n]+)", txt, flags=re.I)
        if not m:
            continue
        for part in [p.strip() for p in m.group(1).split(";") if p.strip()]:
            mm = re.match(r"([A-Za-z0-9 _-]{1,24})\s*=\s*([\d.]+)\s*%?", part)
            if not mm:
                continue
            k = mm.group(1).strip().lower()
            try:
                series[k] = float(mm.group(2))
            except ValueError:
                continue
    return series


def _extract_chart_relative_percent_from_chunks(chunks: list[dict]) -> float | None:
    """
    Prefer explicit 'X% greater than' text from OCR chunks.
    Fallback: compute from US/EGP series if available.
    """
    for c in chunks:
        txt = str(c.get("chunk_text") or "")
        m = re.search(r"\b(\d{1,3}(?:\.\d+)?)\s*%\s*greater than\b", txt, flags=re.I)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    series = _extract_chart_series_from_chunks(chunks)
    us = series.get("us")
    egp = series.get("egp")
    if us is not None and egp is not None and us > 0:
        return round(((egp - us) / us) * 100.0, 1)
    return None


def _enforce_chart_series_consistency(text: str, chunks: list[dict]) -> str:
    """
    If explicit chart series exists, correct conflicting percentages in answer/key points
    for lines that clearly refer to those labels (e.g., US / EGP).
    """
    series = _extract_chart_series_from_chunks(chunks)
    if not series:
        return text
    out_lines: list[str] = []
    for line in text.splitlines():
        lower = line.lower()
        for label, value in series.items():
            if label not in lower:
                continue
            # Replace first percentage in this label line with explicit series value.
            repl = f"{value:g}%"
            line = re.sub(r"\b\d{1,3}(?:\.\d+)?\s*%", repl, line, count=1)
            lower = line.lower()
        out_lines.append(line)

    rel = _extract_chart_relative_percent_from_chunks(chunks)
    if rel is None:
        return "\n".join(out_lines)

    fixed_lines: list[str] = []
    for line in out_lines:
        low = line.lower()
        if "greater than" in low and "u.s" in low and "egp" in low:
            line = re.sub(r"\b\d{1,3}(?:\.\d+)?\s*%\s*greater than\b", f"{rel:g}% greater than", line, flags=re.I)
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


def _postprocess_answer(text: str, chunks: list[dict]) -> str:
    """Keep wording natural and enforce clean citation style."""
    if not text.strip():
        return text

    # Remove raw model artifacts like "[DocumentName p.7 | source_type=ocr | ...]".
    text = re.sub(
        r"\s*\[[^\]\n]*(?:documentname|source_type|confidence|version\s*[:=])[^\]\n]*\]",
        "",
        text,
        flags=re.I,
    )
    text = _enforce_chart_series_consistency(text, chunks)

    sections = re.split(r"\n(?=Sources:\s*$)", text, flags=re.I | re.M)
    if len(sections) == 1:
        # No explicit Sources section: append normalized fallback sources.
        return text.rstrip() + "\n\nSources:\n" + "\n".join(_fallback_sources(chunks))

    head = sections[0].rstrip()
    tail = sections[1]
    lines = [ln for ln in tail.splitlines()[1:] if ln.strip()]
    norm: list[str] = []
    seen: set[str] = set()
    for ln in lines:
        n = _normalize_source_line(ln)
        if not n or n in seen:
            continue
        seen.add(n)
        norm.append(n)
    if not norm:
        norm = _fallback_sources(chunks)
    return head + "\n\nSources:\n" + "\n".join(norm)


def _format_context(chunks: list[dict]) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        header = (
            f"[{i}] document={c['document_name']} | page={c['page_number']} | "
            f"company={c['company_name']} | version={c['version']}"
        )
        tags: list[str] = []
        if c.get("from_stored_metrics"):
            tags.append("structured_metric_store")
        st = c.get("source_type")
        if st:
            tags.append(f"source_type={st}")
        if c.get("confidence"):
            tags.append(f"confidence={c['confidence']}")
        if c.get("ocr_low_confidence"):
            tags.append("ocr_or_low_confidence_extract")
        if tags:
            header += " | " + " ".join(tags)
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
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
    except (APIConnectionError, APITimeoutError) as e:
        endpoint = OPENAI_BASE_URL or "https://api.openai.com/v1"
        hint = (
            "**Local Ollama:** open the Ollama app or run `ollama serve`, then `ollama pull "
            f"{OPENAI_MODEL}` if needed. `.env` should have `USE_OLLAMA=1`.\n\n"
            "**Cloud API:** set `USE_OLLAMA=0`, `OPENAI_API_KEY=...`, and "
            "`OPENAI_BASE_URL=https://api.openai.com/v1` (or your provider)."
        )
        if not USE_OLLAMA:
            hint = (
                "**Cloud:** check `OPENAI_API_KEY`, network/VPN, and `OPENAI_BASE_URL`. "
                "For OpenAI use `https://api.openai.com/v1`.\n\n"
                "**Local:** set `USE_OLLAMA=1`, start Ollama, and use "
                "`OPENAI_BASE_URL=http://127.0.0.1:11434/v1`."
            )
        return (
            "Answer:\n"
            f"Could not reach the language model (`{endpoint}`, model `{OPENAI_MODEL}`): "
            f"{type(e).__name__}.\n\n"
            f"{hint}\n\n"
            "Key Points:\n"
            "- Retrieval ran; only the chat step failed.\n\n"
            "Conflicts (if any):\n"
            "- N/A\n\n"
            "Sources:\n"
            + "\n".join(
                f"- {c['document_name']} (Page {c['page_number']}) [Version: {c['version']}]"
                for c in chunks
            )
        )
    text = (resp.choices[0].message.content or "").strip()
    return _postprocess_answer(text, chunks)
