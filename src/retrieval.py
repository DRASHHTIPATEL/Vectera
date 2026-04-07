"""
Retrieve top similar chunks, then diversify so answers are not dominated by one PDF.
"""

from __future__ import annotations

from collections import defaultdict

from src.config import (
    MAX_CHUNKS_PER_DOCUMENT_IN_BATCH,
    RETRIEVAL_CANDIDATES,
    RETRIEVAL_FINAL_MAX,
    RETRIEVAL_FINAL_MIN,
)
from src.embeddings import embed_texts
from src.persistence import get_chunks_by_keys, vector_similarity_search


def diversified_retrieve(
    query: str,
    final_k: int | None = None,
    client_label: str | None = None,
) -> list[dict]:
    """
    1) Vector search (pgvector or FAISS) for top candidates.
    2) Greedy rerank: cap chunks per document_name; optional cross-company swap.
    """
    k_final = final_k or RETRIEVAL_FINAL_MAX
    k_final = max(RETRIEVAL_FINAL_MIN, min(RETRIEVAL_FINAL_MAX, k_final))

    qv = embed_texts([query])
    if qv.shape[0] == 0:
        return []

    order = vector_similarity_search(qv, RETRIEVAL_CANDIDATES, client_label=client_label)
    if not order:
        return []

    chunk_keys = [int(k) for _, k in order]
    by_key = get_chunks_by_keys(chunk_keys)

    per_doc: dict[str, int] = defaultdict(int)
    selected: list[tuple[float, int]] = []

    for score, ck in order:
        ck = int(ck)
        meta = by_key.get(ck)
        if meta is None:
            continue
        doc_key = meta["document_name"]
        if per_doc[doc_key] >= MAX_CHUNKS_PER_DOCUMENT_IN_BATCH:
            continue
        selected.append((score, ck))
        per_doc[doc_key] += 1
        if len(selected) >= k_final:
            break

    if len(selected) < RETRIEVAL_FINAL_MIN:
        for score, ck in order:
            ck = int(ck)
            if any(ck == s[1] for s in selected):
                continue
            meta = by_key.get(ck)
            if meta is None:
                continue
            selected.append((score, ck))
            if len(selected) >= RETRIEVAL_FINAL_MIN:
                break

    if len(selected) >= 4:
        metas = [by_key[s[1]] for s in selected if s[1] in by_key]
        companies = {m["company_name"] for m in metas}
        if len(companies) == 1:
            only_co = next(iter(companies))
            selected_keys = {s[1] for s in selected}
            for score, ck in order:
                ck = int(ck)
                if ck in selected_keys:
                    continue
                meta = by_key.get(ck)
                if meta is None or meta["company_name"] == only_co:
                    continue
                weakest = min(selected, key=lambda x: x[0])
                if weakest[0] < score:
                    selected = [s for s in selected if s[1] != weakest[1]]
                    selected.append((score, ck))
                break

    results: list[dict] = []
    for score, ck in selected:
        m = by_key.get(ck)
        if m is None:
            continue
        row = dict(m)
        row["score"] = float(score)
        results.append(row)
    return results
