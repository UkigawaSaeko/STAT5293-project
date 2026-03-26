from __future__ import annotations


def citation_hits_retrieved(citations: list[str], retrieved_ids: list[str]) -> float:
    if not citations:
        return 0.0
    r = set(retrieved_ids)
    hits = sum(1 for c in citations if c in r)
    return hits / len(citations)


def citation_precision(citations: list[str], valid_ids: set[str]) -> float:
    if not citations:
        return 0.0
    ok = sum(1 for c in citations if c in valid_ids)
    return ok / len(citations)
