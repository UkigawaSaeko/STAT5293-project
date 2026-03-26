from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from retrievers.base import SystemOutput


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def token_f1(pred: str, gold: str) -> float:
    pt = normalize_text(pred).split()
    gt = normalize_text(gold).split()
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    pc, gc = Counter(pt), Counter(gt)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pt)
    recall = overlap / len(gt)
    return 2 * precision * recall / (precision + recall)


def best_answer_metrics(pred: str, gold_answers: list[str]) -> dict[str, float]:
    if not gold_answers:
        return {"em": 0.0, "f1": 0.0}
    ems = [exact_match(pred, g) for g in gold_answers]
    f1s = [token_f1(pred, g) for g in gold_answers]
    return {"em": float(any(ems)), "f1": float(max(f1s))}


def recall_at_k(retrieved_ids: Iterable[str], gold_ids: Iterable[str]) -> float:
    r = {str(x) for x in retrieved_ids}
    g = {str(x) for x in gold_ids}
    if not g:
        return 0.0
    hit = len(r & g)
    return hit / len(g)


def evidence_hit_rate(retrieved_texts: list[str], evidence_strings: list[str]) -> float:
    if not evidence_strings:
        return 0.0
    blob = "\n".join(retrieved_texts).lower()
    hits = 0
    for ev in evidence_strings:
        e = ev.strip().lower()
        if len(e) < 12:
            continue
        if e[:80] in blob or e in blob:
            hits += 1
    return hits / len(evidence_strings)


def aggregate_efficiency(outputs: list[SystemOutput]) -> dict[str, float]:
    if not outputs:
        return {"avg_latency": 0.0, "total_api_calls": 0, "avg_total_tokens": 0.0}
    n = len(outputs)
    return {
        "avg_latency": sum(o.latency_sec for o in outputs) / n,
        "total_api_calls": float(sum(o.api_calls for o in outputs)),
        "avg_total_tokens": sum(o.total_tokens for o in outputs) / n,
    }
