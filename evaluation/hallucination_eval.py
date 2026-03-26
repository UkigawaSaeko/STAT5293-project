from __future__ import annotations

import re

from generator.llm_client import LLMClient


def heuristic_hallucination(pred: str, contexts: list[str]) -> bool:
    """Cheap proxy: long answer with almost no n-gram overlap with context."""
    if not pred.strip():
        return False
    ctx = " ".join(contexts).lower()
    toks = re.findall(r"[a-z0-9]+", pred.lower())
    if len(toks) < 6:
        return False
    hits = sum(1 for t in toks if len(t) > 3 and t in ctx)
    return hits / max(len(toks), 1) < 0.15


def llm_hallucination_flag(llm: LLMClient, answer: str, evidence: str) -> tuple[bool, str]:
    prompt = (
        "Given the answer and evidence text, reply YES if the answer contains claims "
        "not supported by the evidence, else NO.\n\n"
        f"Evidence:\n{evidence[:6000]}\n\nAnswer:\n{answer}\n\nReply with YES or NO only."
    )
    res = llm.generate(prompt)
    text = res.text.strip().upper()
    unsupported = text.startswith("Y")
    return unsupported, res.text
