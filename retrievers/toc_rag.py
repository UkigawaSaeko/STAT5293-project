from __future__ import annotations

import time
from typing import Literal

import numpy as np

from generator.llm_client import LLMClient, parse_evidence_ids_line
from generator.prompts import answer_with_citations_prompt
from parser.toc_builder import TOCNode
from retrievers.base import SystemOutput

SelectionStrategy = Literal["bm25", "hash_embed"]


def _bm25_scores(question: str, documents: list[str]) -> list[float]:
    from rank_bm25 import BM25Okapi

    tokenized = [d.lower().split() for d in documents]
    if not tokenized:
        return []
    bm25 = BM25Okapi(tokenized)
    return list(bm25.get_scores(question.lower().split()))


def _hash_embed_scores(question: str, documents: list[str]) -> list[float]:
    from retrievers.vector_rag import _HashEmbedder

    emb = _HashEmbedder()
    q = emb.encode([question])[0]
    scores: list[float] = []
    for d in documents:
        v = emb.encode([d])[0]
        scores.append(float(np.dot(q, v)))
    return scores


class TOCRAG:
    def __init__(
        self,
        llm: LLMClient,
        max_depth: int = 5,
        stop_if_score_below: float | None = None,
        selection_strategy: SelectionStrategy = "bm25",
    ):
        self.llm = llm
        self.max_depth = max_depth
        self.stop_if_score_below = stop_if_score_below
        self.selection_strategy = selection_strategy

    def _score_children(self, question: str, children: list[TOCNode]) -> list[float]:
        docs = [(c.title + " " + (c.content or "")[:500]).strip() for c in children]
        if self.selection_strategy == "hash_embed":
            return _hash_embed_scores(question, docs)
        return _bm25_scores(question, docs)

    def navigate(self, question: str, root: TOCNode) -> tuple[TOCNode, list[str]]:
        node = root
        path: list[str] = []
        depth = 0
        if node.title != "ROOT":
            path.append(node.title)
        while node.children and depth < self.max_depth:
            scores = self._score_children(question, node.children)
            if not scores:
                break
            best_i = int(np.argmax(scores))
            best_sc = float(scores[best_i])
            if self.stop_if_score_below is not None and best_sc < self.stop_if_score_below:
                break
            node = node.children[best_i]
            path.append(node.title)
            depth += 1
        return node, path

    def answer(self, question: str, root: TOCNode, doc_id: str = "") -> SystemOutput:
        t0 = time.perf_counter()
        leaf, nav_path = self.navigate(question, root)
        context = (leaf.content or "").strip()
        if not context and leaf.children:
            context = "\n\n".join((c.title + "\n" + (c.content or "")).strip() for c in leaf.children)
        path_hint = " > ".join(nav_path)
        labeled = f"[{leaf.node_id or 'leaf'}]\n{context}" if context else "(empty section)"
        prompt = answer_with_citations_prompt(question, labeled, path_hint=path_hint)
        res = self.llm.generate(prompt)
        answer, cites = parse_evidence_ids_line(res.text)
        lat = time.perf_counter() - t0
        ctx_list = [labeled] if labeled and labeled != "(empty section)" else []
        return SystemOutput(
            question=question,
            answer=answer or res.text.strip(),
            retrieved_contexts=ctx_list,
            retrieved_ids=[str(leaf.node_id or "")],
            citations=cites,
            navigation_path=nav_path,
            latency_sec=lat,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            total_tokens=res.total_tokens or (res.prompt_tokens + res.completion_tokens),
            api_calls=res.api_calls,
            raw_model_text=res.text,
            extra={
                "doc_id": doc_id,
                "method": "toc_rag",
                "max_depth": self.max_depth,
                "stop_if_score_below": self.stop_if_score_below,
                "selection_strategy": self.selection_strategy,
            },
        )


def make_toc_rag(llm: LLMClient, cfg: dict) -> TOCRAG:
    raw = str(cfg.get("toc_selection", "bm25")).lower()
    sel: SelectionStrategy = raw if raw in ("bm25", "hash_embed") else "bm25"
    stop = cfg.get("toc_stop_score")
    stop_f = float(stop) if stop is not None else None
    return TOCRAG(
        llm=llm,
        max_depth=int(cfg.get("toc_max_depth", 5)),
        stop_if_score_below=stop_f,
        selection_strategy=sel,
    )
