from __future__ import annotations

import time

from generator.llm_client import LLMClient, parse_evidence_ids_line
from generator.prompts import rag_context_prompt
from parser.chunker import chunk_words
from parser.toc_builder import TOCNode
from retrievers.base import SystemOutput
from retrievers.toc_rag import TOCRAG, SelectionStrategy
from retrievers.vector_rag import Embedder, VectorRAG


class HybridRAG:
    """
    Two-stage retrieval:
    1) TOC navigation to select a relevant section
    2) Vector retrieval over chunks scoped to that section
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder,
        toc_max_depth: int = 4,
        toc_stop_if_score_below: float | None = None,
        toc_selection_strategy: SelectionStrategy = "bm25",
        scoped_chunk_max_words: int = 220,
        scoped_chunk_overlap_words: int = 40,
    ):
        self.llm = llm
        self.vector = VectorRAG(embedder=embedder, llm=llm)
        self.toc = TOCRAG(
            llm=llm,
            max_depth=toc_max_depth,
            stop_if_score_below=toc_stop_if_score_below,
            selection_strategy=toc_selection_strategy,
        )
        self.scoped_chunk_max_words = scoped_chunk_max_words
        self.scoped_chunk_overlap_words = scoped_chunk_overlap_words

    def _leaf_text(self, leaf: TOCNode) -> str:
        base = (leaf.content or "").strip()
        if base:
            return base
        if leaf.children:
            return "\n\n".join((c.title + "\n" + (c.content or "")).strip() for c in leaf.children)
        return ""

    def _scoped_chunks(self, leaf: TOCNode) -> list[dict]:
        text = self._leaf_text(leaf)
        if not text:
            return []
        chunks = chunk_words(text, self.scoped_chunk_max_words, self.scoped_chunk_overlap_words)
        out: list[dict] = []
        for i, piece in enumerate(chunks):
            # Keep citation IDs in the same `chunk_i` format as other RAG methods,
            # so parsed EVIDENCE_IDS can match retrieved_ids directly.
            out.append({"chunk_id": f"chunk_{i}", "text": piece})
        return out

    def answer(self, question: str, root: TOCNode, top_k: int = 3, doc_id: str = "") -> SystemOutput:
        t0 = time.perf_counter()
        leaf, nav_path = self.toc.navigate(question, root)
        chunks = self._scoped_chunks(leaf)
        self.vector.build_index(chunks)
        docs, ids = self.vector.retrieve(question, top_k=top_k)
        context = "\n\n".join(docs) if docs else "(empty context)"
        prompt = rag_context_prompt(question, context, cite_instruction=True)
        res = self.llm.generate(prompt)
        answer, cites = parse_evidence_ids_line(res.text)
        lat = time.perf_counter() - t0

        return SystemOutput(
            question=question,
            answer=answer or res.text.strip(),
            retrieved_contexts=docs,
            retrieved_ids=ids,
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
                "method": "hybrid_rag",
                "top_k": top_k,
                "toc_max_depth": self.toc.max_depth,
                "toc_selection_strategy": self.toc.selection_strategy,
                "scoped_chunk_max_words": self.scoped_chunk_max_words,
            },
        )


def make_hybrid_rag(llm: LLMClient, embedder: Embedder, cfg: dict) -> HybridRAG:
    raw = str(cfg.get("toc_selection", "bm25")).lower()
    sel: SelectionStrategy = raw if raw in ("bm25", "hash_embed") else "bm25"
    stop = cfg.get("toc_stop_score")
    stop_f = float(stop) if stop is not None else None
    return HybridRAG(
        llm=llm,
        embedder=embedder,
        toc_max_depth=int(cfg.get("hybrid_toc_max_depth", cfg.get("toc_max_depth", 4))),
        toc_stop_if_score_below=stop_f,
        toc_selection_strategy=sel,
        scoped_chunk_max_words=int(cfg.get("hybrid_chunk_max_words", 220)),
        scoped_chunk_overlap_words=int(cfg.get("hybrid_chunk_overlap_words", 40)),
    )
