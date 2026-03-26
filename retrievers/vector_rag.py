from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from generator.llm_client import LLMClient, parse_evidence_ids_line
from generator.prompts import rag_context_prompt
from retrievers.base import SystemOutput


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class _HashEmbedder:
    """Lightweight deterministic embedder for dev without sentence-transformers."""

    dim: int = 256

    def encode(self, texts: list[str]) -> np.ndarray:
        import hashlib

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8", errors="ignore")).digest()
            for j in range(self.dim):
                out[i, j] = int(h[j % len(h)]) / 255.0 - 0.5
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
        return out / norms


def _try_sentence_transformer(model_name: str) -> Embedder:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model_name)

    class STEmbedder:
        def encode(self, texts: list[str]) -> np.ndarray:
            emb = st.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return np.asarray(emb, dtype=np.float32)

    return STEmbedder()


def build_embedder(cfg: dict) -> Embedder:
    name = cfg.get("embedding_model")
    if name:
        return _try_sentence_transformer(name)
    return _HashEmbedder()


class VectorRAG:
    def __init__(self, embedder: Embedder, llm: LLMClient):
        self.embedder = embedder
        self.llm = llm
        self._index = None
        self._chunks: list[dict] = []

    def build_index(self, chunks: list[dict]) -> None:
        import faiss

        self._chunks = list(chunks)
        texts = [c["text"] for c in self._chunks]
        if not texts:
            self._index = None
            return
        embs = self.embedder.encode(texts).astype("float32")
        dim = embs.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embs)
        self._index = index

    def retrieve(self, question: str, top_k: int = 5) -> tuple[list[str], list[str]]:
        if not self._chunks or self._index is None:
            return [], []
        import faiss

        q = self.embedder.encode([question]).astype("float32")
        k = min(top_k, len(self._chunks))
        _, I = self._index.search(q, k)
        ctx: list[str] = []
        ids: list[str] = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self._chunks):
                continue
            c = self._chunks[idx]
            cid = c.get("chunk_id", str(idx))
            ids.append(str(cid))
            ctx.append(f"[{cid}]\n{c['text']}")
        return ctx, ids

    def answer(self, question: str, top_k: int = 5, doc_id: str = "") -> SystemOutput:
        t0 = time.perf_counter()
        docs, ids = self.retrieve(question, top_k=top_k)
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
            navigation_path=[],
            latency_sec=lat,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            total_tokens=res.total_tokens or (res.prompt_tokens + res.completion_tokens),
            api_calls=res.api_calls,
            raw_model_text=res.text,
            extra={"doc_id": doc_id, "method": "vector_rag", "top_k": top_k},
        )
