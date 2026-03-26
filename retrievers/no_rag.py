from __future__ import annotations

import time

from generator.llm_client import LLMClient, parse_evidence_ids_line
from generator.prompts import no_rag_prompt
from retrievers.base import SystemOutput


class NoRAG:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def answer(self, question: str, doc_id: str = "") -> SystemOutput:
        t0 = time.perf_counter()
        prompt = no_rag_prompt(question)
        res = self.llm.generate(prompt)
        answer, cites = parse_evidence_ids_line(res.text)
        lat = time.perf_counter() - t0
        return SystemOutput(
            question=question,
            answer=answer or res.text.strip(),
            retrieved_contexts=[],
            retrieved_ids=[],
            citations=cites,
            navigation_path=[],
            latency_sec=lat,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            total_tokens=res.total_tokens or (res.prompt_tokens + res.completion_tokens),
            api_calls=res.api_calls,
            raw_model_text=res.text,
            extra={"doc_id": doc_id, "method": "no_rag"},
        )
