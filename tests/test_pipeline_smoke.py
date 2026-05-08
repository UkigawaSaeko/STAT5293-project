from __future__ import annotations

from generator.llm_client import MockLLMClient
from main import build_answer_fn, pipeline


def test_no_rag_pipeline_smoke_with_mock_llm() -> None:
    cfg = {"llm_backend": "mock", "use_llm_hallucination_eval": False}
    llm = MockLLMClient()
    answer_fn = build_answer_fn("no_rag", llm, cfg)
    sample = {
        "doc_id": "paper-1",
        "question_id": "q1",
        "question": "What is the method?",
        "answers": ["BM25"],
        "evidence": ["The method uses BM25."],
    }

    output, metrics = pipeline(sample, answer_fn, "no_rag", cfg, llm)

    assert output.answer
    assert output.extra["method"] == "no_rag"
    assert set(["em", "f1", "citation_hit_rate", "prompt_tokens"]).issubset(metrics)
