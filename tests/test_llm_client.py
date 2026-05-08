from __future__ import annotations

from generator.llm_client import MockLLMClient, parse_evidence_ids_line


def test_parse_evidence_ids_line_extracts_citations() -> None:
    answer, ids = parse_evidence_ids_line('Answer text.\nEVIDENCE_IDS: ["chunk_1", "chunk_2"]')
    assert answer == "Answer text."
    assert ids == ["chunk_1", "chunk_2"]


def test_parse_evidence_ids_line_handles_missing_or_bad_ids() -> None:
    answer, ids = parse_evidence_ids_line("Answer without citations.")
    assert answer == "Answer without citations."
    assert ids == []

    answer, ids = parse_evidence_ids_line("Answer.\nEVIDENCE_IDS: [not-json]")
    assert answer == "Answer."
    assert ids == []


def test_mock_llm_client_returns_usage() -> None:
    result = MockLLMClient().generate("Question: what is tested?")
    assert "[mock]" in result.text
    assert result.prompt_tokens > 0
    assert result.api_calls == 1
