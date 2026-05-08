from __future__ import annotations

from evaluation.metrics import best_answer_metrics, exact_match, token_f1


def test_exact_match_normalizes_case_and_space() -> None:
    assert exact_match("  Large   Corpus ", "large corpus")


def test_token_f1_allows_partial_overlap() -> None:
    assert token_f1("large Portuguese corpus", "Portuguese corpus") > 0.0
    assert token_f1("abc", "xyz") == 0.0


def test_best_answer_metrics_uses_best_gold_variant() -> None:
    metrics = best_answer_metrics("yes", ["no", "yes"])
    assert metrics["em"] == 1.0
    assert metrics["f1"] == 1.0


def test_best_answer_metrics_handles_empty_gold() -> None:
    assert best_answer_metrics("anything", []) == {"em": 0.0, "f1": 0.0}
