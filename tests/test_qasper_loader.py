from __future__ import annotations

from data.qasper_loader import expand_qasper_rows, normalize_document


def _row() -> dict:
    return {
        "id": "paper-1",
        "title": "A Test Paper",
        "abstract": "This is an abstract.",
        "full_text": {
            "section_name": ["Introduction", "Method"],
            "paragraphs": [["Intro evidence appears here."], ["The method uses BM25."]],
        },
        "qas": {
            "question": ["What retrieval method is used?"],
            "question_id": ["q1"],
            "answers": [
                {
                    "answer": [
                        {
                            "free_form_answer": "BM25",
                            "extractive_spans": ["BM25"],
                            "evidence": ["The method uses BM25."],
                        }
                    ]
                }
            ],
        },
    }


def test_normalize_document_builds_full_text_and_sections() -> None:
    doc = normalize_document(_row())
    assert doc["doc_id"] == "paper-1"
    assert len(doc["sections"]) == 2
    assert "Abstract" in doc["full_text"]
    assert "The method uses BM25." in doc["full_text"]


def test_expand_qasper_rows_flattens_answers_and_evidence() -> None:
    samples = list(expand_qasper_rows(_row()))
    assert len(samples) == 1
    sample = samples[0]
    assert sample["question_id"] == "q1"
    assert "BM25" in sample["answers"]
    assert sample["evidence"] == ["The method uses BM25."]
    assert sample["evidence_section_indices"] == [1]
