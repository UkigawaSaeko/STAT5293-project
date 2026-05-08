# Testing Guide

The repository includes lightweight tests for unit-level correctness and a mock integration path.

## Run tests

From the repository root:

```bash
pytest
```

## Test files

- `tests/test_metrics.py`: exact match, token F1, and best-gold answer scoring.
- `tests/test_llm_client.py`: mock LLM output and `EVIDENCE_IDS` parsing.
- `tests/test_qasper_loader.py`: Qasper document normalization and answer/evidence flattening.
- `tests/test_pipeline_smoke.py`: shared pipeline smoke test using `MockLLMClient`.

## Why mock integration?

The real OpenAI-compatible backend requires credentials and network access, which are not appropriate for automated grading or CI. The mock integration test verifies the project pipeline without spending API calls:

```text
sample question -> build_answer_fn("no_rag") -> MockLLMClient -> pipeline -> metrics
```

## GitHub Actions

The repository includes `.github/workflows/tests.yml` so GitHub can run the test suite automatically after pushes and pull requests.
