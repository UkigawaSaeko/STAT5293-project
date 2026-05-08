# Rubric Alignment

This file maps the course rubric directly to the repository structure so graders can quickly find the relevant code, documentation, tests, and reproducibility artifacts.

## Clean and organized code structure

- `data/`: Qasper loading and normalization.
- `parser/`: document chunking, section parsing, and TOC construction.
- `retrievers/`: implementations of `no_rag`, `vector_rag`, `toc_rag`, and `hybrid_rag`.
- `generator/`: LLM client abstractions and prompt-output parsing.
- `evaluation/`: answer quality, evidence, citation, and hallucination metrics.
- `experiments/`: reproducible experiment entrypoints and configuration.
- `reports/`: exporting and plotting utilities.
- `tests/`: unit tests and a mock integration smoke test.
- `outputs/`: generated predictions and post-hoc analysis artifacts.

## Comprehensive documentation

- `README.md`: setup, API configuration, experiment commands, analysis, demo, tests, and findings.
- `docs/ARCHITECTURE.md`: system components and data flow.
- `docs/REPRODUCIBILITY.md`: step-by-step instructions for reproducing results.
- `docs/DEPLOYMENT.md`: Streamlit demo deployment and local launch guide.
- `docs/TESTING.md`: unit and integration testing instructions.
- `outputs/README.md`: generated artifact guide.

## Unit tests and error handling

- `tests/test_metrics.py`: answer metric edge cases.
- `tests/test_llm_client.py`: mock LLM behavior and citation parsing.
- `tests/test_qasper_loader.py`: Qasper normalization and sample expansion.
- `tests/test_pipeline_smoke.py`: mock end-to-end pipeline smoke test.
- `generator/llm_client.py`: clear errors and retry behavior for OpenAI-compatible HTTP calls.
- `demo_app.py`: friendly Streamlit errors for missing config, bad YAML, data-loading failures, and malformed prediction CSVs.

Run:

```bash
pytest
```

## Code optimization

- Retrieval uses FAISS for vector search.
- Streamlit data and prediction loading are cached with `st.cache_data`.
- Experiments cap documents/questions through `experiments/config.yaml` to control runtime and cost.
- Post-hoc analysis is vectorized with pandas/numpy where practical.

## Successful implementation of proposed features

Implemented methods:

- `no_rag`
- `vector_rag`
- `toc_rag`
- `hybrid_rag`

Main outputs:

- `outputs/predictions/*_predictions.csv`
- `outputs/predictions/*_summary.json`
- `outputs/analysis/overall_metrics.csv`
- `outputs/analysis/paired_bootstrap_tests.json`
- `outputs/analysis/fig_cost_vs_quality_f1.png`
- `outputs/analysis/fig_cost_vs_citation.png`

## Robust error handling

- API key absence is detected before OpenAI-compatible requests.
- HTTP/network/JSON errors include actionable messages.
- Streamlit explains missing prediction files and configuration issues instead of failing silently.
- CSV validation checks for required `question_id` columns.

## Performance optimization

- Configurable `max_documents`, `max_questions_per_doc`, `top_k`, chunk size, and TOC depth.
- FAISS indexing is used for chunk retrieval.
- Demo uses replayed CSVs to avoid repeated expensive API calls.
- Analysis scripts reuse saved predictions instead of recomputing model outputs.

## Integration testing

- `tests/test_pipeline_smoke.py` verifies that the shared pipeline runs with `MockLLMClient` without requiring network or API credentials.
- `.github/workflows/tests.yml` runs the test suite automatically on GitHub.

## Reproducible experiments

- `experiments/config.yaml` stores seed, data split, model, retrieval hyperparameters, and output paths.
- `experiments/run_baseline.py` supports aligned question IDs through `--match-question-ids-from`.
- `experiments/analyze_outputs.py` performs aligned comparison and paired bootstrap analysis with a fixed seed.
- `docs/REPRODUCIBILITY.md` lists the full command sequence.

## Well-documented experimental setup

- `README.md` and `docs/REPRODUCIBILITY.md` explain the Qasper split, paired alignment, metrics, and generated artifacts.
- `experiments/config.yaml` contains runtime settings and comments.
- `outputs/README.md` explains how to regenerate predictions, analysis tables, and figures.
