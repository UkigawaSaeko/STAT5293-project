# STAT5293 Project

**Does Explicit Document Structure Beat Similarity-Based Retrieval for Long-Document QA?**

This repository evaluates four long-document question-answering pipelines on Qasper under one reproducible experiment framework:

- `no_rag`: answer directly from the LLM without retrieval.
- `vector_rag`: chunk the full paper and retrieve top-k chunks by vector similarity.
- `toc_rag`: use document table-of-contents structure to navigate to a relevant section.
- `hybrid_rag`: use TOC navigation first, then run scoped vector retrieval inside the selected section.

The project compares answer quality, grounding reliability, hallucination proxies, citation behavior, and token cost.

## Quick Grading Guide

The following files are organized around the course rubric:

| Rubric item | Where to look |
| --- | --- |
| Clean and organized code structure | `docs/ARCHITECTURE.md`, `data/`, `parser/`, `retrievers/`, `evaluation/`, `experiments/` |
| Comprehensive documentation | `README.md`, `docs/REPRODUCIBILITY.md`, `docs/DEPLOYMENT.md`, `outputs/README.md` |
| Unit tests and error handling | `tests/`, `docs/TESTING.md`, `generator/llm_client.py`, `demo_app.py` |
| Code/performance optimization | `retrievers/vector_rag.py`, `demo_app.py`, `experiments/config.yaml` |
| Successful proposed features | `retrievers/`, `outputs/analysis/overall_metrics.csv`, `demo_app.py` |
| Robust error handling | `generator/llm_client.py`, `demo_app.py` |
| Integration testing | `tests/test_pipeline_smoke.py`, `.github/workflows/tests.yml` |
| Reproducible experiments | `experiments/config.yaml`, `experiments/run_baseline.py`, `experiments/analyze_outputs.py`, `docs/REPRODUCIBILITY.md` |
| Well-documented experimental setup | `README.md`, `docs/RUBRIC_ALIGNMENT.md`, `outputs/README.md` |

For a one-page rubric checklist, see `docs/RUBRIC_ALIGNMENT.md`.

## Project Structure

```text
STAT5293-project/
  README.md                         # Setup, experiment, demo, and reproducibility guide
  requirements.txt                  # Python dependencies
  .github/workflows/tests.yml        # CI test workflow
  main.py                           # Shared pipeline functions and optional CLI entrypoint
  demo_app.py                       # Streamlit replay demo for four QA systems
  docs/
    ARCHITECTURE.md                 # System components and data flow
    DEPLOYMENT.md                   # Streamlit deployment and local demo guide
    REPRODUCIBILITY.md              # Step-by-step experiment reproduction
    RUBRIC_ALIGNMENT.md             # Direct mapping to course rubric
    TESTING.md                      # Unit/integration testing guide
  experiments/
    config.yaml                     # Main runtime and experiment configuration
    run_baseline.py                 # Run one method and save predictions/summary
    run_ablation.py                 # Run small hyperparameter ablations
    analyze_outputs.py              # Align predictions, bootstrap tests, plots, tables
  data/                             # Qasper loading and normalization
  parser/                           # Paper section parsing, TOC construction, chunking
  retrievers/                       # No-RAG, Vector-RAG, TOC-RAG, Hybrid-RAG
  generator/                        # LLM clients and prompt parsing helpers
  evaluation/                       # EM/F1, evidence, citation, hallucination metrics
  reports/                          # Output export and plotting helpers
  utils/                            # Logging and seeding utilities
  tests/                            # Unit and smoke tests
  outputs/
    README.md                       # Generated artifact guide
    predictions/                    # Generated method prediction CSV/JSON files
    analysis/                       # Generated post-hoc tables, tests, figures, reports
```

Run commands from the repository root, the directory that contains `README.md`, `main.py`, and `demo_app.py`.

## Environment Setup

Python 3.10+ is recommended. Python 3.13 was used in the final local run.

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install the same dependencies before running experiments, tests, or the Streamlit demo.

## API Configuration

The default backend is OpenAI-compatible (`llm_backend: openai` in `experiments/config.yaml`). Do not commit API keys.

macOS/Linux:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional compatible gateway
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

For a cheap offline smoke check, switch the config to:

```yaml
llm_backend: mock
```

The final dense retrieval configuration uses:

```yaml
embedding_model: sentence-transformers/all-MiniLM-L6-v2
```

## Running Baseline Experiments

`experiments/run_baseline.py` is the recommended reproducible entrypoint. `main.py` contains the shared pipeline used by the experiment scripts and can also be used for direct development.

Run each method from the repository root:

```bash
python experiments/run_baseline.py --method no_rag
python experiments/run_baseline.py --method vector_rag
python experiments/run_baseline.py --method toc_rag
python experiments/run_baseline.py --method hybrid_rag
```

Expected outputs:

- `outputs/predictions/no_rag_predictions.csv`
- `outputs/predictions/vector_rag_predictions.csv`
- `outputs/predictions/toc_rag_predictions.csv`
- `outputs/predictions/hybrid_rag_predictions.csv`
- `outputs/predictions/*_summary.json`

For strict fairness, run one method first and align the remaining methods to its `question_id`s:

```bash
python experiments/run_baseline.py --method vector_rag
python experiments/run_baseline.py --method no_rag --match-question-ids-from outputs/predictions/vector_rag_predictions.csv
python experiments/run_baseline.py --method toc_rag --match-question-ids-from outputs/predictions/vector_rag_predictions.csv
python experiments/run_baseline.py --method hybrid_rag --match-question-ids-from outputs/predictions/vector_rag_predictions.csv
```

The submitted analysis used the configured Qasper `train` split as a controlled experiment slice, not as a claim of test-set generalization.

## Post-Hoc Analysis

After all four baseline prediction CSVs exist:

```bash
python experiments/analyze_outputs.py
```

Expected outputs in `outputs/analysis/`:

- `aligned_predictions.csv`
- `overall_metrics.csv`
- `paired_differences.csv`
- `paired_bootstrap_tests.json`
- `fig_cost_vs_quality_f1.png`
- `fig_cost_vs_citation.png`
- `analysis_report.md`

Exact match (`EM`) is reported for completeness, but Qasper answers are free-form and model outputs are often paraphrastic full-sentence responses. In our runs, `EM` can be zero across methods even when token-level `F1` is nonzero. We therefore use token-level `F1`, evidence hit rate, citation hit rate, and token cost as the primary comparison signals.

## Ablation Study

```bash
python experiments/run_ablation.py
```

The ablation uses a smaller configured slice (`ablation_max_documents`, `ablation_max_questions_per_doc`) and writes summaries to `outputs/predictions/ablation/`. It is intended for trend analysis and is not directly comparable to the main aligned `n=398` baseline table.

## Streamlit Demo

The demo is a **historical replay app**: it reads saved prediction CSVs instead of making live API calls. Generate the four baseline CSVs first, then run:

```bash
streamlit run demo_app.py
```

The interface lets users pick a paper and question, inspect the TOC and gold evidence, and compare No-RAG, Vector-RAG, TOC-RAG, and Hybrid-RAG side by side. Sidebar controls document the experiment setup but do not recompute historical answers unless you regenerate the CSV files.

For Streamlit Cloud deployment notes, see `docs/DEPLOYMENT.md`.

## Tests

Run the test suite from the repository root:

```bash
pytest
```

The tests cover metric functions, evidence ID parsing, Qasper normalization, and a mock LLM smoke path. They do not require an OpenAI key.

GitHub Actions also runs the test suite through `.github/workflows/tests.yml`. See `docs/TESTING.md` for details.

## Reproducibility Checklist

- Use the repository root as the working directory.
- Install `requirements.txt` in a fresh virtual environment.
- Keep `seed`, `dataset_split`, `max_documents`, and `max_questions_per_doc` fixed across methods.
- Keep `embedding_model`, chunk sizes, `top_k`, TOC depth, and hybrid settings fixed when comparing methods.
- Use `--match-question-ids-from` for paired fairness across methods.
- Run `experiments/analyze_outputs.py` after regenerating predictions.
- Record backend/model changes, because OpenAI-compatible APIs can change over time.
- Do not commit `.env` files or API keys.

For the full reproduction sequence, see `docs/REPRODUCIBILITY.md`.

## Main Findings

- Retrieval-based methods substantially outperform No-RAG on grounding and answer quality.
- Vector-RAG achieves the strongest quality and grounding metrics but uses the most prompt tokens.
- TOC-RAG provides a lower-cost structure-aware baseline.
- Hybrid-RAG keeps TOC-level efficiency while substantially improving citation reliability over TOC-RAG.
- Paired bootstrap analysis supports the major metric gaps on the aligned experiment slice.
