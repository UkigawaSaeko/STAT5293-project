# Reproducibility Guide

This guide reproduces the main experiment artifacts from a fresh checkout.

## 1. Create environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure API access

For OpenAI-compatible generation:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

For a no-cost smoke check, set this in `experiments/config.yaml`:

```yaml
llm_backend: mock
```

## 3. Review fixed configuration

Main experiment settings live in `experiments/config.yaml`, including:

- `seed`
- `dataset_split`
- `max_documents`
- `max_questions_per_doc`
- `llm_model`
- `embedding_model`
- vector/TOC/hybrid hyperparameters
- output directories

The submitted experiment uses the Qasper `train` split as a controlled paired-analysis slice.

## 4. Run four baseline methods

From the repository root:

```bash
python experiments/run_baseline.py --method vector_rag
python experiments/run_baseline.py --method no_rag --match-question-ids-from outputs/predictions/vector_rag_predictions.csv
python experiments/run_baseline.py --method toc_rag --match-question-ids-from outputs/predictions/vector_rag_predictions.csv
python experiments/run_baseline.py --method hybrid_rag --match-question-ids-from outputs/predictions/vector_rag_predictions.csv
```

This produces:

```text
outputs/predictions/no_rag_predictions.csv
outputs/predictions/vector_rag_predictions.csv
outputs/predictions/toc_rag_predictions.csv
outputs/predictions/hybrid_rag_predictions.csv
outputs/predictions/*_summary.json
```

## 5. Run post-hoc analysis

```bash
python experiments/analyze_outputs.py
```

This produces:

```text
outputs/analysis/aligned_predictions.csv
outputs/analysis/overall_metrics.csv
outputs/analysis/paired_differences.csv
outputs/analysis/paired_bootstrap_tests.json
outputs/analysis/analysis_report.md
outputs/analysis/fig_cost_vs_quality_f1.png
outputs/analysis/fig_cost_vs_citation.png
```

## 6. Run ablations

```bash
python experiments/run_ablation.py
```

Ablation summaries are written to `outputs/predictions/ablation/`. They use a smaller slice and should not be directly compared with the main aligned baseline table.

## 7. Verify implementation

```bash
pytest
```

The tests use mock/local logic and do not require an API key.

## Notes

- Keep all compared methods on the same `question_id` set.
- Keep `embedding_model`, chunk sizes, `top_k`, and TOC depth fixed when comparing methods.
- OpenAI-compatible model behavior may change over time; exact numeric reproduction may require the same backend snapshot.
- `EM` is strict for free-form Qasper answers, so F1 and grounding metrics are the primary quality signals.
