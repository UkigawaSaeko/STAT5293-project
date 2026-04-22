# STAT5293-project

Does Explicit Document Structure Beat Similarity-Based Retrieval?

This project evaluates three long-document QA paradigms on Qasper:

- `no_rag`: answer directly from model parametric knowledge
- `vector_rag`: chunk-based similarity retrieval (Vector RAG)
- `toc_rag`: structure-aware retrieval using TOC navigation

The goal is to compare answer quality, grounding reliability, and inference cost under a unified pipeline.

## 1) Project Structure

- `main.py`: unified experiment entrypoint
- `experiments/run_baseline.py`: run one baseline method and save predictions/summary
- `experiments/run_ablation.py`: run a small hyperparameter grid for ablations
- `experiments/analyze_outputs.py`: paired analysis, bootstrap tests, and figures
- `experiments/config.yaml`: experiment/runtime configuration
- `retrievers/`: implementations of No-RAG / Vector-RAG / TOC-RAG
- `evaluation/`: metrics for answer quality, evidence/citation grounding, hallucination
- `outputs/predictions/`: raw prediction outputs
- `outputs/analysis/`: post-hoc analysis tables/figures/reports

## 2) Environment Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) API Configuration

The default backend is OpenAI (`llm_backend: openai` in `experiments/config.yaml`).

Set your key in shell (do not hardcode in yaml):

```bash
export OPENAI_API_KEY="your-key"
# Optional if using a compatible gateway
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

If you want a quick non-API sanity check, switch to mock backend in `experiments/config.yaml`:

```yaml
llm_backend: mock
```

## 4) Run Baseline Experiments

Run each method from project root:

```bash
python experiments/run_baseline.py --method no_rag
python experiments/run_baseline.py --method vector_rag
python experiments/run_baseline.py --method toc_rag
```

Expected outputs:

- `outputs/predictions/no_rag_predictions.csv`
- `outputs/predictions/vector_rag_predictions.csv`
- `outputs/predictions/toc_rag_predictions.csv`
- `outputs/predictions/*_summary.json`

## 5) Run Post-hoc Analysis

After the three baseline prediction CSVs exist:

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

## 6) Run Ablation Study

```bash
python experiments/run_ablation.py
```

This writes one summary JSON per setting to:

- `outputs/predictions/ablation/`

## 7) Reproducibility Notes

- Fix randomness with `seed` in `experiments/config.yaml`.
- Keep the same `dataset_split`, `max_documents`, and `max_questions_per_doc` across methods.
- Do not compare runs with different evaluation budgets.
- For cost/fairness reporting, include both quality metrics (`f1`, evidence/citation hit) and efficiency metrics (`prompt_tokens`, `completion_tokens`).

## 8) Suggested Reporting Claims (Aligned with Proposal)

- Both RAG variants outperform No-RAG on long-document grounding.
- Vector-RAG tends to achieve the strongest quality/grounding metrics.
- TOC-RAG provides a lower-cost trade-off point with structure-aware retrieval.
- Paired bootstrap CI supports significance of key metric gaps on current test slice.
