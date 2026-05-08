# Deployment Guide

The project includes a Streamlit replay demo in `demo_app.py`.

## Local demo

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate or include the four prediction CSVs:

```text
outputs/predictions/no_rag_predictions.csv
outputs/predictions/vector_rag_predictions.csv
outputs/predictions/toc_rag_predictions.csv
outputs/predictions/hybrid_rag_predictions.csv
```

Run from the repository root:

```bash
streamlit run demo_app.py
```

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the entrypoint file to `demo_app.py`.
5. Make sure `requirements.txt` is present at the repository root.
6. Include the four prediction CSVs in the deployed repository if the app should show historical results immediately.

## Demo behavior

The app is a historical replay demo. It reads saved CSV outputs from `outputs/predictions/` and compares four systems on the same selected question:

- No RAG
- Vector RAG
- TOC-Based RAG
- Hybrid RAG

The sidebar controls document the experiment setup but do not recompute answers live. To update answers, rerun the baseline scripts and redeploy the regenerated CSV files.

## Common deployment issues

- Missing `experiments/config.yaml`: the app shows a configuration error.
- Missing prediction CSVs: the app explains which method outputs are required.
- Dataset loading failure: check HuggingFace/network access and `dataset_split`.
- GitHub URL typo or Streamlit subdomain typo: update the app URL in Streamlit Cloud settings.
