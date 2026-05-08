# Architecture

The project is organized as a reproducible long-document QA pipeline.

## Data Flow

```text
Qasper dataset
  -> data/qasper_loader.py
  -> parser/doc_parser.py
  -> retrievers/{no_rag, vector_rag, toc_rag, hybrid_rag}.py
  -> generator/llm_client.py
  -> evaluation/*.py
  -> outputs/predictions/
  -> experiments/analyze_outputs.py
  -> outputs/analysis/
```

## Components

### Data loading

`data/qasper_loader.py` loads Qasper from HuggingFace parquet files and expands each paper into per-question samples. It normalizes:

- document metadata,
- full text and sections,
- gold answers,
- evidence strings,
- mapped evidence section indices.

### Parsing

`parser/` converts each normalized sample into:

- full-document chunks for Vector-RAG,
- a TOC tree for TOC-RAG and Hybrid-RAG,
- flat section metadata for analysis and retrieval.

### Retrieval methods

`retrievers/` contains four comparable methods:

- `NoRAG`: direct answer generation without document retrieval.
- `VectorRAG`: full-document chunk retrieval with FAISS.
- `TOCRAG`: section navigation using TOC structure and BM25/hash scoring.
- `HybridRAG`: TOC section selection followed by scoped vector retrieval.

### Generation

`generator/llm_client.py` defines:

- `MockLLMClient` for tests and offline smoke checks,
- `OpenAICompatibleClient` for OpenAI-compatible APIs,
- `parse_evidence_ids_line` for citation ID parsing.

### Evaluation

`evaluation/` computes:

- exact match and token F1,
- evidence hit rate,
- citation hit rate and precision,
- heuristic hallucination flags.

### Experiments and reporting

`experiments/run_baseline.py` runs one method and writes predictions. `experiments/analyze_outputs.py` aligns four methods, computes aggregate metrics, paired differences, paired bootstrap tests, and figures.

`demo_app.py` is a Streamlit replay app that reads saved prediction CSVs and displays four-method comparisons for selected questions.
