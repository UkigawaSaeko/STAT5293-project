# Does Explicit Document Structure Beat Similarity-Based Retrieval for Long-Document QA?

## Abstract

Long-document question answering remains challenging for LLM systems because relevant evidence is sparse, distributed across sections, and easy to miss in long contexts. This project evaluates whether structure-aware retrieval can improve grounded QA performance compared with standard similarity retrieval and no retrieval. We conduct a controlled comparison on Qasper with three paradigms: `no_rag`, `vector_rag`, and `toc_rag`. We evaluate answer quality (EM, F1), grounding reliability (evidence hit rate, citation hit rate, citation precision), hallucination-related behavior (heuristic hallucination, abstain), and efficiency (prompt/completion tokens). Using paired analysis over aligned `question_id`s and paired bootstrap confidence intervals (`n_boot=5000`), we find that both retrieval-based methods outperform no retrieval, and Vector-RAG achieves the strongest quality and grounding performance. TOC-RAG does not surpass Vector-RAG on core quality metrics in the current setup, but provides a clear cost-performance trade-off with much lower prompt-token usage. These findings suggest that explicit structure-aware retrieval is a practical middle-ground design for budget-sensitive deployments, while similarity retrieval remains preferable when quality is the primary objective.

## 1. Introduction

### 1.1 Background and Problem Context

Retrieval-augmented generation (RAG) is a common strategy to improve factuality in LLM systems. However, long-document QA introduces additional challenges: key evidence may be buried in the middle of long contexts, naive chunk retrieval can miss cross-section dependencies, and high-recall retrieval can sharply increase token cost. These issues make retrieval design a central systems question, rather than a simple implementation detail.

This project studies whether explicit document structure can improve retrieval effectiveness in long-document QA. We focus on a practical comparison between chunk-based similarity retrieval and TOC-guided structure-aware retrieval under a unified evaluation pipeline.

### 1.2 Research Questions

- **RQ1**: Can TOC-based retrieval outperform vector retrieval in answer quality and reliability?
- **RQ2**: How do retrieval paradigms affect hallucination tendency and citation reliability?
- **RQ3**: What are the quality-cost trade-offs among No-RAG, Vector-RAG, and TOC-RAG?

### 1.3 Contributions

- A unified experimental framework comparing `no_rag`, `vector_rag`, and `toc_rag` on the same QA set.
- A structure-aware TOC navigation baseline designed as a practical middle-ground approach.
- Paired comparison and bootstrap significance testing over aligned question-level predictions.
- A multi-dimensional view of performance that jointly analyzes quality, grounding, and cost.

## 2. Related Work

Prior work on long-context LLMs and retrieval emphasizes that context length alone does not guarantee robust evidence utilization, especially for information placed deep inside documents. Standard RAG pipelines typically rely on chunking plus embedding similarity, which is effective but can be costly and structure-agnostic. Recent structure-aware and hierarchical retrieval paradigms attempt to improve retrieval focus by using document organization. Graph-based retrieval can further increase reasoning capacity but often introduces additional engineering and computational overhead.

Our work positions TOC-guided retrieval as a lightweight structure-aware alternative and evaluates it directly against similarity-based retrieval and no-retrieval baseline in one controlled setup.

> Note: insert formal citations here (course paper list + external references) and ensure consistent bibliography style.

## 3. Methodology

### 3.1 Task and Dataset

We evaluate long-document QA on Qasper. Each sample contains a question, one or more reference answers, and evidence annotations. The evaluation pipeline aligns predictions by `question_id` to enable paired comparisons across methods.

### 3.2 System Overview

All methods share the same generation and evaluation framework, differing only in retrieval strategy:

- Parse input document and question.
- Retrieve supporting context (if retrieval is used).
- Generate answer with citations.
- Compute quality, grounding, hallucination, and efficiency metrics.

### 3.3 Compared Methods

- **No-RAG (`no_rag`)**: direct generation with no external retrieval.
- **Vector-RAG (`vector_rag`)**: chunk document, build vector index, retrieve top-k similar chunks.
- **TOC-RAG (`toc_rag`)**: navigate a table-of-contents hierarchy, then retrieve/generate from selected sections.

### 3.4 Experimental Controls and Fairness

The comparison uses the same question set and aligned `question_id`s across methods. We keep model/backend and evaluation logic consistent and report both quality and cost metrics to avoid biased conclusions based on a single objective.

### 3.5 Metrics

- **Answer quality**: EM, F1
- **Grounding reliability**: evidence hit rate, citation hit rate, citation precision
- **Behavior/safety proxy**: heuristic hallucination, abstain rate
- **Efficiency**: prompt tokens, completion tokens

### 3.6 Statistical Testing

We conduct paired comparisons over aligned predictions and use paired bootstrap (`n_boot=5000`) for confidence intervals on core metrics (`f1`, `evidence_hit_rate`, `citation_hit_rate`).

## 4. Experimental Setup

### 4.1 Implementation Details

Core scripts:

- Baselines: `experiments/run_baseline.py`
- Ablation: `experiments/run_ablation.py`
- Analysis: `experiments/analyze_outputs.py`
- Unified pipeline/metrics: `main.py`, `evaluation/`, `retrievers/`

Configuration is managed through `experiments/config.yaml` (seed, dataset slice, retrieval hyperparameters, model backend).

### 4.2 Baseline and Ablation Settings

Baselines are run once per method:

```bash
python experiments/run_baseline.py --method no_rag
python experiments/run_baseline.py --method vector_rag
python experiments/run_baseline.py --method toc_rag
```

Ablation grid includes:

- Vector-RAG: `top_k ∈ {3,5}`, `chunk_max_words ∈ {200,400}`
- TOC-RAG: `toc_max_depth ∈ {3,5}`, `toc_selection ∈ {bm25, hash_embed}`

## 5. Results

### 5.1 Overall Comparison

From `outputs/analysis/overall_metrics.csv` (n=398 per method):

- **No-RAG**: F1 `0.0175`, evidence hit `0.0000`, citation hit `0.0000`, heuristic hallucination `0.4447`, avg prompt tokens `42.24`.
- **TOC-RAG**: F1 `0.0910`, evidence hit `0.1521`, citation hit `0.1935`, heuristic hallucination `0.2337`, avg prompt tokens `531.03`.
- **Vector-RAG**: F1 `0.1233`, evidence hit `0.4237`, citation hit `0.7663`, heuristic hallucination `0.0075`, avg prompt tokens `2680.34`.

Interpretation:

1. Both retrieval methods significantly improve over No-RAG on quality and grounding.
2. Vector-RAG provides the highest quality/reliability.
3. TOC-RAG is substantially cheaper than Vector-RAG while clearly outperforming No-RAG.

### 5.2 Paired Differences

From `outputs/analysis/paired_differences.csv`, for `vector_rag - toc_rag`:

- F1 mean diff: `+0.0323` (win `0.4095`, lose `0.2286`, tie `0.3618`)
- Evidence hit diff: `+0.2716`
- Citation hit diff: `+0.5729` (win `0.5879`, lose `0.0151`)
- Heuristic hallucination diff: `-0.2261` (lower is better; Vector lower)
- Prompt tokens diff: `+2149.31`

This indicates a strong quality/reliability advantage for Vector-RAG but with much higher input-token cost.

### 5.3 Statistical Significance

From `outputs/analysis/paired_bootstrap_tests.json`:

- `vector_rag - toc_rag`, F1 mean diff `0.0323`, 95% CI `[0.0201, 0.0452]`
- `vector_rag - toc_rag`, evidence hit mean diff `0.2716`, 95% CI `[0.2218, 0.3212]`
- `vector_rag - toc_rag`, citation hit mean diff `0.5729`, 95% CI `[0.5201, 0.6231]`

All above CIs exclude zero. Similarly, `vector_rag - no_rag` and `toc_rag - no_rag` core-metric CIs also exclude zero, supporting statistically robust performance differences on the evaluated slice.

### 5.4 Cost-Quality Trade-off

The two analysis figures (`fig_cost_vs_quality_f1.png`, `fig_cost_vs_citation.png`) show a consistent frontier:

- No-RAG: lowest cost, lowest quality.
- Vector-RAG: highest quality, highest cost.
- TOC-RAG: middle point with meaningful quality gains over No-RAG at far lower cost than Vector-RAG.

For budget-constrained settings, TOC-RAG is a practical compromise; for quality-critical settings, Vector-RAG is preferable.

## 6. Error Analysis and Case Study

To complement aggregate statistics, we include three representative question-level cases from the prediction outputs.

### Case 1: Vector-RAG clearly outperforms TOC-RAG

- **Question ID**: `b43a8a0f4b8496b23c89730f0070172cd5dca06a` (`doc_id=1805.12032`)
- **Question**: "What is the architecture of their model?"
- **Scores**:
  - No-RAG: F1 `0.0000`, evidence hit `0.0`, citation hit `0.0`, prompt tokens `40`
  - Vector-RAG: F1 `0.8023`, evidence hit `1.0`, citation hit `1.0`, prompt tokens `2612`
  - TOC-RAG: F1 `0.0777`, evidence hit `0.0`, citation hit `0.0`, prompt tokens `253`
- **Observation**: Vector-RAG retrieves the detailed architecture description and produces a near-gold answer, while TOC-RAG fails to navigate to the relevant section in this instance.

### Case 2: TOC-RAG provides near-equal quality at much lower cost

- **Question ID**: `ad16c8261c3a0b88c685907387e1a6904eb15066` (`doc_id=1906.09774`)
- **Question**: "What are the research questions posed in the paper regarding EAC studies?"
- **Scores**:
  - No-RAG: F1 `0.0000`, evidence hit `0.0`, citation hit `0.0`, prompt tokens `46`
  - Vector-RAG: F1 `0.7077`, evidence hit `1.0`, citation hit `1.0`, prompt tokens `2506`
  - TOC-RAG: F1 `0.7097`, evidence hit `1.0`, citation hit `0.0`, prompt tokens `584`
- **Observation**: TOC-RAG reaches almost identical answer quality to Vector-RAG on this question while using approximately 4.3x fewer prompt tokens, illustrating the practical cost-quality trade-off.

### Case 3: Failure case with contradictory behavior

- **Question ID**: `8c0621016e96d86a7063cb0c9ec20c76a2dba678` (`doc_id=1806.00722`)
- **Question**: "did they outperform previous methods?"
- **Gold answer**: "yes"
- **Scores**:
  - No-RAG: F1 `0.0000`, evidence hit `0.0`, citation hit `0.0`, prompt tokens `38`
  - Vector-RAG: F1 `0.0000`, evidence hit `0.0`, citation hit `1.0`, prompt tokens `2455`
  - TOC-RAG: F1 `0.0000`, evidence hit `0.0`, citation hit `0.0`, prompt tokens `264`
- **Observation**: Vector-RAG generates a detailed claim but still gets zero F1, suggesting mismatch between generated statement and annotated gold target. This case highlights that citation presence alone does not guarantee answer correctness.

These examples support the quantitative findings: Vector-RAG is strongest on difficult retrieval-heavy questions, TOC-RAG can deliver strong cost-efficiency on many items, and both methods still show failure modes under ambiguous phrasing or retrieval mismatch.

### Case Summary Table (for Presentation)

| Case | Question ID | Pattern | F1 (No / Vec / TOC) | Evidence Hit (No / Vec / TOC) | Citation Hit (No / Vec / TOC) | Prompt Tokens (No / Vec / TOC) | Takeaway |
|---|---|---|---|---|---|---|---|
| Case 1 | `b43a8a0f4b8496b23c89730f0070172cd5dca06a` | Vector advantage | `0.000 / 0.802 / 0.078` | `0.0 / 1.0 / 0.0` | `0.0 / 1.0 / 0.0` | `40 / 2612 / 253` | Vector retrieves key architecture evidence; TOC misses relevant section. |
| Case 2 | `ad16c8261c3a0b88c685907387e1a6904eb15066` | TOC trade-off | `0.000 / 0.708 / 0.710` | `0.0 / 1.0 / 1.0` | `0.0 / 1.0 / 0.0` | `46 / 2506 / 584` | TOC matches Vector answer quality with much lower token cost. |
| Case 3 | `8c0621016e96d86a7063cb0c9ec20c76a2dba678` | Failure mode | `0.000 / 0.000 / 0.000` | `0.0 / 0.0 / 0.0` | `0.0 / 1.0 / 0.0` | `38 / 2455 / 264` | Citation presence does not guarantee answer correctness. |

## 7. Discussion

### 7.1 Answers to RQ1-RQ3

- **RQ1**: In this setup, TOC-RAG does not outperform Vector-RAG on core quality/reliability metrics.
- **RQ2**: Retrieval substantially improves grounding reliability versus No-RAG; Vector-RAG is strongest and has the lowest hallucination proxy.
- **RQ3**: TOC-RAG offers a clear quality-cost middle ground; Vector-RAG offers best quality at significantly higher cost.

### 7.2 Practical Implications

- Prefer **Vector-RAG** when answer fidelity and citation reliability dominate cost concerns.
- Prefer **TOC-RAG** when token budget/latency constraints are important but No-RAG quality is insufficient.
- Use **No-RAG** primarily as a minimal baseline, not a deployment strategy for grounded long-document QA.

### 7.3 Limitations

- Evaluation currently reflects the configured slice and budget, not full-scale exhaustive benchmarking.
- Heuristic hallucination is a proxy metric and should be interpreted with caution.
- Results may vary with chunking settings, TOC construction quality, and retriever hyperparameters.

### 7.4 Future Work

- Hybrid retrieval (TOC-guided candidate narrowing + vector reranking).
- More robust TOC navigation policies and stopping criteria.
- Larger-scale evaluation and possibly user-centered qualitative assessment.

## 8. Conclusion

This project provides a controlled comparison of three retrieval paradigms for long-document QA. Results show that retrieval is essential for grounded performance, Vector-RAG delivers the strongest quality and reliability, and TOC-RAG offers a useful lower-cost compromise. The main practical takeaway is that retrieval strategy selection should be driven by deployment objective: quality-first systems should lean toward similarity retrieval, while cost-aware systems can benefit from structure-aware retrieval as a middle-ground design.

## References

Add final references in your chosen format (for example, ACL/APA/IEEE) and ensure all related-work claims are cited.

## Appendix

- Exact run commands and configs
- Additional ablation tables from `outputs/predictions/ablation/`
- Qualitative examples and error breakdown
- Reproducibility checklist
