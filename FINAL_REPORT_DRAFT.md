# Does Explicit Document Structure Beat Similarity-Based Retrieval for Long-Document QA?

## Abstract

Long-document question answering remains challenging for LLM systems because relevant evidence is sparse, distributed across sections, and easy to miss in long contexts. This project evaluates whether structure-aware retrieval can improve grounded QA performance compared with standard similarity retrieval and no retrieval. We conduct a controlled comparison on Qasper with four paradigms: `no_rag`, `vector_rag`, `toc_rag`, and `hybrid_rag` (TOC-guided section selection followed by scoped vector retrieval). We evaluate answer quality (EM, F1), grounding reliability (evidence hit rate, citation hit rate, citation precision), hallucination-related behavior (heuristic hallucination, abstain), and efficiency (prompt/completion tokens). Using paired analysis over aligned `question_id`s and paired bootstrap confidence intervals (`n_boot=5000`), we find that all retrieval-based methods outperform no retrieval, and Vector-RAG achieves the strongest quality and grounding performance. Hybrid-RAG achieves near-TOC F1 at TOC-level cost but with substantially higher citation reliability than TOC-RAG, while still remaining below Vector-RAG. These findings suggest a three-tier trade-off frontier: Vector-RAG for maximum quality, Hybrid-RAG for stronger grounding under moderate budget, and TOC-RAG for simple low-cost structured retrieval.

## 1. Introduction

### 1.1 Background and Problem Context

Retrieval-augmented generation (RAG) is a common strategy to improve factuality in LLM systems. However, long-document QA introduces additional challenges: key evidence may be buried in the middle of long contexts, naive chunk retrieval can miss cross-section dependencies, and high-recall retrieval can sharply increase token cost. These issues make retrieval design a central systems question, rather than a simple implementation detail.

This project studies whether explicit document structure can improve retrieval effectiveness in long-document QA. We focus on a practical comparison between chunk-based similarity retrieval and TOC-guided structure-aware retrieval under a unified evaluation pipeline.

### 1.2 Research Questions

- **RQ1**: Can TOC-based retrieval outperform vector retrieval in answer quality and reliability?
- **RQ2**: How do retrieval paradigms affect hallucination tendency and citation reliability?
- **RQ3**: What are the quality-cost trade-offs among No-RAG, Vector-RAG, TOC-RAG, and Hybrid-RAG?

### 1.3 Contributions

- A unified experimental framework comparing `no_rag`, `vector_rag`, `toc_rag`, and `hybrid_rag` on the same QA set.
- A structure-aware TOC navigation baseline designed as a practical middle-ground approach.
- A hybrid two-stage retriever that combines TOC-guided narrowing with scoped vector retrieval.
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
- **Hybrid-RAG (`hybrid_rag`)**: first navigate with TOC, then run vector retrieval within the selected section.

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
python experiments/run_baseline.py --method hybrid_rag
```

Ablation grid includes:

- Vector-RAG: `top_k ∈ {3,5}`, `chunk_max_words ∈ {200,400}`
- TOC-RAG: `toc_max_depth ∈ {3,5}`, `toc_selection ∈ {bm25, hash_embed}`

## 5. Results

### 5.1 Overall Comparison

From `outputs/analysis/overall_metrics.csv` (n=398 per method):

- **No-RAG**: F1 `0.0181`, evidence hit `0.0000`, citation hit `0.0000`, heuristic hallucination `0.4623`, avg prompt tokens `42.24`.
- **TOC-RAG**: F1 `0.0910`, evidence hit `0.1521`, citation hit `0.1935`, heuristic hallucination `0.2337`, avg prompt tokens `531.03`.
- **Hybrid-RAG**: F1 `0.0913`, evidence hit `0.1422`, citation hit `0.5352`, heuristic hallucination `0.2538`, avg prompt tokens `521.44`.
- **Vector-RAG**: F1 `0.1233`, evidence hit `0.4237`, citation hit `0.7663`, heuristic hallucination `0.0075`, avg prompt tokens `2680.34`.

Interpretation:

1. All retrieval methods significantly improve over No-RAG on quality.
2. Vector-RAG provides the highest quality/reliability.
3. Hybrid-RAG and TOC-RAG are both substantially cheaper than Vector-RAG.
4. Hybrid-RAG improves citation reliability over TOC-RAG while keeping similar F1 and cost.

### 5.2 Paired Differences

From `outputs/analysis/paired_differences.csv`, key comparisons:

- F1 mean diff: `+0.0323` (win `0.4095`, lose `0.2286`, tie `0.3618`)
- Evidence hit diff: `+0.2716`
- Citation hit diff: `+0.5729` (win `0.5879`, lose `0.0151`)
- Heuristic hallucination diff: `-0.2261` (lower is better; Vector lower)
- Prompt tokens diff: `+2149.31`

For `hybrid_rag - toc_rag`:

- F1 mean diff: `+0.00024` (effectively tied)
- Evidence hit diff: `-0.00991` (hybrid slightly lower)
- Citation hit diff: `+0.34171` (hybrid substantially higher)
- Prompt tokens diff: `-9.59` (hybrid slightly cheaper)

This indicates a strong quality/reliability advantage for Vector-RAG but with much higher input-token cost.

### 5.3 Statistical Significance

From `outputs/analysis/paired_bootstrap_tests.json`:

- `vector_rag - toc_rag`, F1 mean diff `0.0323`, 95% CI `[0.0201, 0.0452]`
- `vector_rag - toc_rag`, evidence hit mean diff `0.2716`, 95% CI `[0.2218, 0.3212]`
- `vector_rag - toc_rag`, citation hit mean diff `0.5729`, 95% CI `[0.5201, 0.6231]`
- `hybrid_rag - toc_rag`, F1 mean diff `0.00024`, 95% CI `[-0.00558, 0.00542]` (not significant; CI crosses 0)
- `hybrid_rag - toc_rag`, citation hit mean diff `0.34171`, 95% CI `[0.29397, 0.39196]` (significant)

For comparisons where the 95% CI does not include zero, the mean difference is statistically supported. The `hybrid_rag` vs `toc_rag` F1 gap is not significant (CI includes zero), while the citation hit gap is. Core metrics for `vector_rag` and `toc_rag` vs `no_rag` have CIs that exclude zero.

### 5.4 Cost-Quality Trade-off

The two analysis figures (`fig_cost_vs_quality_f1.png`, `fig_cost_vs_citation.png`) show a consistent frontier:

- No-RAG: lowest cost, lowest quality.
- Vector-RAG: highest quality, highest cost.
- TOC-RAG: low-cost structured baseline with clear gains over No-RAG.
- Hybrid-RAG: TOC-level cost with much better citation reliability, forming a stronger middle option.

For budget-constrained settings, Hybrid-RAG is a practical compromise; for quality-critical settings, Vector-RAG is preferable.

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

- **RQ1**: In this setup, neither TOC-RAG nor Hybrid-RAG outperforms Vector-RAG on core quality.
- **RQ2**: Retrieval substantially improves grounding reliability versus No-RAG; Vector-RAG is strongest, while Hybrid-RAG improves citation reliability substantially over TOC-RAG.
- **RQ3**: Hybrid-RAG and TOC-RAG offer low-cost alternatives; Hybrid-RAG is the stronger citation-grounding trade-off, while Vector-RAG remains the high-performance option.

### 7.2 Practical Implications

- Prefer **Vector-RAG** when answer fidelity and citation reliability dominate cost concerns.
- Prefer **Hybrid-RAG** when token budget is constrained but citation reliability is still important.
- Prefer **TOC-RAG** when minimal complexity and low cost are prioritized.
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

This project provides a controlled comparison of four retrieval paradigms for long-document QA. Results show that retrieval is essential for grounded performance, Vector-RAG delivers the strongest quality and reliability, and Hybrid-RAG offers a useful budget-aware middle-ground by substantially improving citation reliability over TOC-RAG at similar cost. The main practical takeaway is that retrieval strategy selection should be driven by deployment objective: quality-first systems should lean toward Vector-RAG, while cost-aware systems can benefit from Hybrid-RAG or TOC-RAG depending on reliability requirements.

## References

Add final references in your chosen format (for example, ACL/APA/IEEE) and ensure all related-work claims are cited.

## Appendix

- Exact run commands and configs
- Additional ablation tables from `outputs/predictions/ablation/`
- Qualitative examples and error breakdown
- Reproducibility checklist
