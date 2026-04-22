### Ablation summary (auto-generated)

Slice caps come from each JSON's `ablation_slice` (not the full 398-question baseline).

| method | tag | ablation_docs | ablation_q/doc | n | mean_f1 | mean_evidence_hit | mean_citation_hit | mean_citation_precision | avg_prompt_tok | avg_compl_tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_rag | hk2_hd3 | 12 | 2 | 20 | 0.1283 | 0.1639 | 0.6 | 0.6 | 450.95 | 40.1 |
| hybrid_rag | hk2_hd4 | 12 | 2 | 20 | 0.1283 | 0.1639 | 0.65 | 0.65 | 450.95 | 40.8 |
| hybrid_rag | hk3_hd3 | 12 | 2 | 20 | 0.1227 | 0.1639 | 0.7 | 0.7 | 499.1 | 45.6 |
| hybrid_rag | hk3_hd4 | 12 | 2 | 20 | 0.1248 | 0.1639 | 0.65 | 0.65 | 499.1 | 42.45 |
| toc_rag | d3_bm25 | 12 | 2 | 20 | 0.1221 | 0.1639 | 0.35 | 0.35 | 460.95 | 50.85 |
| toc_rag | d3_hash_embed | 12 | 2 | 20 | 0.0848 | 0.0625 | 0.3 | 0.3 | 565.95 | 44.15 |
| toc_rag | d5_bm25 | 12 | 2 | 20 | 0.1275 | 0.1639 | 0.4 | 0.4 | 460.95 | 48.1 |
| toc_rag | d5_hash_embed | 12 | 2 | 20 | 0.0846 | 0.0625 | 0.3 | 0.3 | 565.95 | 44.3 |
| vector_rag | k3_cw200 | 12 | 2 | 20 | 0.0845 | 0 | 0.55 | 0.55 | 920.35 | 53.1 |
| vector_rag | k3_cw400 | 12 | 2 | 20 | 0.1368 | 0.2 | 0.75 | 0.75 | 1649.8 | 60.7 |
| vector_rag | k5_cw200 | 12 | 2 | 20 | 0.0987 | 0.1125 | 0.7 | 0.7 | 1450.35 | 56.65 |
| vector_rag | k5_cw400 | 12 | 2 | 20 | 0.1492 | 0.4319 | 0.8 | 0.8 | 2685.05 | 70.4 |
