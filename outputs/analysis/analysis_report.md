# Qasper RAG 分析报告（Markdown版）

本文件是对 `experiments/analyze_outputs.py` 产物的文字化分析，作为 ipynb 分析内容的 Markdown 版本。

## 1. 数据与目标

- 数据来源：`outputs/predictions/no_rag_predictions.csv`、`outputs/predictions/toc_rag_predictions.csv`、`outputs/predictions/vector_rag_predictions.csv`
- 对齐方式：按 `question_id` 做配对（paired）比较
- 任务目标：
  - 比较三种方法（No-RAG / TOC-RAG / Vector-RAG）的整体表现
  - 做配对差值分析（同题比较）
  - 做显著性检验（paired bootstrap）
  - 展示效果-成本权衡（cost-quality trade-off）

## 2. 分析步骤（对应脚本2-6步）

1. 读取三份预测结果并按 `question_id` 对齐  
2. 计算每种方法总体均值指标  
3. 计算方法两两配对差值（mean/median/win-rate）  
4. 对核心质量指标做 paired bootstrap（5000次重采样）  
5. 绘制成本-效果散点图（F1 与 Citation Hit）  

输出文件：

- `aligned_predictions.csv`
- `overall_metrics.csv`
- `paired_differences.csv`
- `paired_bootstrap_tests.json`
- `fig_cost_vs_quality_f1.png`
- `fig_cost_vs_citation.png`

## 3. 三种方法区别（No-RAG vs Vector-RAG vs TOC-RAG）

为便于理解结果，先说明三种方法在“是否检索、如何检索、成本结构”上的核心区别。

- **No-RAG**
  - 流程：不做外部检索，模型仅依赖参数记忆直接回答。
  - 优点：实现最简单、token 成本最低、延迟通常最小。
  - 缺点：对长文档任务几乎没有 grounding，容易答非所问或拒答。
  - 适用：快速 baseline、无需文档证据支撑的轻量任务。

- **Vector-RAG**
  - 流程：先把文档切块并向量化，再按相似度检索 top-k 片段供模型回答。
  - 优点：召回能力强，通常在答案质量、证据命中、引用命中上表现最好。
  - 缺点：需要较多上下文拼接，Prompt Tokens 成本高；对文档结构显式利用较弱。
  - 适用：以“效果优先”为目标，且可接受较高推理成本的场景。

- **TOC-RAG**
  - 流程：利用文档目录/层级结构进行逐层导航，再在选中章节内检索与回答。
  - 优点：上下文更聚焦、token 成本显著低于大规模向量拼接，解释性较好。
  - 缺点：若导航路径偏离关键章节，可能损失召回，导致质量低于 Vector-RAG。
  - 适用：希望在“质量与成本”间取得折中、且文档结构较规范的场景。

一句话对比：

- No-RAG：**最低成本基线**
- Vector-RAG：**最高质量方案**
- TOC-RAG：**结构化折中方案**

## 4. 指标讲解（How to Read Metrics）

本项目的指标可以分为四类：答案质量、证据与引用可靠性、幻觉与保守性、成本效率。

- **EM（Exact Match）**  
  预测答案与任一标准答案在归一化后完全一致记为 1，否则为 0。  
  取值范围 `[0,1]`，越高越好。  
  由于开放式问答存在同义改写，EM 往往偏严格，通常与 F1 结合解读。

- **F1（Token-level F1）**  
  预测答案与标准答案的 token 级重叠程度（取与多个 gold 中最优者）。  
  取值范围 `[0,1]`，越高越好。  
  这是当前实验里最稳健的主质量指标。

- **Evidence Hit Rate**  
  检索到的上下文中命中 gold evidence 的比例。  
  取值范围 `[0,1]`，越高越好。  
  反映“检索是否找到了支持答案的证据”。

- **Citation Hit Rate / Citation Precision**  
  模型给出的 citation 是否落在检索到的证据集合中。  
  取值范围 `[0,1]`，越高越好。  
  在本实验输出中二者数值一致，可作为“引用可信度”指标。

- **Heuristic Hallucination**  
  基于规则判断答案是否包含未被检索证据支持的内容。  
  取值范围 `[0,1]`，越低越好。  
  它是启发式指标，适合做相对比较，不宜单独作为最终结论。

- **Abstain Rate**  
  模型是否回答 “I do not know” 等拒答形式。  
  取值范围 `[0,1]`，需结合任务目标解读：  
  拒答高可能降低幻觉，但也可能牺牲回答覆盖率。

- **Prompt Tokens / Completion Tokens**  
  分别表示输入 token 成本与输出 token 成本。  
  越低越省成本。  
  在 RAG 场景下，Prompt Tokens 通常是主要成本来源。

解读原则：

- 质量优先看 `F1 + Evidence/Citation`；
- 可靠性看 `Heuristic Hallucination`（越低越好）；
- 经济性看 `Prompt Tokens`；
- 最终采用“质量-可靠性-成本”三维联合判断，不单看一个指标。

## 5. 总体结果（Overall Metrics）

基于 `overall_metrics.csv`：

- **No-RAG**
  - F1: `0.0175`
  - Evidence Hit Rate: `0.0000`
  - Citation Hit Rate: `0.0000`
  - Heuristic Hallucination: `0.4447`
  - Avg Prompt Tokens: `42.24`

- **TOC-RAG**
  - F1: `0.0910`
  - Evidence Hit Rate: `0.1521`
  - Citation Hit Rate: `0.1935`
  - Heuristic Hallucination: `0.2337`
  - Avg Prompt Tokens: `531.03`

- **Vector-RAG**
  - F1: `0.1233`
  - Evidence Hit Rate: `0.4237`
  - Citation Hit Rate: `0.7663`
  - Heuristic Hallucination: `0.0075`
  - Avg Prompt Tokens: `2680.34`

初步解读：

- 两种 RAG 方法均显著优于 No-RAG（质量和 grounding 指标）。
- Vector-RAG 在质量指标上最好，但 token 成本最高。
- TOC-RAG 质量介于两者之间，成本远低于 Vector-RAG，体现出明显“中间解”特征。

## 6. 配对差值结果（Paired Differences）

基于 `paired_differences.csv`，重点关注 `vector_rag - toc_rag`：

- F1 平均差：`+0.0323`
- Evidence Hit Rate 平均差：`+0.2716`
- Citation Hit Rate 平均差：`+0.5729`
- Heuristic Hallucination 平均差：`-0.2261`（越低越好，说明 Vector 更低）
- Prompt Tokens 平均差：`+2149.31`

同题胜率（vector 相对 toc）：

- F1 win-rate: `0.4095`，lose-rate: `0.2286`，其余为 tie
- Citation win-rate: `0.5879`，lose-rate: `0.0151`

解读：

- Vector-RAG 在更多题目上取得更高引用命中与证据支持，但代价是显著更高的 token 消耗。

## 7. 显著性检验（Paired Bootstrap）

基于 `paired_bootstrap_tests.json`（5000 bootstrap）：

- **vector_rag - toc_rag**
  - F1: mean diff `0.0323`, 95% CI `[0.0201, 0.0452]`
  - Evidence Hit: mean diff `0.2716`, 95% CI `[0.2218, 0.3212]`
  - Citation Hit: mean diff `0.5729`, 95% CI `[0.5201, 0.6231]`

- **vector_rag - no_rag** 与 **toc_rag - no_rag** 的核心指标差值 CI 也均不跨 0。

统计结论：

- 在当前评测集上，核心质量指标差异具有统计显著性（CI 不跨 0）。

## 8. 成本-效果权衡图

- 图1：`fig_cost_vs_quality_f1.png`
  - x轴：平均 Prompt Tokens
  - y轴：平均 F1
- 图2：`fig_cost_vs_citation.png`
  - x轴：平均 Prompt Tokens
  - y轴：平均 Citation Hit Rate

图像结论：

- No-RAG 成本最低但效果最差。
- Vector-RAG 效果最佳但成本最高。
- TOC-RAG 在效果与成本之间形成可解释的折中点。

## 9. 对研究问题（RQ）的对应结论

- **RQ1（结构检索是否优于相似度检索）**  
  在当前数据上，Vector-RAG 的质量指标高于 TOC-RAG；TOC-RAG 并未超过 Vector-RAG，但具备成本优势。

- **RQ2（对可靠性/幻觉的影响）**  
  两类 RAG 都优于 No-RAG；其中 Vector-RAG 的 evidence/citation 指标最高，heuristic hallucination 最低。

- **RQ3（效果-成本权衡）**  
  TOC-RAG 是更低成本的折中方案；Vector-RAG 是更高性能但高成本方案。

## 10. 可复用命令

在项目根目录执行：

```bash
python experiments/analyze_outputs.py
```

重新生成全部分析产物。

## 11. 备注

- 当前 EM 为 0，建议以 F1、evidence、citation 为主要结论指标。
- 本报告为自动分析结果的文字化汇总，可直接纳入最终报告的结果章节初稿。
