"""
Grid over a few retrieval hyperparameters; writes one summary JSON per setting.
Usage from repo root: python experiments/run_ablation.py
"""

from __future__ import annotations

import itertools
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from data.qasper_loader import expand_qasper_rows, load_qasper_split, make_dev_subset
from evaluation.metrics import aggregate_efficiency
from generator.llm_client import build_llm_from_config
from main import build_answer_fn, evaluate_output
from reports.export import save_run_summary
from utils.logger import get_logger
from utils.seed import set_seed


def main() -> None:
    cfg_path = ROOT / "experiments" / "config.yaml"
    base = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    set_seed(int(base.get("seed", 42)))
    log = get_logger("ablation", ROOT / base.get("log_dir", "outputs/logs") / "ablation.log")

    ds = load_qasper_split(base.get("dataset_split", "train"))
    ds = make_dev_subset(ds, max_samples=int(base.get("max_documents", 2)))
    llm = build_llm_from_config(base)
    q_cap = int(base.get("max_questions_per_doc", 3))

    grid = {
        "vector_rag": list(
            itertools.product(
                [3, 5],
                [200, 400],
            )
        ),
        "toc_rag": list(itertools.product([3, 5], ["bm25", "hash_embed"])),
    }
    out_dir = ROOT / base.get("output_dir", "outputs/predictions") / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    for method, combos in grid.items():
        for combo in combos:
            cfg = deepcopy(base)
            if method == "vector_rag":
                cfg["vector_top_k"], cfg["chunk_max_words"] = combo
                tag = f"k{combo[0]}_cw{combo[1]}"
            else:
                cfg["toc_max_depth"], cfg["toc_selection"] = combo
                tag = f"d{combo[0]}_{combo[1]}"
            answer_fn = build_answer_fn(method, llm, cfg)
            outputs = []
            metrics_acc: list[dict] = []
            for row in ds:
                for sample in list(expand_qasper_rows(row))[:q_cap]:
                    out = answer_fn(sample)
                    outputs.append(out)
                    metrics_acc.append(evaluate_output(out, sample, cfg=cfg, llm=llm))
            n = len(metrics_acc)

            llm_h_vals = [m.get("llm_hallucination", -1.0) for m in metrics_acc if m.get("llm_hallucination", -1.0) >= 0.0]
            llm_h_rate = -1.0 if not llm_h_vals else sum(llm_h_vals) / len(llm_h_vals)

            summary = {
                "method": method,
                "tag": tag,
                "config": {k: cfg[k] for k in cfg if k in ("vector_top_k", "chunk_max_words", "toc_max_depth", "toc_selection")},
                "n": n,
                "mean_em": sum(m["em"] for m in metrics_acc) / max(n, 1),
                "mean_f1": sum(m["f1"] for m in metrics_acc) / max(n, 1),
                "mean_evidence_hit": sum(m["evidence_hit_rate"] for m in metrics_acc) / max(n, 1),
                "mean_abstain": sum(m.get("abstain", 0.0) for m in metrics_acc) / max(n, 1),
                "mean_citation_precision": sum(m.get("citation_precision", 0.0) for m in metrics_acc) / max(n, 1),
                "avg_n_citations": sum(m.get("n_citations", 0.0) for m in metrics_acc) / max(n, 1),
                "llm_hallucination_rate": llm_h_rate,
                "toc_avg_nav_steps": sum(m.get("toc_nav_steps", 0.0) for m in metrics_acc) / max(n, 1),
                "toc_early_stop_rate": sum(m.get("toc_early_stop", 0.0) for m in metrics_acc) / max(n, 1),
                **aggregate_efficiency(outputs),
            }
            save_run_summary(summary, out_dir / f"{method}_{tag}.json")
            log.info("saved %s", summary)


if __name__ == "__main__":
    main()
