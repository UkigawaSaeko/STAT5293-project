"""
Run no_rag / vector_rag / toc_rag on a small Qasper slice; save predictions + summary metrics.
Usage from repo root: python experiments/run_baseline.py [--method no_rag|vector_rag|toc_rag]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from data.qasper_loader import expand_qasper_rows, load_qasper_split, make_dev_subset
from evaluation.metrics import aggregate_efficiency
from generator.llm_client import build_llm_from_config
from main import build_answer_fn, pipeline
from reports.export import save_predictions_csv, save_run_summary
from utils.logger import get_logger
from utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("no_rag", "vector_rag", "toc_rag"), default="no_rag")
    ap.add_argument("--config", type=Path, default=ROOT / "experiments" / "config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))

    log = get_logger("run_baseline", ROOT / cfg.get("log_dir", "outputs/logs") / "baseline.log")
    ds = load_qasper_split(cfg.get("dataset_split", "train"))
    ds = make_dev_subset(ds, max_samples=int(cfg.get("max_documents", 20)))

    llm = build_llm_from_config(cfg)
    answer_fn = build_answer_fn(args.method, llm, cfg)

    rows: list[dict] = []
    outputs = []
    doc_cap = int(cfg.get("max_documents", 20))
    q_cap = int(cfg.get("max_questions_per_doc", 100))
    nd = 0
    for row in ds:
        if nd >= doc_cap:
            break
        samples = list(expand_qasper_rows(row))[:q_cap]
        for sample in samples:
            out, metrics = pipeline(sample, answer_fn, args.method, cfg, llm)
            outputs.append(out)
            rows.append(
                {
                    "doc_id": sample["doc_id"],
                    "question_id": sample["question_id"],
                    "question": sample["question"],
                    "pred": out.answer,
                    "gold": sample["answers"],
                    "method": args.method,
                    **metrics,
                }
            )
        nd += 1

    pred_dir = ROOT / cfg.get("output_dir", "outputs/predictions")
    save_predictions_csv(rows, pred_dir / f"{args.method}_predictions.csv")

    avg_em = sum(r.get("em", 0) for r in rows) / max(len(rows), 1)
    avg_f1 = sum(r.get("f1", 0) for r in rows) / max(len(rows), 1)
    avg_ev = sum(r.get("evidence_hit_rate", 0) for r in rows) / max(len(rows), 1)
    avg_cit = sum(r.get("citation_hit_rate", 0) for r in rows) / max(len(rows), 1)
    eff = aggregate_efficiency(outputs)

    abst_vals = [r.get("abstain", 0.0) for r in rows]
    cit_prec_vals = [r.get("citation_precision", 0.0) for r in rows]
    n_cites_vals = [r.get("n_citations", 0.0) for r in rows]
    toc_steps_vals = [r.get("toc_nav_steps", 0.0) for r in rows]
    toc_early_vals = [r.get("toc_early_stop", 0.0) for r in rows]
    llm_vals = [r.get("llm_hallucination", -1.0) for r in rows if r.get("llm_hallucination", -1.0) >= 0.0]
    llm_rate = -1.0 if not llm_vals else sum(llm_vals) / len(llm_vals)

    summary = {
        "method": args.method,
        "n": len(rows),
        "answer_em": avg_em,
        "answer_f1": avg_f1,
        "evidence_hit_rate": avg_ev,
        "citation_hit_rate": avg_cit,
        "abstain_rate": sum(abst_vals) / max(len(abst_vals), 1),
        "citation_precision": sum(cit_prec_vals) / max(len(cit_prec_vals), 1),
        "avg_n_citations": sum(n_cites_vals) / max(len(n_cites_vals), 1),
        "toc_avg_nav_steps": sum(toc_steps_vals) / max(len(toc_steps_vals), 1),
        "toc_early_stop_rate": sum(toc_early_vals) / max(len(toc_early_vals), 1),
        "llm_hallucination_rate": llm_rate,
        **eff,
    }
    save_run_summary(summary, pred_dir / f"{args.method}_summary.json")
    log.info("Wrote %s and summary %s", pred_dir / f"{args.method}_predictions.csv", summary)


if __name__ == "__main__":
    main()
