from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import yaml

from data.qasper_loader import expand_qasper_rows, load_qasper_split, make_dev_subset
from evaluation.citation_eval import citation_hits_retrieved
from evaluation.metrics import best_answer_metrics, evidence_hit_rate
from evaluation.hallucination_eval import heuristic_hallucination
from generator.llm_client import build_llm_from_config
from parser.doc_parser import parse_sample_document
from retrievers.base import SystemOutput
from retrievers.no_rag import NoRAG
from retrievers.toc_rag import make_toc_rag
from retrievers.vector_rag import VectorRAG, build_embedder
from evaluation.metrics import aggregate_efficiency
from reports.export import save_predictions_csv, save_run_summary
from utils.logger import get_logger
from utils.seed import set_seed


def evaluate_output(output: SystemOutput, sample: dict, use_heuristic_hallucination: bool = True) -> dict:
    gold = sample.get("answers") or []
    m = best_answer_metrics(output.answer, gold)
    ev = evidence_hit_rate(output.retrieved_contexts, sample.get("evidence") or [])
    cit = citation_hits_retrieved(output.citations, output.retrieved_ids)
    hall = 0.0
    if use_heuristic_hallucination:
        hall = 1.0 if heuristic_hallucination(output.answer, output.retrieved_contexts) else 0.0
    return {
        "em": m["em"],
        "f1": m["f1"],
        "evidence_hit_rate": ev,
        "citation_hit_rate": cit,
        "heuristic_hallucination": hall,
    }


def build_answer_fn(method: str, llm, cfg: dict) -> Callable[[dict], SystemOutput]:
    if method == "no_rag":
        rag = NoRAG(llm)
        return lambda s: rag.answer(s["question"], doc_id=s.get("doc_id", ""))

    if method == "vector_rag":
        embedder = build_embedder(cfg)
        rag = VectorRAG(embedder, llm)

        def _vec(s: dict) -> SystemOutput:
            doc = parse_sample_document(
                s,
                chunk_max_words=int(cfg.get("chunk_max_words", 400)),
                chunk_overlap_words=int(cfg.get("chunk_overlap_words", 50)),
            )
            rag.build_index(doc.vector_chunks)
            return rag.answer(
                s["question"],
                top_k=int(cfg.get("vector_top_k", 5)),
                doc_id=s.get("doc_id", ""),
            )

        return _vec

    if method == "toc_rag":
        rag = make_toc_rag(llm, cfg)

        def _toc(s: dict) -> SystemOutput:
            doc = parse_sample_document(
                s,
                chunk_max_words=int(cfg.get("chunk_max_words", 400)),
                chunk_overlap_words=int(cfg.get("chunk_overlap_words", 50)),
            )
            return rag.answer(s["question"], doc.toc_root, doc_id=s.get("doc_id", ""))

        return _toc

    raise ValueError(f"Unknown method: {method}")


def pipeline(sample: dict, answer_fn: Callable[[dict], SystemOutput], method: str, cfg: dict) -> tuple[SystemOutput, dict]:
    out = answer_fn(sample)
    metrics = evaluate_output(out, sample)
    out.extra["method"] = method
    return out, metrics


def run_experiment(
    method_name: str, dataset, cfg: dict, llm
) -> tuple[list[SystemOutput], list[dict], list[dict]]:
    answer_fn = build_answer_fn(method_name, llm, cfg)
    outputs: list[SystemOutput] = []
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    q_cap = int(cfg.get("max_questions_per_doc", 100))
    for row in dataset:
        for sample in list(expand_qasper_rows(row))[:q_cap]:
            out, m = pipeline(sample, answer_fn, method_name, cfg)
            outputs.append(out)
            metric_rows.append({"doc_id": sample["doc_id"], "question_id": sample["question_id"], **m})
            prediction_rows.append(
                {
                    "doc_id": sample["doc_id"],
                    "question_id": sample["question_id"],
                    "question": sample["question"],
                    "pred": out.answer,
                    "gold": sample["answers"],
                    "method": method_name,
                    **m,
                }
            )
    return outputs, metric_rows, prediction_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Qasper RAG experiment entrypoint")
    ap.add_argument("--method", choices=("no_rag", "vector_rag", "toc_rag"), default="no_rag")
    ap.add_argument("--config", type=Path, default=Path("experiments/config.yaml"))
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))

    log = get_logger("main", root / cfg.get("log_dir", "outputs/logs") / "main.log")
    ds = load_qasper_split(cfg.get("dataset_split", "train"))
    ds = make_dev_subset(ds, max_samples=int(cfg.get("max_documents", 3)))

    llm = build_llm_from_config(cfg)
    outputs, rows, pred_rows = run_experiment(args.method, ds, cfg, llm)
    log.info("Ran %d QA instances with %s", len(outputs), args.method)
    if rows:
        log.info(
            "mean em=%.3f f1=%.3f evidence_hit=%.3f",
            sum(r["em"] for r in rows) / len(rows),
            sum(r["f1"] for r in rows) / len(rows),
            sum(r["evidence_hit_rate"] for r in rows) / len(rows),
        )

    pred_dir = (root / cfg.get("output_dir", "outputs/predictions")).resolve()
    csv_path = pred_dir / f"{args.method}_predictions.csv"
    json_path = pred_dir / f"{args.method}_summary.json"
    save_predictions_csv(pred_rows, csv_path)
    if pred_rows:
        summary = {
            "method": args.method,
            "n": len(pred_rows),
            "answer_em": sum(r["em"] for r in pred_rows) / len(pred_rows),
            "answer_f1": sum(r["f1"] for r in pred_rows) / len(pred_rows),
            "evidence_hit_rate": sum(r["evidence_hit_rate"] for r in pred_rows) / len(pred_rows),
            "citation_hit_rate": sum(r["citation_hit_rate"] for r in pred_rows) / len(pred_rows),
            **aggregate_efficiency(outputs),
        }
        save_run_summary(summary, json_path)
    log.info("Wrote predictions CSV: %s", csv_path)
    if pred_rows:
        log.info("Wrote summary JSON: %s", json_path)


if __name__ == "__main__":
    main()
