"""
Load and normalize allenai/qasper for long-document QA experiments.
Schema: full_text = {section_name: [...], paragraphs: [[...], ...]}; qas = parallel dicts.
"""

from __future__ import annotations

from typing import Any, Iterator

from datasets import Dataset, load_dataset


def _sections_from_full_text(full_text: dict[str, Any]) -> list[dict[str, Any]]:
    names = full_text.get("section_name") or []
    paras = full_text.get("paragraphs") or []
    sections: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        plist = paras[i] if i < len(paras) else []
        text = "\n\n".join(p for p in plist if isinstance(p, str) and p.strip())
        sections.append(
            {
                "section_idx": i,
                "header": name or f"section_{i}",
                "text": text,
            }
        )
    return sections


def _full_text_string(sections: list[dict[str, Any]], abstract: str) -> str:
    parts = []
    if abstract and abstract.strip():
        parts.append(f"Abstract\n\n{abstract.strip()}")
    for s in sections:
        h = s["header"].strip() or "Section"
        if s["text"].strip():
            parts.append(f"{h}\n\n{s['text'].strip()}")
    return "\n\n".join(parts)


def _flatten_answers(answers_block: dict[str, Any]) -> list[dict[str, Any]]:
    """Qasper stores annotator answer dicts under answers[i]['answer'] (list of variants)."""
    inner = answers_block.get("answer") or []
    if not isinstance(inner, list):
        return []
    out: list[dict[str, Any]] = []
    for v in inner:
        if isinstance(v, dict):
            out.append(v)
    return out


def _gold_answer_strings(flat: list[dict[str, Any]]) -> list[str]:
    gold: list[str] = []
    for v in flat:
        if v.get("unanswerable"):
            gold.append("unanswerable")
            continue
        yn = v.get("yes_no")
        if yn is True:
            gold.append("yes")
        elif yn is False:
            gold.append("no")
        elif isinstance(yn, str) and yn.lower() in ("yes", "no"):
            gold.append(yn.lower())
        ff = (v.get("free_form_answer") or "").strip()
        if ff:
            gold.append(ff)
        for span in v.get("extractive_spans") or []:
            if isinstance(span, str) and span.strip():
                gold.append(span.strip())
    seen: set[str] = set()
    uniq: list[str] = []
    for g in gold:
        key = g.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(g)
    return uniq


def _evidence_strings(flat: list[dict[str, Any]]) -> list[str]:
    ev: list[str] = []
    for v in flat:
        for e in v.get("evidence") or []:
            if isinstance(e, str) and e.strip():
                ev.append(e.strip())
        for e in v.get("highlighted_evidence") or []:
            if isinstance(e, str) and e.strip():
                ev.append(e.strip())
    seen: set[str] = set()
    out: list[str] = []
    for s in ev:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _map_evidence_to_sections(
    evidence: list[str], sections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for ev in evidence:
        best_idx = -1
        best_len = 0
        norm_ev = ev.replace(" ", "").lower()
        for s in sections:
            body = s["text"]
            if not body:
                continue
            if ev in body or (len(norm_ev) > 20 and norm_ev[:80] in body.replace(" ", "").lower()):
                if len(body) > best_len:
                    best_len = len(body)
                    best_idx = s["section_idx"]
        mapped.append(
            {
                "evidence": ev,
                "section_idx": best_idx,
                "section_header": sections[best_idx]["header"] if best_idx >= 0 else None,
            }
        )
    return mapped


def normalize_document(row: dict[str, Any]) -> dict[str, Any]:
    full_text = row.get("full_text") or {}
    if not isinstance(full_text, dict):
        full_text = {}
    abstract = row.get("abstract") or ""
    if not isinstance(abstract, str):
        abstract = str(abstract)
    sections = _sections_from_full_text(full_text)
    return {
        "doc_id": row.get("id", ""),
        "title": row.get("title") or "",
        "abstract": abstract,
        "full_text": _full_text_string(sections, abstract),
        "sections": sections,
        "figures_and_tables": row.get("figures_and_tables") or {},
    }


def expand_qasper_rows(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one sample per (document, question) with gold lists."""
    doc = normalize_document(row)
    qas = row.get("qas") or {}
    if not isinstance(qas, dict):
        return
    questions = qas.get("question") or []
    qids = qas.get("question_id") or []
    answers_list = qas.get("answers") or []
    n = len(questions)
    for i in range(n):
        q = questions[i] if i < len(questions) else ""
        qid = qids[i] if i < len(qids) else f"{doc['doc_id']}_{i}"
        ablock = answers_list[i] if i < len(answers_list) else {}
        if not isinstance(ablock, dict):
            ablock = {}
        flat = _flatten_answers(ablock)
        answers = _gold_answer_strings(flat)
        evidence = _evidence_strings(flat)
        ev_mapped = _map_evidence_to_sections(evidence, doc["sections"])
        section_ids = sorted(
            {m["section_idx"] for m in ev_mapped if m["section_idx"] is not None and m["section_idx"] >= 0}
        )
        yield {
            "doc_id": doc["doc_id"],
            "question_id": qid,
            "title": doc["title"],
            "abstract": doc["abstract"],
            "full_text": doc["full_text"],
            "sections": doc["sections"],
            "question": q,
            "answers": answers,
            "evidence": evidence,
            "evidence_section_indices": section_ids,
            "evidence_mapped": ev_mapped,
        }


def _normalize_qasper_split(split: str) -> str:
    s = (split or "train").strip().lower()
    aliases = {"dev": "validation", "val": "validation"}
    s = aliases.get(s, s)
    if s not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported Qasper split {split!r}; use train, validation, or test.")
    return s


def load_qasper_split(split: str = "train") -> Dataset:
    # datasets>=4 no longer runs Hub dataset scripts (qasper.py). Load Parquet from the hub convert ref.
    s = _normalize_qasper_split(split)
    url = f"hf://datasets/allenai/qasper@refs/convert/parquet/qasper/{s}/*.parquet"
    return load_dataset("parquet", data_files=url, split="train")


def make_dev_subset(dataset: Dataset, max_samples: int, seed: int = 42) -> Dataset:
    if len(dataset) <= max_samples:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def iter_normalized_samples(dataset: Dataset) -> Iterator[dict[str, Any]]:
    for row in dataset:
        yield from expand_qasper_rows(row)
