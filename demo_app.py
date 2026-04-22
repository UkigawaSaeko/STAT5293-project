from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from data.qasper_loader import expand_qasper_rows, load_qasper_split, make_dev_subset
from parser.doc_parser import parse_sample_document
from parser.toc_builder import TOCNode


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_config() -> dict[str, Any]:
    cfg_path = _repo_root() / "experiments" / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _load_docs(split: str, max_docs: int, seed: int) -> list[dict[str, Any]]:
    ds = load_qasper_split(split=split)
    ds = make_dev_subset(ds, max_samples=max_docs, seed=seed)
    docs: list[dict[str, Any]] = []
    for row in ds:
        q_samples = list(expand_qasper_rows(row))
        if not q_samples:
            continue
        base = q_samples[0]
        docs.append(
            {
                "doc_id": base["doc_id"],
                "title": base.get("title") or base["doc_id"],
                "abstract": base.get("abstract") or "",
                "full_text": base.get("full_text") or "",
                "sections": base.get("sections") or [],
                "qas": q_samples,
            }
        )
    return docs


def _render_toc_tree(node: TOCNode, level: int = 0) -> str:
    if node.title == "ROOT":
        lines: list[str] = []
        for c in node.children:
            lines.append(_render_toc_tree(c, level=0))
        return "\n".join(x for x in lines if x)
    indent = "  " * level
    text = f"{indent}- {node.title} ({node.node_id})"
    for c in node.children:
        child = _render_toc_tree(c, level + 1)
        if child:
            text += f"\n{child}"
    return text


def _prepare_sample(doc: dict[str, Any], question_id: str) -> dict[str, Any]:
    for s in doc["qas"]:
        if s["question_id"] == question_id:
            return s
    return doc["qas"][0]


@st.cache_data(show_spinner=False)
def _load_method_predictions(method: str) -> dict[str, dict[str, str]]:
    p = _repo_root() / "outputs" / "predictions" / f"{method}_predictions.csv"
    if not p.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("question_id", "")).strip()
            if qid:
                out[qid] = row
    return out


def _to_f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _build_replay_results(
    sample: dict[str, Any],
    no_row: dict[str, str],
    vec_row: dict[str, str],
    toc_row: dict[str, str],
    parsed_doc: Any,
) -> dict[str, Any]:
    method_rows: dict[str, dict[str, Any]] = {}
    source_title = str(sample.get("title", "")).strip()
    row_by_method = {"no_rag": no_row, "vector_rag": vec_row, "toc_rag": toc_row}
    for method in ("no_rag", "vector_rag", "toc_rag"):
        row = row_by_method[method]
        source_doc_id = str(row.get("doc_id", "")).strip()
        source_article = (
            f"{source_title} ({source_doc_id})"
            if source_title and source_doc_id
            else (source_title or source_doc_id or "(unknown)")
        )
        method_rows[method] = {
            "method": method,
            "answer": str(row.get("pred", "")).strip() or "(empty prediction)",
            "source_title": source_title,
            "source_doc_id": source_doc_id,
            "source_article": source_article,
            "metrics": {
                "f1": _to_f(row, "f1"),
                "em": _to_f(row, "em"),
                "evidence_hit_rate": _to_f(row, "evidence_hit_rate"),
                "citation_hit_rate": _to_f(row, "citation_hit_rate"),
                "citation_precision": _to_f(row, "citation_precision"),
                "heuristic_hallucination": _to_f(row, "heuristic_hallucination"),
                "abstain": _to_f(row, "abstain"),
            },
            "efficiency": {
                "prompt_tokens": int(_to_f(row, "prompt_tokens")),
                "completion_tokens": int(_to_f(row, "completion_tokens")),
                "total_tokens": int(_to_f(row, "prompt_tokens") + _to_f(row, "completion_tokens")),
            },
        }

    return {
        "parsed_doc": parsed_doc,
        "question": sample.get("question", ""),
        "question_id": sample.get("question_id", ""),
        "source": "historical_replay",
        "methods": method_rows,
    }


def _method_panel(title: str, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    eff = payload["efficiency"]
    st.subheader(title)
    st.write(payload.get("answer", "(empty answer)"))
    st.markdown(
        f"**Article:** {payload.get('source_title') or '(unknown title)'}  \n"
        f"`doc_id`: {payload.get('source_doc_id') or '(unknown)'}"
    )

    # Use a 2x2 metric layout so numeric values do not truncate.
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    r1c1.metric("F1", f"{metrics['f1']:.3f}")
    r1c2.metric("Cit-Prec", f"{metrics['citation_precision']:.3f}")
    r2c1.metric("EM", f"{metrics['em']:.3f}")
    r2c2.metric("Tokens", f"{eff['total_tokens']}")
    st.caption("F1 = answer correctness, Cit-Prec = citation precision, EM = exact match.")


def main() -> None:
    st.set_page_config(page_title="Long Doc QA Live Comparator", layout="wide")
    st.title("Interactive Long-Document QA Benchmark")
    st.caption(
        "Real-time comparison across No RAG, Vector RAG, and TOC-Based RAG on the same document and question."
    )

    cfg_base = _default_config()
    with st.sidebar:
        st.header("Demo Controls")
        max_docs = st.slider("Load # documents", min_value=1, max_value=30, value=5)
        seed = st.number_input("Sampling seed", min_value=1, max_value=9999, value=42)
        vector_top_k = st.slider("Vector top-k", min_value=3, max_value=8, value=5)
        toc_depth = st.slider("TOC max depth", min_value=2, max_value=4, value=3)
        toc_selection = st.selectbox("TOC selection", ["bm25", "hash_embed"], index=0)
        run_btn = st.button("Run live comparison", type="primary")

    default_split = str(cfg_base.get("dataset_split", "train"))
    docs = _load_docs(split=default_split, max_docs=max_docs, seed=int(seed))
    if not docs:
        st.error("No documents loaded from dataset.")
        return

    chosen_doc = st.selectbox(
        "Choose one long document",
        docs,
        format_func=lambda d: f"{d['title'][:100]} ({len(d['sections'])} sections)",
    )
    questions = chosen_doc["qas"]
    chosen_qid = st.selectbox(
        "Choose baseline question (you can edit below)",
        [q["question_id"] for q in questions],
    )
    selected_sample = _prepare_sample(chosen_doc, chosen_qid)
    user_question = st.text_area(
        "Question input",
        value=selected_sample["question"],
        height=90,
        help="Use cross-section questions for stronger contrast.",
    )

    sample = dict(selected_sample)
    sample["question"] = user_question.strip()
    parsed_doc = parse_sample_document(
        sample,
        chunk_max_words=int(cfg_base.get("chunk_max_words", 400)),
        chunk_overlap_words=int(cfg_base.get("chunk_overlap_words", 50)),
    )
    preds_no = _load_method_predictions("no_rag")
    preds_vec = _load_method_predictions("vector_rag")
    preds_toc = _load_method_predictions("toc_rag")

    left, right = st.columns([1, 2])

    with left:
        st.markdown("### Document / TOC")
        st.write(f"**Title:** {chosen_doc['title']}")
        st.write(f"**Doc ID:** `{chosen_doc['doc_id']}`")
        st.write(f"**Sections:** {len(chosen_doc['sections'])}")
        with st.expander("Abstract", expanded=False):
            st.write(chosen_doc["abstract"] or "(no abstract)")
        with st.expander("Full text (preview)", expanded=False):
            st.write((chosen_doc["full_text"] or "")[:4000] or "(empty)")

    if run_btn:
        qid = str(sample.get("question_id", "")).strip()
        if qid not in preds_no or qid not in preds_vec or qid not in preds_toc:
            with right:
                st.error(
                    "This question_id is missing from one or more files in outputs/predictions/"
                    "(no_rag/vector_rag/toc_rag predictions CSV)."
                )
            return

        with st.spinner("Loading historical benchmark results..."):
            results = _build_replay_results(sample, preds_no[qid], preds_vec[qid], preds_toc[qid], parsed_doc)

        with left:
            st.markdown("### TOC Tree")
            st.code(_render_toc_tree(results["parsed_doc"].toc_root), language="text")
            st.markdown("### Gold answers / evidence")
            st.write(sample.get("answers") or ["(none)"])
            with st.expander("Gold evidence snippets", expanded=False):
                for idx, ev in enumerate(sample.get("evidence") or [], start=1):
                    st.write(f"{idx}. {ev}")

        with right:
            st.info(
                "Replay mode: values come from historical run files "
                "`outputs/predictions/no_rag_predictions.csv`, "
                "`outputs/predictions/vector_rag_predictions.csv`, "
                "`outputs/predictions/toc_rag_predictions.csv`."
            )
            st.markdown("### Results: same question, three systems")
            r1, r2, r3 = st.columns(3)
            with r1:
                _method_panel("No RAG", results["methods"]["no_rag"])
            with r2:
                _method_panel("Vector RAG", results["methods"]["vector_rag"])
            with r3:
                _method_panel("TOC-Based RAG", results["methods"]["toc_rag"])

            st.markdown("### Source Article / Efficiency / Quality")
            for key, label in (
                ("no_rag", "No RAG"),
                ("vector_rag", "Vector RAG"),
                ("toc_rag", "TOC-Based RAG"),
            ):
                payload = results["methods"][key]
                metrics = payload["metrics"]
                eff = payload["efficiency"]
                with st.expander(label, expanded=(key == "toc_rag")):
                    st.write(f"**Source article:** {payload.get('source_title') or '(unknown title)'}")
                    st.write(f"**Source doc_id:** {payload.get('source_doc_id') or '(unknown)'}")
                    st.write(
                        f"**Efficiency:** prompt_tokens={eff['prompt_tokens']}, "
                        f"completion_tokens={eff['completion_tokens']}, "
                        f"total_tokens={eff['total_tokens']}"
                    )
                    st.write(
                        f"**Grounding metrics:** evidence_hit={metrics['evidence_hit_rate']:.3f}, "
                        f"citation_hit={metrics['citation_hit_rate']:.3f}, "
                        f"hallucination={metrics['heuristic_hallucination']:.1f}"
                    )
    else:
        with right:
            st.info("Select a paper and question, then click `Run live comparison`.")


if __name__ == "__main__":
    main()
