def no_rag_prompt(question: str) -> str:
    return (
        "Answer the following question based on your knowledge. "
        "If you cannot answer reliably, say you do not know.\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )


def rag_context_prompt(question: str, context: str, cite_instruction: bool = True) -> str:
    cite = ""
    if cite_instruction:
        cite = (
            "\nAfter the answer, on a new line output a JSON line exactly like:\n"
            'EVIDENCE_IDS: ["chunk_0","sec_1"]\n'
            "using only ids that appear in the context headers. If none apply, use [].\n"
        )
    return (
        "Use only the provided context to answer. If the context is insufficient, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:{cite}"
    )


def answer_with_citations_prompt(question: str, context: str, path_hint: str = "") -> str:
    path_line = f"Relevant section path: {path_hint}\n\n" if path_hint else ""
    return rag_context_prompt(question, path_line + context, cite_instruction=True)


def toc_nav_prompt(children_titles: list[str], question: str) -> str:
    lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(children_titles))
    return (
        "You are navigating a paper outline. Pick the single best section index (1-based).\n"
        f"Current sections:\n{lines}\n\nQuestion: {question}\n\n"
        "Reply with one integer only."
    )
