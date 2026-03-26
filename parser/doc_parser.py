from __future__ import annotations

from dataclasses import dataclass, field

from parser.chunker import chunk_words
from parser.toc_builder import TOCNode, build_toc_from_sections, flat_sections_from_root


@dataclass
class ParsedDocument:
    doc_id: str
    full_text: str
    sections: list[dict]
    toc_root: TOCNode
    flat_sections: list[dict]
    vector_chunks: list[dict] = field(default_factory=list)

    def chunk_payloads(self) -> list[str]:
        return [c["text"] for c in self.vector_chunks]


def parse_sample_document(
    sample: dict,
    chunk_max_words: int = 400,
    chunk_overlap_words: int = 50,
) -> ParsedDocument:
    doc_id = sample.get("doc_id", "")
    full_text = sample.get("full_text") or ""
    sections = list(sample.get("sections") or [])
    toc_root = build_toc_from_sections(doc_id, sections)
    flat_sections = flat_sections_from_root(toc_root)
    chunks: list[dict] = []
    for i, piece in enumerate(chunk_words(full_text, chunk_max_words, chunk_overlap_words)):
        chunks.append({"chunk_id": f"chunk_{i}", "text": piece})
    return ParsedDocument(
        doc_id=doc_id,
        full_text=full_text,
        sections=sections,
        toc_root=toc_root,
        flat_sections=flat_sections,
        vector_chunks=chunks,
    )
