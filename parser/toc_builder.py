from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TOCNode:
    title: str
    level: int
    content: str
    parent: TOCNode | None = None
    children: list[TOCNode] = field(default_factory=list)
    node_id: str | None = None
    path: str = ""

    def assign_ids(self, prefix: str = "sec") -> None:
        counter = 0
        stack: list[TOCNode] = [self]
        while stack:
            node = stack.pop(0)
            node.node_id = f"{prefix}_{counter}"
            counter += 1
            stack.extend(node.children)


def build_toc_from_sections(doc_id: str, sections: list[dict]) -> TOCNode:
    root = TOCNode(title="ROOT", level=0, content="", parent=None, path="")
    path_nodes: dict[tuple[str, ...], TOCNode] = {}

    for s in sections:
        header = (s.get("header") or "").strip()
        text = (s.get("text") or "").strip()
        parts = [p.strip() for p in header.split(":::") if p.strip()] if header else []
        if not parts:
            parts = [f"section_{s.get('section_idx', len(path_nodes))}"]
        parent = root
        path_so_far: list[str] = []
        for depth, part in enumerate(parts):
            path_so_far.append(part)
            key = tuple(path_so_far)
            if key not in path_nodes:
                node = TOCNode(
                    title=part,
                    level=depth + 1,
                    content="",
                    parent=parent,
                    path=" > ".join(path_so_far),
                )
                parent.children.append(node)
                path_nodes[key] = node
            parent = path_nodes[key]
        leaf = path_nodes[tuple(parts)]
        leaf.content = (leaf.content + "\n\n" + text if leaf.content else text).strip()

    for node in path_nodes.values():
        node.path = " > ".join(_path_parts(node))

    root.assign_ids(prefix=f"{doc_id}_sec" if doc_id else "sec")
    return root


def _path_parts(node: TOCNode) -> list[str]:
    parts: list[str] = []
    cur: TOCNode | None = node
    while cur and cur.title != "ROOT":
        parts.append(cur.title)
        cur = cur.parent
    return list(reversed(parts))


def flat_sections_from_root(root: TOCNode) -> list[dict]:
    out: list[dict] = []
    stack = list(root.children)
    while stack:
        n = stack.pop(0)
        stack.extend(n.children)
        if n.title == "ROOT":
            continue
        text = n.content.strip()
        if not text:
            continue
        out.append(
            {
                "node_id": n.node_id,
                "path": n.path,
                "title": n.title,
                "text": text,
            }
        )
    return out
