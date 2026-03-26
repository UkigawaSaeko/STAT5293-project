def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Character-level sliding windows (as in spec)."""
    if chunk_size <= 0:
        return [text] if text else []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += chunk_size - overlap
    return chunks


def chunk_words(text: str, max_tokens: int = 400, overlap_words: int = 50) -> list[str]:
    """Word-based chunks; ~token budget without tiktoken."""
    words = text.split()
    if not words:
        return []
    step = max(1, max_tokens - overlap_words)
    out: list[str] = []
    i = 0
    while i < len(words):
        chunk = words[i : i + max_tokens]
        out.append(" ".join(chunk))
        i += step
    return out
