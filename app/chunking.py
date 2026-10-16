def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks, i = [], 0
    while True:
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            return chunks
        i += step