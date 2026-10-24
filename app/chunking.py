from typing import List, Tuple


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split text into ~`size`-char chunks that share `overlap` chars with the previous one.

    Cuts are snapped back to a whitespace boundary so words are not split.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        nxt = max(end - overlap, start + 1)
        # begin the next chunk at a word boundary, not mid-word
        if nxt > 0 and text[nxt - 1] != " ":
            space = text.find(" ", nxt, end)
            nxt = space + 1 if space != -1 else end
        start = nxt
    return chunks


def chunk_pages(pages: List[str], size: int, overlap: int) -> List[Tuple[int, str]]:
    """Chunk each page separately so every chunk maps to one page number (1-based)."""
    out = []
    for page_no, page_text in enumerate(pages, start=1):
        for piece in chunk_text(page_text, size, overlap):
            out.append((page_no, piece))
    return out
