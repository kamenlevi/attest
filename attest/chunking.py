"""Step 2 of the pipeline: split a long document into small passages ("chunks").

Why chunk at all? We retrieve *passages*, not whole books. A question gets
answered from the few most relevant passages, so we slice the text into pieces
small enough to be precise but big enough to keep their meaning.

`overlap` repeats a few words between neighbouring chunks so we don't slice a
sentence (or an idea) cleanly in half and lose it.

This module is pure Python — no model needed — so it runs and tests anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One passage of the source document.

    `index` is its position / citation id, so when the model says "[3]" we know
    exactly which passage it meant. `source` records which file it came from
    (used once a single index spans many documents). `page` is the 1-based page
    the passage starts on (None for plain text files) — it's what makes a
    citation checkable by a person: "qb.pdf, p. 112" instead of "chunk 73".
    """

    index: int
    text: str
    source: str = ""
    page: int | None = None


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Split `text` into overlapping word-windows.

    `chunk_size` and `overlap` are counted in **words**. They are knobs we will
    tune later and measure the effect of — bigger chunks = more context but more
    noise; smaller = more precise but can lose meaning.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(Chunk(index=index, text=" ".join(window)))
        index += 1
        if start + chunk_size >= len(words):
            break  # this window reached the end; stop
        start += step
    return chunks


def chunk_pages(
    pages: list[tuple[int | None, str]], chunk_size: int = 200, overlap: int = 40
) -> list[Chunk]:
    """Like `chunk_text`, but for page-tagged text: [(page_number, text), ...].

    The windowing is identical (the word stream flows across page boundaries, so
    nothing is lost at a page break). Each chunk records the page its first word
    sits on — that's the page a person opens to verify the citation.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words: list[str] = []
    word_pages: list[int | None] = []
    for page, text in pages:
        for w in text.split():
            words.append(w)
            word_pages.append(page)
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(Chunk(index=index, text=" ".join(window), page=word_pages[start]))
        index += 1
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks
