"""Steps 3-5: embed the chunks, then find the ones most relevant to a question.

"Most relevant" = highest cosine similarity between the question's vector and
each chunk's vector. Because our embedder returns unit-length vectors, cosine
similarity is just a dot product.

For Phase 1 the corpus is small (one chapter), so a plain NumPy search is plenty
— no heavyweight vector database needed. Simplicity first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .interfaces import Embedder


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> None:
        """Embed every chunk once and keep the matrix in memory."""
        self._chunks = list(chunks)
        if not chunks:
            self._matrix = None
            return
        self._matrix = self._embedder.embed([c.text for c in chunks])

    def search(self, query: str, k: int = 4) -> list[Retrieved]:
        """Return the top-`k` chunks most similar to `query`, best first."""
        if self._matrix is None:
            return []
        q = self._embedder.embed([query])[0]
        scores = self._matrix @ q  # one dot product per chunk
        order = np.argsort(scores)[::-1][:k]
        return [Retrieved(chunk=self._chunks[i], score=float(scores[i])) for i in order]
