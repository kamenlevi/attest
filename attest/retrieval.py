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


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], c: int = 60
) -> list[int]:
    """Fuse several ranked lists of chunk indices into one (RRF).

    Each list is best-first. A chunk's fused score is the sum of 1/(c + rank)
    across the lists it appears in (rank starts at 1). Items ranked highly by
    *either* retriever bubble up — so we get the keyword hits AND the meaning
    hits. `c` damps the influence of low ranks; 60 is the common default.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank)
    return sorted(scores, key=lambda key: scores[key], reverse=True)


class HybridRetriever:
    """Combines several retrievers (e.g. lexical + semantic) via RRF.

    Duck-typed to match Retriever: it has build() and search(), so the eval and
    CLI use it interchangeably.
    """

    def __init__(self, retrievers: list[Retriever], pool: int = 25) -> None:
        self._retrievers = retrievers
        self._pool = pool  # candidates to pull from each retriever before fusing
        self._by_index: dict[int, object] = {}

    def build(self, chunks) -> None:
        for r in self._retrievers:
            r.build(chunks)
        self._by_index = {c.index: c for c in chunks}

    def search(self, query: str, k: int = 4) -> list[Retrieved]:
        ranked_lists = [
            [r.chunk.index for r in retr.search(query, k=self._pool)]
            for retr in self._retrievers
        ]
        fused = reciprocal_rank_fusion(ranked_lists)[:k]
        # score here is the fused rank position turned into a descending value.
        return [
            Retrieved(chunk=self._by_index[idx], score=float(len(fused) - pos))
            for pos, idx in enumerate(fused)
        ]
