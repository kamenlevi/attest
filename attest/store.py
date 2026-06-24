"""Persistent index: embed every chunk ONCE, save to disk, query fast.

The bottleneck before this module was that we re-embedded the whole corpus on
every question. That doesn't scale to a student's library of hundreds of
700-page books. Here we embed once (at "upload"/index time), save the vectors,
and at query time embed only the question and search the saved vectors.

For now search is brute-force NumPy (fine up to ~100k chunks). The next step is
an ANN index (FAISS/hnswlib) to keep it millisecond-fast at millions of chunks —
this class is the seam where that swap happens, with no change to callers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .interfaces import Embedder
from .retrieval import Retrieved


@dataclass(frozen=True)
class StoredChunk:
    uid: int          # global id across all indexed documents (used as citation)
    source: str       # which file it came from
    local_index: int  # its position within that file
    text: str


class IndexedStore:
    """A saved, queryable index. Duck-typed like Retriever: it has search()."""

    def __init__(self, stored: list[StoredChunk], matrix: np.ndarray, embedder: Embedder) -> None:
        self._stored = stored
        self._matrix = matrix          # (N, dim) unit vectors
        self._embedder = embedder      # used to embed queries only

    @classmethod
    def build(cls, docs: list[tuple[str, list[Chunk]]], embedder: Embedder) -> "IndexedStore":
        """Embed all chunks from several documents into one index (done once)."""
        stored: list[StoredChunk] = []
        uid = 0
        for source, chunks in docs:
            for c in chunks:
                stored.append(StoredChunk(uid, source, c.index, c.text))
                uid += 1
        if stored:
            matrix = embedder.embed([s.text for s in stored]).astype(np.float32)
        else:
            matrix = np.zeros((0, 1), dtype=np.float32)
        return cls(stored, matrix, embedder)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "vectors.npy", self._matrix)
        with open(p / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for s in self._stored:
                fh.write(json.dumps(s.__dict__) + "\n")

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> "IndexedStore":
        p = Path(path)
        matrix = np.load(p / "vectors.npy")
        stored: list[StoredChunk] = []
        with open(p / "chunks.jsonl", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                stored.append(StoredChunk(d["uid"], d["source"], d["local_index"], d["text"]))
        return cls(stored, matrix, embedder)

    def __len__(self) -> int:
        return len(self._stored)

    def search(self, query: str, k: int = 4) -> list[Retrieved]:
        if self._matrix.shape[0] == 0:
            return []
        q = self._embedder.embed([query])[0]
        scores = self._matrix @ q
        order = np.argsort(scores)[::-1][:k]
        return [
            Retrieved(
                chunk=Chunk(index=self._stored[i].uid, text=self._stored[i].text,
                            source=self._stored[i].source),
                score=float(scores[i]),
            )
            for i in order
        ]
