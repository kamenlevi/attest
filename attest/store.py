"""Persistent index: embed every chunk ONCE, save to disk, query fast.

The bottleneck before this module was that we re-embedded the whole corpus on
every question. That doesn't scale to a student's library of hundreds of
700-page books. Here we embed once (at "upload"/index time), save the vectors,
and at query time embed only the question and search the saved vectors.

Two properties make it usable at scale:

  * INCREMENTAL & IDEMPOTENT. Each file's content is hashed into a manifest.
    Re-indexing a file that's already present (unchanged) is skipped entirely —
    so adding one new book to a library of 100 costs one book's work, and
    re-running the same command is instant. Editing a file re-indexes just that
    file (its old chunks are dropped, new ones appended).

  * BATCHED EMBEDDING. A 700-page book is embedded in bounded batches rather than
    one giant call, so memory stays flat and progress is visible.

Search is brute-force NumPy (fine up to ~100k chunks). The next step is an ANN
index (FAISS/hnswlib) to stay millisecond-fast at millions of chunks — this class
is the seam where that swaps in, with no change to callers.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .interfaces import Embedder
from .retrieval import Retrieved

_BATCH = 256  # chunks embedded per call — bounds memory on big books


def file_fingerprint(path: str | Path) -> str:
    """A cheap content fingerprint of a file, from its raw bytes.

    Used to decide "already indexed, unchanged -> skip" WITHOUT extracting or
    chunking the file (which for a 600-page PDF costs tens of seconds). Hashing
    the bytes of even a 50 MB book is well under a second.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _hash_chunks(chunks: list[Chunk]) -> str:
    """A content fingerprint for a document, from its chunk texts.

    Same file -> same extraction -> same chunks -> same hash, so we can tell when
    a previously-indexed file is unchanged (skip) vs edited (re-index).
    """
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _embed_batched(embedder: Embedder, texts: list[str], batch: int = _BATCH) -> np.ndarray:
    """Embed many texts in bounded batches; show progress for long jobs."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    out: list[np.ndarray] = []
    for start in range(0, len(texts), batch):
        out.append(embedder.embed(texts[start : start + batch]).astype(np.float32))
        if len(texts) > batch:
            done = min(start + batch, len(texts))
            print(f"    embedded {done}/{len(texts)} chunks", file=sys.stderr)
    return np.vstack(out)


@dataclass(frozen=True)
class StoredChunk:
    uid: int          # global id across all indexed documents (used as citation)
    source: str       # which file it came from
    local_index: int  # its position within that file
    text: str


class IndexedStore:
    """A saved, queryable index. Duck-typed like Retriever: it has search()."""

    def __init__(
        self,
        stored: list[StoredChunk],
        matrix: np.ndarray,
        embedder: Embedder,
        manifest: dict[str, str] | None = None,
    ) -> None:
        self._stored = stored
        self._matrix = matrix          # (N, dim) unit vectors
        self._embedder = embedder      # used to embed queries only
        self._manifest = manifest or {}  # source -> content hash

    @classmethod
    def build(cls, docs: list[tuple[str, list[Chunk]]], embedder: Embedder) -> "IndexedStore":
        """Embed all chunks from several documents into one fresh index."""
        store = cls([], np.zeros((0, 1), dtype=np.float32), embedder, {})
        store.add(docs, embedder)
        return store

    def fingerprint_of(self, source: str) -> str | None:
        """The fingerprint recorded for a source, or None if not indexed.

        Lets a caller decide to skip a file *before* extracting/chunking it.
        """
        return self._manifest.get(source)

    def add(
        self,
        docs: list[tuple[str, list[Chunk]]],
        embedder: Embedder,
        fingerprints: dict[str, str] | None = None,
    ) -> dict[str, int]:
        """Add documents incrementally. Returns {"added", "skipped", "updated"}.

        A file already in the manifest with the same fingerprint is skipped. A
        file with a changed fingerprint has its old chunks dropped and is
        re-embedded. `fingerprints` lets the caller supply a cheap file-level
        fingerprint (see `file_fingerprint`); without it we fall back to hashing
        the chunk texts.
        """
        fingerprints = fingerprints or {}
        stats = {"added": 0, "skipped": 0, "updated": 0}
        next_uid = (max((s.uid for s in self._stored), default=-1)) + 1
        new_chunks: list[StoredChunk] = []
        new_texts: list[str] = []
        for source, chunks in docs:
            digest = fingerprints.get(source) or _hash_chunks(chunks)
            if self._manifest.get(source) == digest:
                stats["skipped"] += 1
                continue
            if source in self._manifest:
                self._drop_source(source)  # edited file: remove its stale chunks
                stats["updated"] += 1
            else:
                stats["added"] += 1
            for c in chunks:
                new_chunks.append(StoredChunk(next_uid, source, c.index, c.text))
                new_texts.append(c.text)
                next_uid += 1
            self._manifest[source] = digest

        if new_texts:
            new_matrix = _embed_batched(embedder, new_texts)
            if self._matrix.shape[0] == 0:
                self._matrix = new_matrix
            else:
                self._matrix = np.vstack([self._matrix, new_matrix])
            self._stored.extend(new_chunks)
        return stats

    def _drop_source(self, source: str) -> None:
        """Remove all chunks (and their vectors) belonging to one source."""
        keep = [i for i, s in enumerate(self._stored) if s.source != source]
        self._stored = [self._stored[i] for i in keep]
        self._matrix = self._matrix[keep] if self._matrix.shape[0] else self._matrix
        self._manifest.pop(source, None)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "vectors.npy", self._matrix)
        with open(p / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for s in self._stored:
                fh.write(json.dumps(s.__dict__) + "\n")
        with open(p / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(self._manifest, fh, indent=2)

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> "IndexedStore":
        p = Path(path)
        matrix = np.load(p / "vectors.npy")
        stored: list[StoredChunk] = []
        with open(p / "chunks.jsonl", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                stored.append(StoredChunk(d["uid"], d["source"], d["local_index"], d["text"]))
        manifest_path = p / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        return cls(stored, matrix, embedder, manifest)

    def sources(self) -> list[str]:
        """The files currently in the index."""
        return list(self._manifest)

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
