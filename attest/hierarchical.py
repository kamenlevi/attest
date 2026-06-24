"""Hierarchical retrieval: rank sections first, then search chunks within them.

The idea (your design): instead of scanning every chunk of every book, give each
section a compact "where am I about" vector, rank sections against the question,
then do the expensive exact chunk search only inside the most promising sections.
Across many books this is how you "look fast at scale".

The compact section vector here is the **centroid** (normalised mean) of its
chunk vectors. That's the compression — cheap and LLM-free, so it can't
hallucinate. Its risk is *over*-compression: a section about many topics gets a
blurry centroid and a single sharply-relevant chunk inside an off-topic section
can be missed. We defend against that by ranking GENEROUSLY (top_sections high)
and we MEASURE recall vs. flat search before trusting it (see scripts/compare).

Save/load is intentionally deferred until the approach proves itself; this v1 is
built in memory for the more-good-than-harm experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk, chunk_text
from .interfaces import Embedder
from .retrieval import Retrieved
from .structure import split_into_sections


@dataclass(frozen=True)
class Section:
    id: int
    title: str
    source: str


class HierarchicalStore:
    def __init__(self, stored, chunk_matrix, sections, section_matrix, chunk_section, embedder):
        self._stored = stored                  # list of dicts: uid, source, section_id, title, text
        self._cm = chunk_matrix                 # (N, dim) chunk vectors
        self._sections = sections               # list[Section]
        self._sm = section_matrix               # (S, dim) section centroids
        self._chunk_section = chunk_section     # (N,) section id per chunk
        self._embedder = embedder

    @classmethod
    def build(cls, docs: list[tuple[str, str]], embedder: Embedder, **split_kw) -> "HierarchicalStore":
        stored: list[dict] = []
        chunk_section: list[int] = []
        sections: list[Section] = []
        sec_id = uid = 0
        for source, text in docs:
            for title, sec_text in split_into_sections(text, **split_kw):
                chunks = chunk_text(sec_text)
                if not chunks:
                    continue
                sections.append(Section(sec_id, title, source))
                for c in chunks:
                    stored.append({"uid": uid, "source": source, "section_id": sec_id,
                                   "title": title, "text": c.text})
                    chunk_section.append(sec_id)
                    uid += 1
                sec_id += 1

        if stored:
            cm = embedder.embed([s["text"] for s in stored]).astype(np.float32)
        else:
            cm = np.zeros((0, 1), dtype=np.float32)
        cs = np.array(chunk_section, dtype=np.int64)

        dim = cm.shape[1] if cm.shape[0] else 1
        sm = np.zeros((len(sections), dim), dtype=np.float32)
        for s in sections:
            v = cm[cs == s.id].mean(axis=0)
            n = float(np.linalg.norm(v))
            sm[s.id] = v / n if n > 0 else v
        return cls(stored, cm, sections, sm, cs, embedder)

    def ranked_sections(self, query: str, n: int = 10) -> list[tuple[str, float]]:
        """The section ranking — i.e. 'where the app decided to look'."""
        qv = self._embedder.embed([query])[0]
        scores = self._sm @ qv
        order = np.argsort(scores)[::-1][:n]
        return [(self._sections[int(i)].title, float(scores[int(i)])) for i in order]

    def search(self, query: str, k: int = 4, top_sections: int = 5) -> list[Retrieved]:
        if self._cm.shape[0] == 0:
            return []
        qv = self._embedder.embed([query])[0]
        sec_scores = self._sm @ qv
        top = set(int(i) for i in np.argsort(sec_scores)[::-1][:top_sections])
        cand = np.array([i for i in range(len(self._stored))
                         if int(self._chunk_section[i]) in top])
        cscores = self._cm[cand] @ qv
        chosen = cand[np.argsort(cscores)[::-1][:k]]
        out = []
        for i in chosen:
            s = self._stored[int(i)]
            out.append(Retrieved(
                chunk=Chunk(index=s["uid"], text=s["text"], source=f'{s["source"]}#{s["title"]}'),
                score=float(self._cm[int(i)] @ qv),
            ))
        return out
