"""Deterministic fake backends so the whole pipeline runs without a model.

These are NOT trying to be smart. They exist so we can build and test the
plumbing and the trust-measurement on a laptop with no GPU. On the Mac we
replace them with real MLX models behind the same interfaces.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from ..interfaces import Embedder, Generator

_WORD = re.compile(r"[a-z0-9]+")
# Tiny stop-word list so trivial words don't create false "matches".
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "and",
    "or", "for", "what", "which", "who", "how", "does", "do", "did", "this",
    "that", "it", "as", "by", "with", "at", "from", "be", "can",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3 and w not in _STOP}


class MockEmbedder(Embedder):
    """A bag-of-words 'embedding': each content word bumps one slot of a vector.

    It's purely lexical (no real meaning), but it's enough that passages sharing
    words land near each other under cosine similarity — which lets us test
    retrieval end-to-end.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _content_words(text):
                slot = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
                vecs[i, slot] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm  # unit length, so dot product == cosine similarity
        return vecs


class MockGenerator(Generator):
    """Pretends to be a *well-behaved grounded model*.

    It reads the passages and question out of the prompt, then:
      - answers (citing the best-matching passage) if the question shares a
        content word with some passage, otherwise
      - replies exactly 'NOT IN SOURCES'.

    This lets us exercise both the answer path and the abstention path — and so
    measure a (trivially zero) bluff rate — without any real model.
    """

    _PASSAGE = re.compile(r"PASSAGE \[(\d+)\]:\n(.*?)(?=\n\nPASSAGE \[|\n\nQUESTION:)", re.DOTALL)
    _QUESTION = re.compile(r"QUESTION:\s*(.*?)\nANSWER:", re.DOTALL)

    def generate(self, prompt: str) -> str:
        passages = [(int(n), txt) for n, txt in self._PASSAGE.findall(prompt)]
        qmatch = self._QUESTION.search(prompt)
        question = qmatch.group(1) if qmatch else ""
        q_words = _content_words(question)

        best_idx, best_overlap = None, 0
        for idx, txt in passages:
            overlap = len(q_words & _content_words(txt))
            if overlap > best_overlap:
                best_idx, best_overlap = idx, overlap

        if best_idx is None:
            return "NOT IN SOURCES"
        return f"Based on the source, see passage [{best_idx}]."
