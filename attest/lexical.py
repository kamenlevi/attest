"""BM25 keyword retrieval — the half of recall that embeddings miss.

Semantic (embedding) search matches *meaning*, which is exactly why it's weak on
*exact* tokens: proper nouns ("Walter of Merton"), labels ("Box 2.1"), section
numbers, rare technical words. Those are precisely the things a document-specific
question hinges on, and we watched semantic-only retrieval abstain on them
(Exp 10: B3 "section 4.5", B7 "Box 2.1", B11 "merely a tool").

BM25 is the standard keyword retriever: it scores a chunk by how many query terms
it contains, weighting rare terms higher (idf) and saturating term frequency. It's
the complement to embeddings — fuse the two (RRF) and you catch both the paraphrase
and the exact string.

Pure standard-library, and fast to build (tokenise + count over ~1k chunks is
instant), so we build it from the index's chunks at load time rather than
persisting a second index.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .retrieval import Retrieved

# Keep dotted labels ("2.1", "2.87", "9.2") as single tokens — they're how this
# kind of book refers to equations, sections and boxes, and splitting them into
# "2"/"1" destroys the one distinctive term in a question like "what is Box 2.1?".
_TOKEN = re.compile(r"\d+(?:\.\d+)+|[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Keyword retriever over a fixed set of chunks. Duck-typed like Retriever."""

    def __init__(self, chunks, k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks = list(chunks)  # each has .index, .text, (optional) .source
        self._k1 = k1
        self._b = b
        docs = [_tokenize(c.text) for c in self._chunks]
        self._tf = [Counter(d) for d in docs]
        self._len = [len(d) for d in docs]
        n = len(docs)
        self._avgdl = (sum(self._len) / n) if n else 0.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        # idf with the BM25 +0.5 smoothing; max(…, tiny) keeps very common terms ≥0.
        self._idf = {
            t: max(1e-6, math.log(1 + (n - f + 0.5) / (f + 0.5)))
            for t, f in df.items()
        }

    def search(self, query: str, k: int = 4) -> list[Retrieved]:
        if not self._chunks:
            return []
        terms = _tokenize(query)
        scores = [0.0] * len(self._chunks)
        for i, tf in enumerate(self._tf):
            dl = self._len[i]
            denom_norm = self._k1 * (1 - self._b + self._b * dl / (self._avgdl or 1))
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    s += self._idf.get(t, 0.0) * (f * (self._k1 + 1)) / (f + denom_norm)
            scores[i] = s
        order = sorted(range(len(self._chunks)), key=lambda i: scores[i], reverse=True)[:k]
        # Return the chunk objects as given — rebuilding them here would silently
        # drop metadata like the page number that citations depend on.
        return [
            Retrieved(chunk=self._chunks[i], score=scores[i])
            for i in order
            if scores[i] > 0  # don't return chunks with no query-term overlap
        ]
