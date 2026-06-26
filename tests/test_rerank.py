"""Reranking-retriever tests: a mock reranker proves the reorder logic, no model."""

from attest.chunking import Chunk
from attest.retrieval import RerankingRetriever, Retrieved


class _FakeBase:
    """Returns a fixed candidate list (best-first by the bi-encoder)."""

    def __init__(self, ids):
        self._ids = ids

    def search(self, query, k=4):
        return [Retrieved(chunk=Chunk(index=i, text=f"passage {i}"), score=1.0 / (r + 1))
                for r, i in enumerate(self._ids[:k])]


class _KeywordReranker:
    """Scores a passage by whether it contains the target id we 'want' on top."""

    def __init__(self, want):
        self._want = want

    def score(self, query, passages):
        return [10.0 if f"passage {self._want}" == p else 0.0 for p in passages]


def test_rerank_promotes_the_best_candidate_above_bi_encoder_order():
    # Bi-encoder ranks 99 last; the reranker knows 99 is the real answer.
    base = _FakeBase([1, 2, 3, 4, 99])
    retr = RerankingRetriever(base, _KeywordReranker(want=99), pool=10)
    out = [r.chunk.index for r in retr.search("q", k=3)]
    assert out[0] == 99  # promoted from last to first


def test_rerank_pulls_a_wide_pool_then_trims_to_k():
    base = _FakeBase(list(range(50)))
    seen = {}

    class _SpyBase(_FakeBase):
        def search(self, query, k=4):
            seen["k"] = k
            return super().search(query, k=k)

    retr = RerankingRetriever(_SpyBase(list(range(50))), _KeywordReranker(want=7), pool=40)
    out = retr.search("q", k=5)
    assert seen["k"] == 40   # asked the base for the wide pool, not just k
    assert len(out) == 5     # trimmed to k
    assert out[0].chunk.index == 7


def test_rerank_handles_empty_base():
    assert RerankingRetriever(_FakeBase([]), _KeywordReranker(0)).search("q", k=3) == []
