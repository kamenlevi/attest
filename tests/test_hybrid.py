from attest.chunking import Chunk
from attest.interfaces import Embedder
from attest.retrieval import HybridRetriever, Retriever, reciprocal_rank_fusion

import numpy as np


def test_rrf_rewards_agreement_across_lists():
    # chunk 2 is ranked highly by both lists -> should win overall.
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 3, 1]])
    assert fused[0] == 2


def test_rrf_surfaces_each_retrievers_top_hit_when_no_overlap():
    # No shared chunks: each list's #1 should lead. (When lists DO overlap, RRF
    # rewards agreement instead — that's by design, and the reason naive hybrid
    # can bury a lone strong hit. See docs/experiments.md.)
    fused = reciprocal_rank_fusion([[10, 11], [20, 21]])
    assert set(fused[:2]) == {10, 20}


class _OneHotEmbedder(Embedder):
    """Each chunk gets a distinct unit vector; query matches by substring tag."""

    def __init__(self, tag_to_dim):
        self._map = tag_to_dim
        self._dim = max(tag_to_dim.values()) + 1

    def embed(self, texts):
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tag, d in self._map.items():
                if tag in t:
                    out[i, d] = 1.0
        return out


def test_hybrid_build_and_search_returns_chunks():
    chunks = [Chunk(0, "alpha"), Chunk(1, "beta"), Chunk(2, "gamma")]
    emb = _OneHotEmbedder({"alpha": 0, "beta": 1, "gamma": 2})
    hybrid = HybridRetriever([Retriever(emb), Retriever(emb)], pool=3)
    hybrid.build(chunks)
    results = hybrid.search("beta", k=1)
    assert results[0].chunk.index == 1
