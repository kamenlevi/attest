"""BM25 lexical retriever + fusion tests — the exact-term half of recall."""

from attest.chunking import Chunk
from attest.lexical import BM25Retriever, _tokenize
from attest.retrieval import FusedRetriever, Retrieved


def _chunks():
    return [
        Chunk(0, "This book is a consequence of the vision of Walter of Merton in 1264"),
        Chunk(1, "Box 2.1: Hermitian operators have real eigenvalues and orthogonal eigenvectors"),
        Chunk(2, "The harmonic oscillator has equally spaced energy levels"),
        Chunk(3, "Probability is conserved by the continuity equation"),
    ]


def test_tokenizer_keeps_dotted_labels():
    assert "2.1" in _tokenize("the subject of Box 2.1 is operators")
    assert "2.87" in _tokenize("equation (2.87) gives the current")


def test_bm25_finds_rare_proper_noun_semantic_would_miss():
    bm25 = BM25Retriever(_chunks())
    hits = bm25.search("who was Walter of Merton", k=1)
    assert hits and hits[0].chunk.index == 0  # the rare names dominate the score


def test_bm25_finds_label_by_dotted_token():
    bm25 = BM25Retriever(_chunks())
    hits = bm25.search("what is in Box 2.1", k=1)
    assert hits and hits[0].chunk.index == 1


def test_bm25_returns_nothing_when_no_term_overlap():
    bm25 = BM25Retriever(_chunks())
    assert bm25.search("xylophone zebra quasar", k=3) == []


class _FakeSemantic:
    """Returns a fixed semantic ranking that misses the proper-noun chunk."""

    def search(self, query, k=4):
        return [Retrieved(Chunk(i, f"c{i}"), 1.0) for i in (2, 3)][:k]


def test_fusion_unions_semantic_and_lexical_recall():
    # Semantic returns {2,3}; BM25 finds the proper-noun chunk 0. Fused must have both.
    fused = FusedRetriever([_FakeSemantic(), BM25Retriever(_chunks())], pool=10)
    got = {r.chunk.index for r in fused.search("Walter of Merton", k=4)}
    assert 0 in got          # recovered purely via the lexical retriever
    assert got & {2, 3}      # semantic hits still present
