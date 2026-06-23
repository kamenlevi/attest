"""Integration test: the whole pipeline on the mock backend, no model needed.

A well-behaved grounded system should answer questions whose answers are in the
document and abstain on traps -> bluff rate 0. This guards the plumbing end to
end and is the Linux-runnable proxy for what we'll later measure on the Mac.
"""

from attest.backends.mock import MockEmbedder, MockGenerator
from attest.chunking import chunk_text
from attest.eval import EvalItem, compute_metrics, run_eval
from attest.retrieval import Retriever

DOC = (
    "Photosynthesis happens in chloroplasts. The pigment chlorophyll absorbs "
    "sunlight. Plants take in carbon dioxide through pores called stomata and "
    "release oxygen."
)

ITEMS = [
    EvalItem("What pigment absorbs sunlight?", answerable=True),
    EvalItem("What are the pores called?", answerable=True),
    EvalItem("What is the boiling point of helium?", answerable=False),
    EvalItem("Who wrote the Moonlight Sonata?", answerable=False),
]


def test_mock_pipeline_has_zero_bluff_and_some_coverage():
    retriever = Retriever(MockEmbedder())
    retriever.build(chunk_text(DOC, chunk_size=20, overlap=5))
    results = run_eval(ITEMS, retriever, MockGenerator(), k=3)
    m = compute_metrics(results)

    assert m["bluff_rate"] == 0.0           # abstains on every trap
    assert m["answer_coverage"] > 0.0       # answers at least some answerable ones
