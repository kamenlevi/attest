"""End-to-end demo on the MOCK backend — runs on Linux, no model needed.

    python -m attest.demo

It chunks a tiny text-only document, builds the retriever, asks a few questions
(some answerable, some traps), and prints the trust report. This proves the
whole pipeline and the measurement work before we ever touch a real model.
"""

from __future__ import annotations

from .backends.mock import MockEmbedder, MockGenerator
from .chunking import chunk_text
from .eval import EvalItem, compute_metrics, format_report, run_eval
from .retrieval import Retriever

# A short, text-only document (deliberately no equations for the first run).
DOCUMENT = """
Photosynthesis is the process by which green plants convert light energy into
chemical energy. It takes place mainly in the leaves, inside organelles called
chloroplasts. The green pigment chlorophyll absorbs sunlight, primarily red and
blue wavelengths, and reflects green light, which is why leaves look green.

During photosynthesis, plants take in carbon dioxide from the air through small
pores called stomata, and absorb water from the soil through their roots. Using
light energy, they combine carbon dioxide and water to produce glucose, a sugar
that stores energy, and release oxygen as a by-product.

The process has two main stages. The light-dependent reactions capture energy
from sunlight and store it temporarily. The light-independent reactions, also
called the Calvin cycle, use that stored energy to build glucose from carbon
dioxide.
""".strip()

# Ground truth: which questions are genuinely answerable from the document above.
EVAL_ITEMS = [
    EvalItem("What pigment absorbs sunlight in plants?", answerable=True),
    EvalItem("Through what pores do plants take in carbon dioxide?", answerable=True),
    EvalItem("What is the Calvin cycle used for?", answerable=True),
    EvalItem("Why do leaves look green?", answerable=True),
    # Traps: plausible, but the answers are NOT in this document.
    EvalItem("What is the boiling point of helium?", answerable=False),
    EvalItem("Who composed the Moonlight Sonata?", answerable=False),
    EvalItem("How does a transistor amplify current?", answerable=False),
]


def main() -> None:
    chunks = chunk_text(DOCUMENT, chunk_size=60, overlap=15)
    retriever = Retriever(MockEmbedder())
    retriever.build(chunks)

    results = run_eval(EVAL_ITEMS, retriever, MockGenerator(), k=3)
    print(f"Document split into {len(chunks)} chunks.\n")
    for r in results:
        verdict = "ABSTAINED" if r.answer.abstained else f"answered {r.answer.citations}"
        print(f"  [{'answerable' if r.item.answerable else 'trap      '}] "
              f"{r.item.question}\n      -> {verdict}")
    print()
    print(format_report(compute_metrics(results)))


if __name__ == "__main__":
    main()
