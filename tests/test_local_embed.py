"""Real-embedding test. Skipped by default because it downloads a model.

Run it deliberately with:
    ATTEST_TEST_LOCAL_EMBED=1 python -m pytest tests/test_local_embed.py
(after `pip install -e '.[embed-local]'`).
"""

import os

import numpy as np
import pytest

if not os.environ.get("ATTEST_TEST_LOCAL_EMBED"):
    pytest.skip(
        "set ATTEST_TEST_LOCAL_EMBED=1 to run (downloads a small model)",
        allow_module_level=True,
    )


def test_local_embedder_finds_meaning_not_just_words():
    from attest.backends.local_embed import LocalEmbedder
    from attest.chunking import Chunk
    from attest.retrieval import Retriever

    chunks = [
        Chunk(0, "A car is a road vehicle with an engine."),
        Chunk(1, "Bananas are a yellow tropical fruit."),
    ]
    retriever = Retriever(LocalEmbedder())
    retriever.build(chunks)

    # "automobile" never appears, but a meaning-aware embedder should still rank
    # the car passage first — something the lexical mock embedder cannot do.
    top = retriever.search("What is an automobile?", k=1)[0]
    assert top.chunk.index == 0
    # Sanity: vectors are unit length.
    v = LocalEmbedder().embed(["hello world"])[0]
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3
