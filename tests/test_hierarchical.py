import numpy as np

from attest.hierarchical import HierarchicalStore
from attest.interfaces import Embedder
from attest.structure import split_into_sections


def test_split_detects_real_chapter_headings():
    body = " ".join(["content"] * 400)  # big enough to pass the sanity check
    text = f"Chapter 1 Intro\n{body}\nChapter 2 Methods\n{body}\n"
    secs = split_into_sections(text, min_section_words=300)
    titles = [t for t, _ in secs]
    assert any("Chapter 1" in t for t in titles)
    assert any("Chapter 2" in t for t in titles)


def test_problem_bank_labels_do_NOT_explode_into_sections():
    # Many "3.14"-style lines with tiny bodies — the old bug made ~1 section each.
    # The sanity check must reject this and fall back to fixed blocks.
    text = "".join(f"{i}.1 Problem\nshort body text here\n" for i in range(1, 60))
    secs = split_into_sections(text, target_words=200, min_section_words=300)
    assert all(t.startswith("Section ") for t, _ in secs)  # fell back to blocks
    assert len(secs) < 30  # NOT ~60 tiny sections


def test_split_falls_back_to_blocks_without_headings():
    text = " ".join(["word"] * 7000)
    secs = split_into_sections(text, target_words=3000)
    assert len(secs) == 3
    assert secs[0][0] == "Section 1"


class _TagEmbedder(Embedder):
    _tags = {"alpha": 0, "beta": 1, "gamma": 2, "delta": 3}

    def embed(self, texts):
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            for tag, d in self._tags.items():
                if tag in t:
                    out[i, d] = 1.0
        return out


def test_hierarchical_routes_to_right_section_and_chunk():
    text = (
        "Chapter 1 A\n" + " ".join(["alpha"] * 400) + "\n"
        "Chapter 2 B\n" + " ".join(["gamma"] * 400) + "\n"
    )
    store = HierarchicalStore.build([("book.txt", text)], _TagEmbedder(), min_section_words=300)
    hits = store.search("gamma", k=1, top_sections=1)
    assert hits and "gamma" in hits[0].chunk.text
    assert "#" in hits[0].chunk.source  # records which section it came from
