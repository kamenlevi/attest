from attest.backends.mock import MockEmbedder
from attest.chunking import Chunk
from attest.store import IndexedStore


def _docs():
    return [
        ("bookA.txt", [Chunk(0, "photosynthesis happens in chloroplasts"),
                       Chunk(1, "stomata let carbon dioxide in")]),
        ("bookB.txt", [Chunk(0, "the lorentz transformation mixes space and time")]),
    ]


def test_build_assigns_global_uids_and_sources():
    store = IndexedStore.build(_docs(), MockEmbedder())
    assert len(store) == 3
    hits = store.search("chloroplasts", k=1)
    assert hits[0].chunk.source == "bookA.txt"
    assert hits[0].chunk.text == "photosynthesis happens in chloroplasts"


def test_save_then_load_round_trips(tmp_path):
    IndexedStore.build(_docs(), MockEmbedder()).save(tmp_path / "idx")
    loaded = IndexedStore.load(tmp_path / "idx", MockEmbedder())
    assert len(loaded) == 3
    # Query the second book; uid should be global (2), source preserved.
    hit = loaded.search("lorentz space and time", k=1)[0]
    assert hit.chunk.source == "bookB.txt"
    assert hit.chunk.index == 2  # global uid across both docs


def test_empty_index_search_returns_nothing():
    store = IndexedStore.build([], MockEmbedder())
    assert store.search("anything", k=3) == []
