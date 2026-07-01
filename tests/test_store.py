from attest.backends.mock import MockEmbedder
from attest.chunking import Chunk
from attest.store import IndexedStore, file_fingerprint


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


class _CountingEmbedder(MockEmbedder):
    """Counts how many texts get embedded, to prove we don't re-embed."""

    def __init__(self) -> None:
        super().__init__()
        self.embedded = 0

    def embed(self, texts):
        self.embedded += len(texts)
        return super().embed(texts)


def test_add_is_idempotent_skips_already_indexed():
    emb = _CountingEmbedder()
    store = IndexedStore.build(_docs(), emb)
    after_first = emb.embedded
    # Re-adding the SAME docs must skip everything — no new embedding.
    stats = store.add(_docs(), emb)
    assert stats == {"added": 0, "skipped": 2, "updated": 0}
    assert emb.embedded == after_first  # nothing re-embedded
    assert len(store) == 3


def test_add_new_file_appends_only_the_new_one():
    emb = _CountingEmbedder()
    store = IndexedStore.build(_docs(), emb)
    before = emb.embedded
    stats = store.add([("bookC.txt", [Chunk(0, "a brand new passage about tides")])], emb)
    assert stats == {"added": 1, "skipped": 0, "updated": 0}
    assert emb.embedded == before + 1  # only the one new chunk embedded
    assert len(store) == 4
    assert store.search("tides", k=1)[0].chunk.source == "bookC.txt"


def test_changed_file_is_reindexed_old_chunks_dropped():
    store = IndexedStore.build(_docs(), MockEmbedder())
    edited = [("bookB.txt", [Chunk(0, "the lorentz transformation mixes space and time"),
                             Chunk(1, "time dilation slows moving clocks")])]
    stats = store.add(edited, MockEmbedder())
    assert stats == {"added": 0, "skipped": 0, "updated": 1}
    assert len(store) == 4  # bookA(2) + new bookB(2); old bookB(1) dropped
    assert store.search("time dilation clocks", k=1)[0].chunk.source == "bookB.txt"


def test_file_fingerprint_tracks_content(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("abc")
    first = file_fingerprint(p)
    assert file_fingerprint(p) == first  # stable for identical content
    p.write_text("abcd")
    assert file_fingerprint(p) != first  # changes when the file changes


def test_explicit_fingerprints_skip_without_reembedding():
    # Mirrors how the CLI works: decide skip from a cheap file fingerprint,
    # never hashing or re-embedding the chunks of an unchanged file.
    emb = _CountingEmbedder()
    store = IndexedStore.build([], emb)
    docs = [("a.txt", [Chunk(0, "hello world")])]
    store.add(docs, emb, fingerprints={"a.txt": "FP1"})
    assert store.fingerprint_of("a.txt") == "FP1"
    before = emb.embedded
    stats = store.add(docs, emb, fingerprints={"a.txt": "FP1"})  # unchanged fp
    assert stats["skipped"] == 1 and emb.embedded == before
    stats = store.add(docs, emb, fingerprints={"a.txt": "FP2"})  # changed fp
    assert stats["updated"] == 1


def test_manifest_persists_so_reload_still_skips(tmp_path):
    IndexedStore.build(_docs(), MockEmbedder()).save(tmp_path / "idx")
    reloaded = IndexedStore.load(tmp_path / "idx", MockEmbedder())
    assert set(reloaded.sources()) == {"bookA.txt", "bookB.txt"}
    stats = reloaded.add(_docs(), MockEmbedder())  # unchanged -> all skipped
    assert stats["skipped"] == 2 and stats["added"] == 0


def test_page_numbers_round_trip_through_save_and_load(tmp_path):
    docs = [("paged.pdf", [Chunk(0, "text on the fifth page", page=5),
                           Chunk(1, "text with no page")])]
    IndexedStore.build(docs, MockEmbedder()).save(tmp_path / "idx")
    loaded = IndexedStore.load(tmp_path / "idx", MockEmbedder())
    hit = loaded.search("fifth page text", k=1)[0]
    assert hit.chunk.page == 5
    pages = {c.index: c.page for c in loaded.chunks()}
    assert pages == {0: 5, 1: None}
