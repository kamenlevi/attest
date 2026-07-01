from attest.chunking import chunk_pages, chunk_text


def test_chunks_cover_text_with_overlap():
    text = " ".join(str(i) for i in range(100))  # 100 words: "0 1 2 ... 99"
    chunks = chunk_text(text, chunk_size=10, overlap=2)

    # step = 10 - 2 = 8; windows start at 0,8,16,... -> 13 chunks to cover 100.
    assert len(chunks) == 13
    assert chunks[0].index == 0
    assert chunks[0].text.split()[:10] == [str(i) for i in range(10)]
    # Overlap: last 2 words of chunk 0 reappear as first 2 words of chunk 1.
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]


def test_empty_text_yields_no_chunks():
    assert chunk_text("   ") == []


def test_invalid_overlap_rejected():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=5, overlap=5)


def test_chunk_pages_records_starting_page():
    pages = [(1, " ".join(f"p1w{i}" for i in range(10))),
             (2, " ".join(f"p2w{i}" for i in range(10)))]
    chunks = chunk_pages(pages, chunk_size=8, overlap=0)
    # 20 words in windows of 8: starts at word 0 (page 1), 8 (page 1), 16 (page 2)
    assert [c.page for c in chunks] == [1, 1, 2]
    # The word stream flows across the page boundary — nothing lost at the break.
    assert "p1w8" in chunks[1].text and "p2w0" in chunks[1].text


def test_chunk_pages_none_page_for_plain_text():
    chunks = chunk_pages([(None, "just some plain text with no pages")])
    assert chunks[0].page is None
