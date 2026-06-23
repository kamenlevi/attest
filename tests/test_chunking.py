from attest.chunking import chunk_text


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
