import pytest

from app.chunking import chunk_pages, chunk_text


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 3
    assert all(len(c) <= 200 for c in chunks)
    # consecutive chunks share text
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.split()[-3:]) & set(b.split()[:8])
    # nothing lost: every word appears somewhere
    assert {w for c in chunks for w in c.split()} == set(text.split())


def test_no_mid_word_cuts():
    chunks = chunk_text("alpha beta gamma delta " * 50, size=60, overlap=10)
    assert all(w in {"alpha", "beta", "gamma", "delta"} for c in chunks for w in c.split())


def test_short_text_single_chunk_and_empty():
    assert chunk_text("hello world", 100, 10) == ["hello world"]
    assert chunk_text("   ", 100, 10) == []


def test_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("x", 10, 10)


def test_pages_numbered():
    out = chunk_pages(["first page", "", "third page"], 100, 10)
    assert out == [(1, "first page"), (3, "third page")]
