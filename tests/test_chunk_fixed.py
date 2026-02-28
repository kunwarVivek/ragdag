"""Tests for _chunk_fixed — overlap guard and edge cases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from ragdag.core import RagDag


def _make_dag(tmp_path):
    """Create a minimal RagDag for testing chunking methods."""
    store = tmp_path / ".ragdag"
    store.mkdir(exist_ok=True)
    (store / ".config").write_text("[general]\nchunk_size = 100\n")
    (store / ".edges").write_text("")
    (store / ".processed").write_text("")
    (store / ".domain-rules").write_text("")
    return RagDag(str(tmp_path))


class TestChunkFixed:
    """Tests for _chunk_fixed method."""

    def test_basic_chunking(self, tmp_path):
        """Text longer than chunk_size produces multiple chunks."""
        dag = _make_dag(tmp_path)
        text = "a" * 250
        chunks = dag._chunk_fixed(text, chunk_size=100, overlap=0)
        assert len(chunks) == 3
        assert chunks[0] == "a" * 100
        assert chunks[1] == "a" * 100
        assert chunks[2] == "a" * 50

    def test_overlap_produces_overlapping_content(self, tmp_path):
        """Chunks should overlap by the specified amount."""
        dag = _make_dag(tmp_path)
        text = "abcdefghij" * 10  # 100 chars
        chunks = dag._chunk_fixed(text, chunk_size=40, overlap=10)
        # Each chunk starts 30 chars after the previous
        assert len(chunks) >= 3
        # Verify overlap: end of chunk[0] == start of chunk[1]
        assert chunks[0][-10:] == chunks[1][:10]

    def test_zero_overlap(self, tmp_path):
        """Zero overlap produces non-overlapping chunks."""
        dag = _make_dag(tmp_path)
        text = "x" * 200
        chunks = dag._chunk_fixed(text, chunk_size=100, overlap=0)
        assert len(chunks) == 2
        assert "".join(chunks) == text

    def test_overlap_equals_chunk_size_no_infinite_loop(self, tmp_path):
        """When overlap >= chunk_size, must not infinite loop."""
        dag = _make_dag(tmp_path)
        text = "hello world this is a test of chunking"
        # This would infinite loop with naive `start = end - overlap`
        chunks = dag._chunk_fixed(text, chunk_size=10, overlap=10)
        assert len(chunks) >= 1
        # Must terminate and cover the full text
        assert chunks[0].startswith("hello worl")

    def test_overlap_greater_than_chunk_size_no_infinite_loop(self, tmp_path):
        """When overlap > chunk_size, must not infinite loop."""
        dag = _make_dag(tmp_path)
        text = "a" * 100
        chunks = dag._chunk_fixed(text, chunk_size=10, overlap=50)
        assert len(chunks) >= 1
        # Must terminate

    def test_text_shorter_than_chunk_size(self, tmp_path):
        """Short text produces a single chunk."""
        dag = _make_dag(tmp_path)
        text = "short"
        chunks = dag._chunk_fixed(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == "short"

    def test_empty_text(self, tmp_path):
        """Empty text produces no chunks."""
        dag = _make_dag(tmp_path)
        chunks = dag._chunk_fixed("", chunk_size=100, overlap=10)
        assert len(chunks) == 0

    def test_whitespace_only_chunks_skipped(self, tmp_path):
        """Chunks that are whitespace-only should be skipped."""
        dag = _make_dag(tmp_path)
        text = "hello" + " " * 100 + "world"
        chunks = dag._chunk_fixed(text, chunk_size=10, overlap=0)
        # All chunks should have non-whitespace content
        for chunk in chunks:
            assert chunk.strip() != ""

    def test_chunk_size_one(self, tmp_path):
        """Chunk size of 1 produces individual character chunks."""
        dag = _make_dag(tmp_path)
        text = "abc"
        chunks = dag._chunk_fixed(text, chunk_size=1, overlap=0)
        assert len(chunks) == 3
        assert chunks == ["a", "b", "c"]

    def test_exact_chunk_size_text(self, tmp_path):
        """Text exactly chunk_size long produces one chunk."""
        dag = _make_dag(tmp_path)
        text = "x" * 100
        chunks = dag._chunk_fixed(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text
