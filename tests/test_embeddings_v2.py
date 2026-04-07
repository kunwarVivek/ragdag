"""Tests for append-only embedding writes (engines/embeddings_v2.py)."""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure engines/ is importable
_engines_dir = Path(__file__).parent.parent / "engines"
sys.path.insert(0, str(_engines_dir))

from embeddings import (
    FORMAT_VERSION,
    HEADER_SIZE,
    MAGIC,
    _content_hash,
    _read_manifest_paths,
    load_embeddings,
    load_manifest,
    model_hash,
    write_embeddings,
)
from embeddings_v2 import append_embeddings, needs_rewrite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vectors(n: int, dims: int, seed: int = 42) -> list[list[float]]:
    """Generate deterministic random float32 vectors as nested lists."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, dims)).astype(np.float32)
    return arr.tolist()


def _make_chunk_paths(n: int, prefix: str = "doc") -> list[str]:
    return [f"chunks/{prefix}_{i}.md" for i in range(n)]


def _read_header_uint32(path: Path, offset: int) -> int:
    data = path.read_bytes()
    return struct.unpack_from("I", data, offset)[0]


# ---------------------------------------------------------------------------
# needs_rewrite
# ---------------------------------------------------------------------------

class TestNeedsRewrite:
    def test_returns_false_when_no_manifest(self, tmp_path):
        """No manifest file means all paths are new — no rewrite needed."""
        assert needs_rewrite(str(tmp_path), ["a.md", "b.md"]) is False

    def test_returns_false_for_new_paths(self, tmp_path):
        """Paths not in manifest should not trigger rewrite."""
        dims = 8
        vecs = _make_vectors(2, dims)
        paths = ["existing/a.md", "existing/b.md"]
        write_embeddings(str(tmp_path), vecs, paths, dims, "model-nr")

        assert needs_rewrite(str(tmp_path), ["new/c.md", "new/d.md"]) is False

    def test_returns_true_for_existing_paths(self, tmp_path):
        """Paths already in manifest should trigger rewrite."""
        dims = 8
        vecs = _make_vectors(2, dims)
        paths = ["existing/a.md", "existing/b.md"]
        write_embeddings(str(tmp_path), vecs, paths, dims, "model-nr")

        assert needs_rewrite(str(tmp_path), ["existing/a.md"]) is True

    def test_returns_true_for_partial_overlap(self, tmp_path):
        """Even one overlapping path should trigger rewrite."""
        dims = 8
        vecs = _make_vectors(2, dims)
        paths = ["existing/a.md", "existing/b.md"]
        write_embeddings(str(tmp_path), vecs, paths, dims, "model-nr")

        assert needs_rewrite(str(tmp_path), ["new/c.md", "existing/a.md"]) is True

    def test_returns_false_for_empty_paths(self, tmp_path):
        """Empty new_paths means nothing to check."""
        assert needs_rewrite(str(tmp_path), []) is False


# ---------------------------------------------------------------------------
# append_embeddings — new file creation
# ---------------------------------------------------------------------------

class TestAppendNewFile:
    def test_creates_files_from_scratch(self, tmp_path):
        """append_embeddings creates bin + manifest when no file exists."""
        dims = 8
        vecs = _make_vectors(3, dims)
        paths = _make_chunk_paths(3)
        append_embeddings(str(tmp_path), vecs, paths, dims, "model-a")

        assert (tmp_path / "embeddings.bin").exists()
        assert (tmp_path / "manifest.tsv").exists()

    def test_new_file_header_correct(self, tmp_path):
        """New file should have correct header fields."""
        dims = 16
        vecs = _make_vectors(5, dims)
        paths = _make_chunk_paths(5)
        append_embeddings(str(tmp_path), vecs, paths, dims, "model-hdr")

        bin_path = tmp_path / "embeddings.bin"
        assert _read_header_uint32(bin_path, 0) == MAGIC
        assert _read_header_uint32(bin_path, 4) == FORMAT_VERSION
        assert _read_header_uint32(bin_path, 8) == dims
        assert _read_header_uint32(bin_path, 12) == 5
        assert _read_header_uint32(bin_path, 16) == model_hash("model-hdr")

    def test_new_file_roundtrip(self, tmp_path):
        """Vectors written by append_embeddings can be loaded back correctly."""
        dims = 8
        vecs = _make_vectors(3, dims, seed=99)
        paths = _make_chunk_paths(3)
        append_embeddings(str(tmp_path), vecs, paths, dims, "model-rt")

        loaded, d, c, mh = load_embeddings(str(tmp_path / "embeddings.bin"))
        assert d == dims
        assert c == 3
        assert np.allclose(loaded, np.array(vecs, dtype=np.float32))


# ---------------------------------------------------------------------------
# append_embeddings — appending to existing
# ---------------------------------------------------------------------------

class TestAppendToExisting:
    def test_append_preserves_existing_bytes(self, tmp_path):
        """Appending must NOT rewrite the existing binary data bytes."""
        dims = 8

        # Write initial data
        v1 = _make_vectors(3, dims, seed=1)
        p1 = _make_chunk_paths(3, prefix="first")
        append_embeddings(str(tmp_path), v1, p1, dims, "model-ap")

        # Read original binary content (after header)
        bin_path = tmp_path / "embeddings.bin"
        original_bytes = bin_path.read_bytes()
        original_vector_bytes = original_bytes[HEADER_SIZE:]

        # Append new vectors
        v2 = _make_vectors(2, dims, seed=2)
        p2 = _make_chunk_paths(2, prefix="second")
        append_embeddings(str(tmp_path), v2, p2, dims, "model-ap")

        # The first 3 vectors' bytes must be identical
        new_bytes = bin_path.read_bytes()
        assert new_bytes[HEADER_SIZE:HEADER_SIZE + len(original_vector_bytes)] == original_vector_bytes

    def test_header_count_updated_after_append(self, tmp_path):
        """Header count field must reflect total vectors after append."""
        dims = 8
        v1 = _make_vectors(3, dims, seed=1)
        p1 = _make_chunk_paths(3, prefix="first")
        append_embeddings(str(tmp_path), v1, p1, dims, "model-ct")

        v2 = _make_vectors(2, dims, seed=2)
        p2 = _make_chunk_paths(2, prefix="second")
        append_embeddings(str(tmp_path), v2, p2, dims, "model-ct")

        bin_path = tmp_path / "embeddings.bin"
        assert _read_header_uint32(bin_path, 12) == 5  # 3 + 2

    def test_append_vectors_loadable(self, tmp_path):
        """All vectors (original + appended) must be loadable and correct."""
        dims = 8
        v1 = _make_vectors(3, dims, seed=1)
        p1 = _make_chunk_paths(3, prefix="first")
        append_embeddings(str(tmp_path), v1, p1, dims, "model-ld")

        v2 = _make_vectors(2, dims, seed=2)
        p2 = _make_chunk_paths(2, prefix="second")
        append_embeddings(str(tmp_path), v2, p2, dims, "model-ld")

        loaded, d, c, _ = load_embeddings(str(tmp_path / "embeddings.bin"))
        assert c == 5
        expected = np.array(v1 + v2, dtype=np.float32)
        np.testing.assert_allclose(loaded, expected)

    def test_manifest_grows_with_append(self, tmp_path):
        """Manifest should contain all paths after multiple appends."""
        dims = 8
        v1 = _make_vectors(2, dims, seed=1)
        p1 = ["a.md", "b.md"]
        append_embeddings(str(tmp_path), v1, p1, dims, "model-mg")

        v2 = _make_vectors(2, dims, seed=2)
        p2 = ["c.md", "d.md"]
        append_embeddings(str(tmp_path), v2, p2, dims, "model-mg")

        all_paths = _read_manifest_paths(str(tmp_path / "manifest.tsv"))
        assert all_paths == ["a.md", "b.md", "c.md", "d.md"]

    def test_manifest_entries_correct_after_append(self, tmp_path):
        """Manifest entries should have correct indices and byte offsets."""
        dims = 8
        v1 = _make_vectors(2, dims, seed=1)
        p1 = ["a.md", "b.md"]
        append_embeddings(str(tmp_path), v1, p1, dims, "model-me")

        v2 = _make_vectors(1, dims, seed=2)
        p2 = ["c.md"]
        append_embeddings(str(tmp_path), v2, p2, dims, "model-me")

        entries = load_manifest(str(tmp_path / "manifest.tsv"))
        assert len(entries) == 3
        for i, entry in enumerate(entries):
            assert entry[1] == i  # index
            assert entry[2] == HEADER_SIZE + i * dims * 4  # byte_offset


# ---------------------------------------------------------------------------
# append_embeddings — empty vectors
# ---------------------------------------------------------------------------

class TestAppendEmpty:
    def test_empty_vectors_is_noop(self, tmp_path):
        """Appending empty vectors should not create or modify files."""
        append_embeddings(str(tmp_path), [], [], 8, "model-noop")
        assert not (tmp_path / "embeddings.bin").exists()

    def test_empty_append_to_existing_is_noop(self, tmp_path):
        """Appending empty vectors to existing file should not modify it."""
        dims = 8
        v1 = _make_vectors(3, dims, seed=1)
        p1 = _make_chunk_paths(3)
        append_embeddings(str(tmp_path), v1, p1, dims, "model-en")

        bin_path = tmp_path / "embeddings.bin"
        original_bytes = bin_path.read_bytes()

        append_embeddings(str(tmp_path), [], [], dims, "model-en")

        assert bin_path.read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# Content hashes with append
# ---------------------------------------------------------------------------

class TestAppendContentHashes:
    def test_content_hashes_written(self, tmp_path):
        """Content hashes should be recorded in manifest on append."""
        dims = 4
        vecs = _make_vectors(2, dims)
        paths = ["a.md", "b.md"]
        texts = ["hello", "world"]
        append_embeddings(str(tmp_path), vecs, paths, dims, "model-ch",
                          chunk_texts=texts)

        entries = load_manifest(str(tmp_path / "manifest.tsv"))
        assert entries[0][4] == _content_hash("hello")
        assert entries[1][4] == _content_hash("world")

    def test_content_hashes_on_subsequent_append(self, tmp_path):
        """Content hashes should be correct for appended entries too."""
        dims = 4
        v1 = _make_vectors(1, dims, seed=1)
        append_embeddings(str(tmp_path), v1, ["a.md"], dims, "model-ch2",
                          chunk_texts=["first"])

        v2 = _make_vectors(1, dims, seed=2)
        append_embeddings(str(tmp_path), v2, ["b.md"], dims, "model-ch2",
                          chunk_texts=["second"])

        entries = load_manifest(str(tmp_path / "manifest.tsv"))
        assert len(entries) == 2
        assert entries[0][4] == _content_hash("first")
        assert entries[1][4] == _content_hash("second")


# ---------------------------------------------------------------------------
# Replace triggers full rewrite
# ---------------------------------------------------------------------------

class TestReplaceTriggersRewrite:
    def test_replace_existing_path_produces_correct_output(self, tmp_path):
        """When replacing an existing path, output must be correct."""
        dims = 8

        # Write initial
        v1 = _make_vectors(3, dims, seed=1)
        p1 = ["a.md", "b.md", "c.md"]
        write_embeddings(str(tmp_path), v1, p1, dims, "model-rp")

        # Replace b.md — needs_rewrite should be True, so append_embeddings
        # should fall back to full rewrite
        v2 = _make_vectors(1, dims, seed=99)
        p2 = ["b.md"]
        append_embeddings(str(tmp_path), v2, p2, dims, "model-rp")

        loaded, _, c, _ = load_embeddings(str(tmp_path / "embeddings.bin"))
        # Should have 3 entries: a.md, c.md (kept), b.md (replaced)
        assert c == 3

        result_paths = _read_manifest_paths(str(tmp_path / "manifest.tsv"))
        assert set(result_paths) == {"a.md", "b.md", "c.md"}

    def test_replace_uses_new_vector(self, tmp_path):
        """Replaced path should have the new vector, not the old one."""
        dims = 4

        v1 = [[0.1, 0.2, 0.3, 0.4]]
        write_embeddings(str(tmp_path), v1, ["a.md"], dims, "model-rv")

        v2 = [[0.9, 0.8, 0.7, 0.6]]
        append_embeddings(str(tmp_path), v2, ["a.md"], dims, "model-rv")

        loaded, _, c, _ = load_embeddings(str(tmp_path / "embeddings.bin"))
        assert c == 1
        np.testing.assert_allclose(loaded[0], v2[0], atol=1e-6)


# ---------------------------------------------------------------------------
# Fast-path integration (write_embeddings delegates to append_embeddings)
# ---------------------------------------------------------------------------

class TestFastPathIntegration:
    def test_write_embeddings_append_new_paths_uses_fast_path(self, tmp_path):
        """write_embeddings with append=True and new paths should use fast path
        and preserve existing bytes."""
        dims = 8

        # Initial write
        v1 = _make_vectors(3, dims, seed=10)
        p1 = _make_chunk_paths(3, prefix="init")
        write_embeddings(str(tmp_path), v1, p1, dims, "model-fp")

        # Capture original vector bytes
        bin_path = tmp_path / "embeddings.bin"
        original_bytes = bin_path.read_bytes()
        original_vector_bytes = original_bytes[HEADER_SIZE:]

        # Append via write_embeddings (should use fast path)
        v2 = _make_vectors(2, dims, seed=20)
        p2 = _make_chunk_paths(2, prefix="new")
        write_embeddings(str(tmp_path), v2, p2, dims, "model-fp", append=True)

        # Verify original bytes preserved
        new_bytes = bin_path.read_bytes()
        assert new_bytes[HEADER_SIZE:HEADER_SIZE + len(original_vector_bytes)] == original_vector_bytes

        # Verify all data correct
        loaded, _, c, _ = load_embeddings(str(bin_path))
        assert c == 5
        expected = np.array(v1 + v2, dtype=np.float32)
        np.testing.assert_allclose(loaded, expected)
