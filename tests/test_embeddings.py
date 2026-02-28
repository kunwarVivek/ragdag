"""Tests for the ragdag binary embeddings format (engines/embeddings.py)."""

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
    _read_manifest_paths,
    load_embeddings,
    load_embeddings_mmap,
    load_manifest,
    model_hash,
    write_embeddings,
)


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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_header_size_is_32(self):
        """HEADER_SIZE constant must be 32 bytes."""
        assert HEADER_SIZE == 32

    def test_magic_value(self):
        """MAGIC must equal 0x52414744 ('RAGD')."""
        assert MAGIC == 0x52414744

    def test_format_version(self):
        """FORMAT_VERSION must be 1."""
        assert FORMAT_VERSION == 1


# ---------------------------------------------------------------------------
# model_hash
# ---------------------------------------------------------------------------

class TestModelHash:
    def test_model_hash_deterministic(self):
        """Same model name always produces the same hash."""
        name = "text-embedding-3-small"
        assert model_hash(name) == model_hash(name)

    def test_model_hash_different_models(self):
        """Different model names produce different hashes."""
        h1 = model_hash("text-embedding-3-small")
        h2 = model_hash("text-embedding-ada-002")
        assert h1 != h2

    def test_model_hash_is_uint32(self):
        """Hash must fit in an unsigned 32-bit integer."""
        h = model_hash("any-model-name")
        assert 0 <= h < 2**32


# ---------------------------------------------------------------------------
# write_embeddings — file creation
# ---------------------------------------------------------------------------

class TestWriteCreatesFiles:
    def test_write_creates_binary_file(self, tmp_path):
        """write_embeddings must create embeddings.bin."""
        vecs = _make_vectors(3, 8)
        paths = _make_chunk_paths(3)
        write_embeddings(str(tmp_path), vecs, paths, 8, "model-a")
        assert (tmp_path / "embeddings.bin").exists()

    def test_manifest_created(self, tmp_path):
        """write_embeddings must create manifest.tsv alongside the bin."""
        vecs = _make_vectors(3, 8)
        paths = _make_chunk_paths(3)
        write_embeddings(str(tmp_path), vecs, paths, 8, "model-a")
        assert (tmp_path / "manifest.tsv").exists()

    def test_write_empty_vectors(self, tmp_path):
        """Empty vector list must not create any file."""
        write_embeddings(str(tmp_path), [], [], 8, "model-a")
        assert not (tmp_path / "embeddings.bin").exists()
        assert not (tmp_path / "manifest.tsv").exists()


# ---------------------------------------------------------------------------
# Binary header fields
# ---------------------------------------------------------------------------

class TestBinaryHeader:
    @pytest.fixture()
    def bin_path(self, tmp_path):
        dims = 16
        vecs = _make_vectors(5, dims)
        paths = _make_chunk_paths(5)
        write_embeddings(str(tmp_path), vecs, paths, dims, "model-x")
        return tmp_path / "embeddings.bin"

    def _read_header_uint32(self, path: Path, offset: int) -> int:
        data = path.read_bytes()
        return struct.unpack_from("I", data, offset)[0]

    def test_binary_magic_number(self, bin_path):
        """First 4 bytes of the binary file must be the MAGIC number."""
        assert self._read_header_uint32(bin_path, 0) == MAGIC

    def test_binary_format_version(self, bin_path):
        """Bytes 4-8 must contain FORMAT_VERSION (1)."""
        assert self._read_header_uint32(bin_path, 4) == FORMAT_VERSION

    def test_binary_dimensions_field(self, bin_path):
        """Bytes 8-12 must match the dimensions argument (16)."""
        assert self._read_header_uint32(bin_path, 8) == 16

    def test_binary_vector_count(self, bin_path):
        """Bytes 12-16 must equal the number of vectors written (5)."""
        assert self._read_header_uint32(bin_path, 12) == 5

    def test_binary_model_hash_field(self, bin_path):
        """Bytes 16-20 must equal model_hash('model-x')."""
        assert self._read_header_uint32(bin_path, 16) == model_hash("model-x")

    def test_binary_reserved_bytes_are_zero(self, bin_path):
        """Bytes 20-32 (reserved) must be zero-filled."""
        data = bin_path.read_bytes()
        assert data[20:32] == b"\x00" * 12

    def test_binary_total_size(self, bin_path):
        """File size = header (32) + count * dims * 4."""
        expected = HEADER_SIZE + 5 * 16 * 4
        assert bin_path.stat().st_size == expected


# ---------------------------------------------------------------------------
# load round-trips
# ---------------------------------------------------------------------------

class TestLoadRoundtrip:
    def test_load_roundtrip(self, tmp_path):
        """write then load must return the same vectors (np.allclose)."""
        dims = 32
        vecs = _make_vectors(4, dims)
        paths = _make_chunk_paths(4)
        write_embeddings(str(tmp_path), vecs, paths, dims, "model-rt")

        loaded, d, c, mh = load_embeddings(str(tmp_path / "embeddings.bin"))
        assert d == dims
        assert c == 4
        assert mh == model_hash("model-rt")
        assert np.allclose(loaded, np.array(vecs, dtype=np.float32))

    def test_load_mmap_roundtrip(self, tmp_path):
        """write then load_mmap must return the same vectors (np.allclose)."""
        dims = 32
        vecs = _make_vectors(4, dims, seed=99)
        paths = _make_chunk_paths(4)
        write_embeddings(str(tmp_path), vecs, paths, dims, "model-mm")

        loaded, d, c, mh = load_embeddings_mmap(str(tmp_path / "embeddings.bin"))
        assert d == dims
        assert c == 4
        assert mh == model_hash("model-mm")
        assert np.allclose(loaded, np.array(vecs, dtype=np.float32))

    def test_load_and_mmap_agree(self, tmp_path):
        """Both load functions must return identical data."""
        dims = 64
        vecs = _make_vectors(10, dims, seed=7)
        paths = _make_chunk_paths(10)
        write_embeddings(str(tmp_path), vecs, paths, dims, "model-cmp")

        bin_file = str(tmp_path / "embeddings.bin")
        v1, d1, c1, h1 = load_embeddings(bin_file)
        v2, d2, c2, h2 = load_embeddings_mmap(bin_file)

        assert d1 == d2
        assert c1 == c2
        assert h1 == h2
        np.testing.assert_array_equal(v1, v2)


# ---------------------------------------------------------------------------
# load validation
# ---------------------------------------------------------------------------

class TestLoadValidation:
    def test_load_invalid_magic_raises(self, tmp_path):
        """A file with wrong magic bytes must raise ValueError."""
        bad = tmp_path / "bad.bin"
        bad.write_bytes(b"\x00" * 64)
        with pytest.raises(ValueError, match="Not a ragdag embeddings file"):
            load_embeddings(str(bad))

    def test_load_mmap_invalid_magic_raises(self, tmp_path):
        """load_embeddings_mmap must also reject bad magic."""
        bad = tmp_path / "bad.bin"
        bad.write_bytes(b"\xff" * 64)
        with pytest.raises(ValueError, match="Not a ragdag embeddings file"):
            load_embeddings_mmap(str(bad))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_roundtrip(self, tmp_path):
        """load_manifest returns (path, index, offset, dims) for each entry."""
        dims = 8
        vecs = _make_vectors(3, dims)
        paths = _make_chunk_paths(3, prefix="mf")
        write_embeddings(str(tmp_path), vecs, paths, dims, "model-mf")

        entries = load_manifest(str(tmp_path / "manifest.tsv"))
        assert len(entries) == 3
        for i, (p, idx, offset, d) in enumerate(entries):
            assert p == paths[i]
            assert idx == i
            assert offset == HEADER_SIZE + i * dims * 4
            assert d == dims

    def test_manifest_paths(self, tmp_path):
        """_read_manifest_paths returns just the chunk path strings."""
        dims = 8
        vecs = _make_vectors(2, dims)
        paths = ["a/b.md", "c/d.md"]
        write_embeddings(str(tmp_path), vecs, paths, dims, "model-p")

        result = _read_manifest_paths(str(tmp_path / "manifest.tsv"))
        assert result == paths

    def test_manifest_skips_comments_and_blanks(self, tmp_path):
        """Manifest parser must skip comment and blank lines."""
        mf = tmp_path / "manifest.tsv"
        mf.write_text(
            "# header comment\n"
            "\n"
            "path/a.md\t0\t32\t8\n"
            "# another comment\n"
            "path/b.md\t1\t64\t8\n"
        )
        assert _read_manifest_paths(str(mf)) == ["path/a.md", "path/b.md"]
        entries = load_manifest(str(mf))
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# Append mode
# ---------------------------------------------------------------------------

class TestAppendMode:
    def test_append_mode_adds_vectors(self, tmp_path):
        """append=True adds new vectors to existing file."""
        dims = 8

        # First write: 3 vectors
        v1 = _make_vectors(3, dims, seed=1)
        p1 = _make_chunk_paths(3, prefix="first")
        write_embeddings(str(tmp_path), v1, p1, dims, "model-app")

        # Second write: 2 new vectors (different paths)
        v2 = _make_vectors(2, dims, seed=2)
        p2 = _make_chunk_paths(2, prefix="second")
        write_embeddings(str(tmp_path), v2, p2, dims, "model-app", append=True)

        loaded, d, c, _ = load_embeddings(str(tmp_path / "embeddings.bin"))
        assert c == 5
        assert d == dims

        # The first 3 vectors should match v1, last 2 should match v2
        expected = np.array(v1 + v2, dtype=np.float32)
        np.testing.assert_allclose(loaded, expected)

        # Manifest should list all 5 paths
        all_paths = _read_manifest_paths(str(tmp_path / "manifest.tsv"))
        assert all_paths == p1 + p2

    def test_append_dedup(self, tmp_path):
        """Re-appending the same paths replaces old vectors with new ones."""
        dims = 8

        # First write
        v1 = _make_vectors(3, dims, seed=10)
        paths = _make_chunk_paths(3, prefix="dup")
        write_embeddings(str(tmp_path), v1, paths, dims, "model-dd")

        # Second write — same paths, different vectors
        v2 = _make_vectors(3, dims, seed=20)
        write_embeddings(str(tmp_path), v2, paths, dims, "model-dd", append=True)

        loaded, _, c, _ = load_embeddings(str(tmp_path / "embeddings.bin"))
        assert c == 3  # not 6 — old ones replaced

        expected = np.array(v2, dtype=np.float32)
        np.testing.assert_allclose(loaded, expected)

    def test_append_partial_overlap(self, tmp_path):
        """Appending with partial path overlap deduplicates only the overlap."""
        dims = 4

        v1 = _make_vectors(3, dims, seed=30)
        p1 = ["a.md", "b.md", "c.md"]
        write_embeddings(str(tmp_path), v1, p1, dims, "model-po")

        # Overlap on b.md; add d.md
        v2 = _make_vectors(2, dims, seed=40)
        p2 = ["b.md", "d.md"]
        write_embeddings(str(tmp_path), v2, p2, dims, "model-po", append=True)

        loaded, _, c, _ = load_embeddings(str(tmp_path / "embeddings.bin"))
        assert c == 4  # a, c (kept), b (replaced), d (new)

        result_paths = _read_manifest_paths(str(tmp_path / "manifest.tsv"))
        assert set(result_paths) == {"a.md", "c.md", "b.md", "d.md"}

    def test_append_false_overwrites(self, tmp_path):
        """append=False ignores existing data and overwrites."""
        dims = 8

        v1 = _make_vectors(5, dims, seed=50)
        p1 = _make_chunk_paths(5, prefix="old")
        write_embeddings(str(tmp_path), v1, p1, dims, "model-ow")

        v2 = _make_vectors(2, dims, seed=60)
        p2 = _make_chunk_paths(2, prefix="new")
        write_embeddings(str(tmp_path), v2, p2, dims, "model-ow", append=False)

        loaded, _, c, _ = load_embeddings(str(tmp_path / "embeddings.bin"))
        assert c == 2
        expected = np.array(v2, dtype=np.float32)
        np.testing.assert_allclose(loaded, expected)
