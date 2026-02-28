"""Tests for the ragdag cosine similarity engine (engines/similarity.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable (for `from engines.X import ...`)
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from engines.similarity import cosine_similarity, search_vectors
from engines.embeddings import write_embeddings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    return v / (np.linalg.norm(v) + 1e-10)


def _write_domain(
    store_dir: Path,
    domain: str,
    vectors: list[list[float]],
    chunk_paths: list[str],
    dims: int,
) -> None:
    """Write embeddings into a domain subdirectory of a .ragdag store."""
    domain_dir = store_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    write_embeddings(str(domain_dir), vectors, chunk_paths, dims, "test-model")


# ---------------------------------------------------------------------------
# cosine_similarity -- unit tests
# ---------------------------------------------------------------------------

class TestCosineIdenticalVectors:
    def test_identical_vectors_similarity_is_one(self):
        """Identical vectors must produce cosine similarity approximately 1.0."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        matrix = v.reshape(1, -1)
        scores = cosine_similarity(v, matrix)
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(1.0, abs=1e-6)

    def test_identical_unit_vectors(self):
        """Identical unit vectors must yield similarity exactly ~1.0."""
        v = _norm(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
        matrix = v.reshape(1, -1)
        scores = cosine_similarity(v, matrix)
        assert scores[0] == pytest.approx(1.0, abs=1e-6)


class TestCosineOrthogonalVectors:
    def test_orthogonal_vectors_similarity_is_zero(self):
        """Orthogonal vectors must produce cosine similarity approximately 0.0."""
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        scores = cosine_similarity(q, v)
        assert scores[0] == pytest.approx(0.0, abs=1e-6)

    def test_all_basis_orthogonal(self):
        """Each basis vector is orthogonal to the others."""
        basis = np.eye(3, dtype=np.float32)
        q = basis[0]  # [1, 0, 0]
        scores = cosine_similarity(q, basis)
        assert scores[0] == pytest.approx(1.0, abs=1e-6)
        assert scores[1] == pytest.approx(0.0, abs=1e-6)
        assert scores[2] == pytest.approx(0.0, abs=1e-6)


class TestCosineOppositeVectors:
    def test_opposite_vectors_similarity_is_negative_one(self):
        """Opposite vectors must produce cosine similarity approximately -1.0."""
        q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
        scores = cosine_similarity(q, v)
        assert scores[0] == pytest.approx(-1.0, abs=1e-6)


class TestCosineSingleVectorMatrix:
    def test_single_vector_matrix_returns_one_score(self):
        """Query vs. a single-row matrix returns exactly one score."""
        q = np.array([3.0, 4.0], dtype=np.float32)
        v = np.array([[4.0, 3.0]], dtype=np.float32)
        scores = cosine_similarity(q, v)
        assert scores.shape == (1,)
        # cos(q, v) = (12+12) / (5 * 5) = 24/25 = 0.96
        assert scores[0] == pytest.approx(24.0 / 25.0, abs=1e-5)


class TestCosineMultipleVectors:
    def test_multiple_vectors_ordering(self):
        """Similarity scores must rank vectors by closeness to the query."""
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        matrix = np.array(
            [
                [0.0, 1.0, 0.0],   # orthogonal -> 0.0
                [1.0, 0.0, 0.0],   # identical  -> 1.0
                [1.0, 1.0, 0.0],   # 45 degrees -> ~0.707
                [-1.0, 0.0, 0.0],  # opposite   -> -1.0
            ],
            dtype=np.float32,
        )
        scores = cosine_similarity(q, matrix)
        assert len(scores) == 4
        # Check relative ordering: identical > 45-deg > orthogonal > opposite
        assert scores[1] > scores[2] > scores[0] > scores[3]
        assert scores[1] == pytest.approx(1.0, abs=1e-6)
        assert scores[0] == pytest.approx(0.0, abs=1e-6)
        assert scores[3] == pytest.approx(-1.0, abs=1e-6)

    def test_returns_ndarray(self):
        """Return type must be a numpy array."""
        q = np.array([1.0, 0.0], dtype=np.float32)
        v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        scores = cosine_similarity(q, v)
        assert isinstance(scores, np.ndarray)


class TestCosineZeroVector:
    def test_zero_query_does_not_crash(self):
        """A zero query vector must not raise (epsilon prevents division by zero)."""
        q = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        v = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        scores = cosine_similarity(q, v)
        assert scores.shape == (1,)
        # Score should be ~0 since zero vector has no direction
        assert scores[0] == pytest.approx(0.0, abs=1e-4)

    def test_zero_matrix_row_does_not_crash(self):
        """A zero vector in the matrix must not raise."""
        q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        scores = cosine_similarity(q, v)
        assert scores.shape == (1,)
        # Score should be ~0 since zero vector has no direction
        assert scores[0] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# search_vectors -- integration tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def vector_store(tmp_path):
    """Create a temp .ragdag store with two domains and known embeddings.

    Domain "code":
        chunk_a -> [1, 0, 0]
        chunk_b -> [0, 1, 0]
        chunk_c -> [0, 0, 1]

    Domain "docs":
        chunk_d -> [1, 1, 0]  (normalized in storage, raw here)
        chunk_e -> [0, 1, 1]
    """
    store = tmp_path / ".ragdag"
    store.mkdir()

    dims = 3

    _write_domain(
        store,
        "code",
        vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        chunk_paths=["chunk_a", "chunk_b", "chunk_c"],
        dims=dims,
    )
    _write_domain(
        store,
        "docs",
        vectors=[[1, 1, 0], [0, 1, 1]],
        chunk_paths=["chunk_d", "chunk_e"],
        dims=dims,
    )
    return store


class TestSearchVectorsBasic:
    def test_search_returns_sorted_results(self, vector_store):
        """search_vectors must return results sorted by descending score."""
        query = [1.0, 0.0, 0.0]  # most similar to chunk_a ([1,0,0])
        results = search_vectors(query, str(vector_store))
        assert len(results) > 0

        # Verify descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

        # chunk_a should be the top result (exact match to query direction)
        assert results[0][0] == "chunk_a"
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)

    def test_search_returns_tuples(self, vector_store):
        """Each result must be a (path: str, score: float) tuple."""
        results = search_vectors([1.0, 0.0, 0.0], str(vector_store))
        for path, score in results:
            assert isinstance(path, str)
            assert isinstance(score, float)


class TestSearchVectorsTopK:
    def test_top_k_limits_results(self, vector_store):
        """top_k=2 must return at most 2 results even if more exist."""
        query = [1.0, 0.0, 0.0]
        results = search_vectors(query, str(vector_store), top_k=2)
        assert len(results) == 2

    def test_top_k_larger_than_total(self, vector_store):
        """top_k larger than total vectors returns all available."""
        query = [1.0, 0.0, 0.0]
        results = search_vectors(query, str(vector_store), top_k=100)
        # 3 (code) + 2 (docs) = 5 total
        assert len(results) == 5

    def test_top_k_one(self, vector_store):
        """top_k=1 returns only the single best match."""
        query = [0.0, 1.0, 0.0]  # best match: chunk_b
        results = search_vectors(query, str(vector_store), top_k=1)
        assert len(results) == 1
        assert results[0][0] == "chunk_b"


class TestSearchVectorsDomainFilter:
    def test_domain_filter_restricts_to_domain(self, vector_store):
        """Specifying domain='code' only searches that domain's embeddings."""
        query = [1.0, 1.0, 0.0]  # chunk_d in docs would score highest overall
        results = search_vectors(query, str(vector_store), domain="code")
        result_paths = {path for path, _ in results}
        # All results must be from the code domain
        assert result_paths <= {"chunk_a", "chunk_b", "chunk_c"}
        # chunk_d and chunk_e must NOT appear
        assert "chunk_d" not in result_paths
        assert "chunk_e" not in result_paths

    def test_domain_docs_only(self, vector_store):
        """domain='docs' returns only docs chunks."""
        query = [0.0, 1.0, 0.0]
        results = search_vectors(query, str(vector_store), domain="docs")
        result_paths = {path for path, _ in results}
        assert result_paths <= {"chunk_d", "chunk_e"}

    def test_nonexistent_domain_returns_empty(self, vector_store):
        """A domain that does not exist must return an empty list."""
        query = [1.0, 0.0, 0.0]
        results = search_vectors(query, str(vector_store), domain="nonexistent")
        assert results == []


class TestSearchVectorsCandidateFilter:
    def test_candidate_paths_filter(self, vector_store):
        """candidate_paths restricts scoring to only the listed paths."""
        query = [1.0, 0.0, 0.0]
        results = search_vectors(
            query,
            str(vector_store),
            candidate_paths=["chunk_b", "chunk_c"],
        )
        result_paths = {path for path, _ in results}
        # Only chunk_b and chunk_c should appear
        assert result_paths == {"chunk_b", "chunk_c"}
        # chunk_a should NOT appear even though it matches the query best
        assert "chunk_a" not in result_paths

    def test_candidate_paths_none_searches_all(self, vector_store):
        """candidate_paths=None (default) scores all vectors."""
        query = [1.0, 0.0, 0.0]
        results = search_vectors(query, str(vector_store), candidate_paths=None)
        assert len(results) == 5

    def test_candidate_paths_no_match(self, vector_store):
        """candidate_paths with no matching paths returns empty list."""
        query = [1.0, 0.0, 0.0]
        results = search_vectors(
            query,
            str(vector_store),
            candidate_paths=["does_not_exist"],
        )
        assert results == []


class TestSearchVectorsEmptyStore:
    def test_empty_store_returns_empty(self, tmp_path):
        """An empty store directory (no domains) must return an empty list."""
        store = tmp_path / ".ragdag_empty"
        store.mkdir()
        query = [1.0, 0.0, 0.0]
        results = search_vectors(query, str(store))
        assert results == []

    def test_store_with_empty_domain_dir(self, tmp_path):
        """A domain directory without embeddings.bin returns empty."""
        store = tmp_path / ".ragdag_nobin"
        store.mkdir()
        (store / "code").mkdir()
        # No embeddings.bin or manifest.tsv
        query = [1.0, 0.0, 0.0]
        results = search_vectors(query, str(store))
        assert results == []
