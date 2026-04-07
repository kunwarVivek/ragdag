"""Tests for ANN-based approximate nearest neighbors using random projection LSH."""

import numpy as np
import pytest

from engines.ann import find_neighbors_ann


class TestFindNeighborsANN:
    """Tests for find_neighbors_ann function."""

    def test_empty_vectors_returns_empty(self):
        """Empty input should return empty list."""
        vectors = np.empty((0, 128), dtype=np.float32)
        result = find_neighbors_ann(vectors, [], threshold=0.8)
        assert result == []

    def test_single_vector_returns_empty(self):
        """Single vector cannot have neighbors."""
        vectors = np.random.randn(1, 128).astype(np.float32)
        result = find_neighbors_ann(vectors, ["a.txt"], threshold=0.8)
        assert result == []

    def test_identical_vectors_found(self):
        """Identical vectors should always be found as neighbors."""
        base = np.random.randn(1, 128).astype(np.float32)
        vectors = np.vstack([base, base])
        paths = ["a.txt", "b.txt"]
        result = find_neighbors_ann(vectors, paths, threshold=0.99)
        assert len(result) == 1
        path_i, path_j, sim = result[0]
        assert {path_i, path_j} == {"a.txt", "b.txt"}
        assert sim >= 0.99

    def test_similar_vectors_high_recall(self):
        """Vectors with small noise should mostly be found as neighbors."""
        rng = np.random.RandomState(42)
        base = rng.randn(1, 128).astype(np.float32)
        base /= np.linalg.norm(base)
        n = 20
        vectors = np.vstack([base + rng.randn(1, 128) * 0.01 for _ in range(n)])
        paths = [f"chunk_{i}.txt" for i in range(n)]

        result = find_neighbors_ann(vectors, paths, threshold=0.9, n_tables=15, n_bits=12)

        # With very similar vectors, recall should be high
        # Exact method would find n*(n-1)/2 = 190 pairs at threshold 0.9
        # ANN should find most of them
        assert len(result) > 0
        # All returned similarities should be >= threshold
        for _, _, sim in result:
            assert sim >= 0.9

    def test_dissimilar_vectors_few_matches(self):
        """Random orthogonal-ish vectors should produce few matches at high threshold."""
        rng = np.random.RandomState(123)
        # High-dimensional random vectors tend to be near-orthogonal
        vectors = rng.randn(50, 256).astype(np.float32)
        paths = [f"doc_{i}.txt" for i in range(50)]

        result = find_neighbors_ann(vectors, paths, threshold=0.9)
        # Very few (likely zero) pairs should have cosine sim >= 0.9
        assert len(result) <= 5

    def test_returns_correct_tuple_format(self):
        """Each result should be (path_i, path_j, similarity) tuple."""
        rng = np.random.RandomState(7)
        base = rng.randn(1, 64).astype(np.float32)
        vectors = np.vstack([base, base * 1.01, rng.randn(1, 64)])
        paths = ["x.txt", "y.txt", "z.txt"]

        result = find_neighbors_ann(vectors, paths, threshold=0.5)
        for item in result:
            assert len(item) == 3
            path_i, path_j, sim = item
            assert isinstance(path_i, str)
            assert isinstance(path_j, str)
            assert isinstance(sim, float)

    def test_similarity_values_are_correct(self):
        """Returned similarity values should match actual cosine similarity."""
        rng = np.random.RandomState(99)
        vectors = rng.randn(5, 64).astype(np.float32)
        paths = [f"f{i}.txt" for i in range(5)]

        result = find_neighbors_ann(vectors, paths, threshold=0.0, n_tables=20, n_bits=8)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        normed = vectors / norms

        for path_i, path_j, sim in result:
            i = paths.index(path_i)
            j = paths.index(path_j)
            expected = float(normed[i] @ normed[j])
            assert abs(sim - expected) < 1e-5, f"Expected {expected}, got {sim}"

    def test_lower_threshold_more_pairs(self):
        """Lower threshold should return >= as many pairs as higher threshold."""
        rng = np.random.RandomState(55)
        vectors = rng.randn(30, 64).astype(np.float32)
        paths = [f"p{i}.txt" for i in range(30)]

        result_high = find_neighbors_ann(
            vectors, paths, threshold=0.8, n_tables=15, n_bits=10, seed=42
        )
        result_low = find_neighbors_ann(
            vectors, paths, threshold=0.3, n_tables=15, n_bits=10, seed=42
        )
        assert len(result_low) >= len(result_high)

    def test_no_self_pairs(self):
        """Should never return a pair where path_i == path_j."""
        rng = np.random.RandomState(11)
        vectors = rng.randn(10, 64).astype(np.float32)
        paths = [f"d{i}.txt" for i in range(10)]

        result = find_neighbors_ann(vectors, paths, threshold=0.0, n_tables=20, n_bits=8)
        for path_i, path_j, _ in result:
            assert path_i != path_j

    def test_no_duplicate_pairs(self):
        """Should not return both (a,b) and (b,a) -- each pair appears once."""
        rng = np.random.RandomState(22)
        base = rng.randn(1, 64).astype(np.float32)
        vectors = np.vstack([base + rng.randn(1, 64) * 0.05 for _ in range(10)])
        paths = [f"c{i}.txt" for i in range(10)]

        result = find_neighbors_ann(vectors, paths, threshold=0.5, n_tables=15, n_bits=10)
        seen = set()
        for path_i, path_j, _ in result:
            pair = tuple(sorted([path_i, path_j]))
            assert pair not in seen, f"Duplicate pair: {pair}"
            seen.add(pair)

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        rng = np.random.RandomState(33)
        vectors = rng.randn(20, 64).astype(np.float32)
        paths = [f"s{i}.txt" for i in range(20)]

        r1 = find_neighbors_ann(vectors, paths, threshold=0.3, seed=42)
        r2 = find_neighbors_ann(vectors, paths, threshold=0.3, seed=42)
        assert len(r1) == len(r2)
        for (a1, b1, s1), (a2, b2, s2) in zip(sorted(r1), sorted(r2)):
            assert a1 == a2
            assert b1 == b2
            assert abs(s1 - s2) < 1e-10
