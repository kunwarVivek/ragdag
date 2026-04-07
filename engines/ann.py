"""Approximate nearest neighbors using random projection LSH.

Replaces O(n^2) pairwise cosine similarity with O(n * avg_bucket_size)
by hashing vectors into buckets using random hyperplanes.
"""

from typing import List, Tuple

import numpy as np


def find_neighbors_ann(
    vectors: np.ndarray,
    paths: List[str],
    threshold: float = 0.8,
    n_tables: int = 10,
    n_bits: int = 16,
    seed: int = 42,
) -> List[Tuple[str, str, float]]:
    """Find approximate nearest neighbors using random projection LSH.

    Args:
        vectors: (n, dims) float32 array of embedding vectors.
        paths: Corresponding chunk paths (length n).
        threshold: Minimum cosine similarity to include a pair.
        n_tables: Number of hash tables (more = better recall).
        n_bits: Bits per hash (more = smaller buckets, fewer candidates).
        seed: Random seed for reproducibility.

    Returns:
        List of (path_i, path_j, similarity) tuples with sim >= threshold.
    """
    n = vectors.shape[0] if vectors.ndim == 2 else 0
    if n < 2:
        return []

    dims = vectors.shape[1]

    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    normed = vectors / norms

    rng = np.random.RandomState(seed)

    # Collect candidate pairs across all hash tables
    candidates = set()

    for _ in range(n_tables):
        # Random hyperplanes: (dims, n_bits)
        hyperplanes = rng.randn(dims, n_bits).astype(np.float32)

        # Project and take sign -> binary hash per vector: (n, n_bits)
        projections = normed @ hyperplanes
        hashes = (projections > 0).astype(np.uint8)

        # Convert binary hash to a hashable key per vector
        # Pack bits into bytes for efficiency
        hash_keys = [hashes[i].tobytes() for i in range(n)]

        # Group by hash bucket
        buckets = {}
        for idx, key in enumerate(hash_keys):
            buckets.setdefault(key, []).append(idx)

        # Generate candidate pairs from same bucket
        for bucket_indices in buckets.values():
            for a in range(len(bucket_indices)):
                for b in range(a + 1, len(bucket_indices)):
                    i, j = bucket_indices[a], bucket_indices[b]
                    # Canonical ordering to avoid duplicates
                    pair = (min(i, j), max(i, j))
                    candidates.add(pair)

    # Compute exact cosine similarity for candidate pairs and filter
    results = []
    for i, j in candidates:
        sim = float(normed[i] @ normed[j])
        if sim >= threshold:
            results.append((paths[i], paths[j], sim))

    return results
