"""Cosine similarity computation over packed embeddings."""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .embeddings import load_embeddings_mmap, load_manifest


def cosine_similarity(query_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and matrix of vectors."""
    # Normalize
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    vectors_norm = vectors / norms
    return vectors_norm @ query_norm


def search_vectors(
    query_embedding: List[float],
    store_dir: str,
    domain: str = "",
    top_k: int = 10,
    candidate_paths: List[str] = None,
) -> List[Tuple[str, float]]:
    """Search for similar chunks using vector similarity.

    Args:
        query_embedding: Query vector
        store_dir: Path to .ragdag store
        domain: Optional domain filter
        top_k: Number of results
        candidate_paths: Optional pre-filtered paths (for hybrid search)

    Returns:
        List of (chunk_path, similarity_score) tuples
    """
    store = Path(store_dir)
    query_vec = np.array(query_embedding, dtype=np.float32)

    # Collect all embedding files to search
    embed_dirs = []
    if domain:
        domain_dir = store / domain
        if domain_dir.exists():
            embed_dirs.append(domain_dir)
    else:
        # Search all domains (subdirectories with embeddings.bin)
        for d in store.iterdir():
            if d.is_dir() and (d / "embeddings.bin").exists():
                embed_dirs.append(d)

    all_results = []

    for embed_dir in embed_dirs:
        bin_path = embed_dir / "embeddings.bin"
        manifest_path = embed_dir / "manifest.tsv"

        if not bin_path.exists() or not manifest_path.exists():
            continue

        vectors, dims, count, _ = load_embeddings_mmap(str(bin_path))
        manifest = load_manifest(str(manifest_path))

        if count == 0:
            continue

        # If we have candidate paths, filter to only those
        if candidate_paths is not None:
            candidate_set = set(candidate_paths)
            indices = [i for i, (p, _, _, _) in enumerate(manifest) if p in candidate_set]
            if not indices:
                continue
            idx_array = np.array(indices)
            filtered_vectors = vectors[idx_array]
            filtered_manifest = [manifest[i] for i in indices]
        else:
            filtered_vectors = vectors
            filtered_manifest = manifest

        # Compute similarity
        similarities = cosine_similarity(query_vec, filtered_vectors)

        for i, (path, _, _, _) in enumerate(filtered_manifest):
            all_results.append((path, float(similarities[i])))

    # Sort by score descending
    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results[:top_k]
