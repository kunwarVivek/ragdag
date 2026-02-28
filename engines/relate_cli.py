#!/usr/bin/env python3
"""Compute semantic edges between chunks based on embedding similarity."""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="ragdag relate CLI")
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--domain", default="")
    parser.add_argument("--threshold", type=float, default=0.8)

    args = parser.parse_args()

    store = Path(args.store_dir)
    edges_file = store / ".edges"

    # Load existing edges to avoid duplicates
    existing_edges = set()
    if edges_file.exists():
        for line in edges_file.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 3 and parts[2] == "related_to":
                existing_edges.add((parts[0], parts[1]))
                existing_edges.add((parts[1], parts[0]))

    # Find embedding files
    embed_dirs = []
    if args.domain:
        d = store / args.domain
        if d.exists() and (d / "embeddings.bin").exists():
            embed_dirs.append(d)
    else:
        for d in store.iterdir():
            if d.is_dir() and (d / "embeddings.bin").exists():
                embed_dirs.append(d)

    added = 0

    for embed_dir in embed_dirs:
        bin_path = embed_dir / "embeddings.bin"
        manifest_path = embed_dir / "manifest.tsv"

        if not bin_path.exists() or not manifest_path.exists():
            continue

        from .embeddings import load_embeddings, load_manifest

        vectors, dims, count, _ = load_embeddings(str(bin_path))
        manifest = load_manifest(str(manifest_path))

        if count < 2:
            continue

        # Compute pairwise cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        normed = vectors / norms
        sim_matrix = normed @ normed.T

        # Find pairs above threshold
        new_edges = []
        for i in range(count):
            for j in range(i + 1, count):
                if sim_matrix[i, j] >= args.threshold:
                    path_i = manifest[i][0]
                    path_j = manifest[j][0]
                    if (path_i, path_j) not in existing_edges:
                        new_edges.append(
                            f"{path_i}\t{path_j}\trelated_to\tsimilarity={sim_matrix[i,j]:.4f}"
                        )
                        existing_edges.add((path_i, path_j))
                        existing_edges.add((path_j, path_i))
                        added += 1

        # Append to edges file
        if new_edges:
            with open(edges_file, "a") as f:
                for edge in new_edges:
                    f.write(edge + "\n")

    print(f"Added {added} related_to edges (threshold={args.threshold})")


if __name__ == "__main__":
    main()
