#!/usr/bin/env python3
"""Compute semantic edges between chunks based on embedding similarity."""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np


def _find_clusters(adjacency, min_size=3):
    """Find connected components from an adjacency list. Returns list of sets."""
    visited = set()
    clusters = []
    for node in adjacency:
        if node in visited:
            continue
        # BFS
        queue = [node]
        component = set()
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            component.add(n)
            for neighbor in adjacency.get(n, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(component) >= min_size:
            clusters.append(component)
    return clusters


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
    # Track new related_to edges for cluster detection
    adjacency = {}

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
                    # Build adjacency for cluster detection
                    adjacency.setdefault(path_i, []).append(path_j)
                    adjacency.setdefault(path_j, []).append(path_i)
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

    # Cluster synthesis (if enabled)
    _run_cluster_synthesis(store, edges_file, adjacency)


def _run_cluster_synthesis(store, edges_file, adjacency):
    """Generate synthesis nodes for clusters of related chunks."""
    # Read config
    config_path = store / ".config"
    if not config_path.exists():
        return

    config = {}
    section = ""
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            config[f"{section}.{key.strip()}"] = val.strip()

    if config.get("synthesis.enabled", "false") != "true":
        return
    if "clusters" not in config.get("synthesis.on_relate", "clusters"):
        return
    if config.get("llm.provider", "none") == "none":
        return

    provider = config["llm.provider"]
    model = config.get("llm.model", "gpt-4o-mini")

    clusters = _find_clusters(adjacency, min_size=3)
    if not clusters:
        return

    from .synthesis import synthesize_cluster, write_synthesis_node, chunk_file_hash

    synth_dir = store / "_synthesis"
    synth_dir.mkdir(parents=True, exist_ok=True)

    synth_count = 0
    new_edges = []

    for cluster in clusters:
        paths = sorted(cluster)
        chunks = []
        source_hashes = []
        for p in paths:
            full = store / p
            if full.exists():
                text = full.read_text(encoding="utf-8")
                # Strip frontmatter
                if text.startswith("---\n"):
                    end = text.find("\n---\n", 4)
                    if end != -1:
                        text = text[end + 5:]
                chunks.append(text)
                source_hashes.append(chunk_file_hash(full))
            else:
                chunks.append("")
                source_hashes.append("")

        if len([c for c in chunks if c.strip()]) < 3:
            continue

        try:
            synthesis_text, contradicts = synthesize_cluster(
                chunks, paths, provider, model
            )
        except Exception:
            continue

        cluster_hash = hashlib.sha256(
            ",".join(paths).encode()
        ).hexdigest()[:8]
        synth_path = synth_dir / f"_synth_{cluster_hash}.txt"
        write_synthesis_node(
            synth_path, synthesis_text, "synthesis", paths, source_hashes
        )
        synth_rel = str(synth_path.relative_to(store))

        for p in paths:
            new_edges.append(f"{synth_rel}\t{p}\tsynthesizes\trelate")

        for cp in contradicts:
            if cp in paths:
                new_edges.append(f"{synth_rel}\t{cp}\tcontradicts\trelate")

        synth_count += 1

    if new_edges:
        with open(edges_file, "a") as f:
            for edge in new_edges:
                f.write(edge + "\n")

    if synth_count:
        print(f"Created {synth_count} cluster synthesis node(s)")


if __name__ == "__main__":
    main()
