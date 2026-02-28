#!/usr/bin/env python3
"""CLI bridge for vector/hybrid search — called from bash."""

import argparse
import json
import sys
from pathlib import Path


def get_engine(store_dir: str):
    """Get embedding engine from store config."""
    config = _read_config(store_dir)
    provider = config.get("embedding.provider", "openai")
    model = config.get("embedding.model", "text-embedding-3-small")
    dims = int(config.get("embedding.dimensions", "1536"))

    if provider == "openai":
        from .openai_engine import OpenAIEngine
        return OpenAIEngine(model=model, dims=dims)
    elif provider == "local":
        from .local_engine import LocalEngine
        return LocalEngine(model=model, dims=dims)
    else:
        print(f"Unknown embedding provider: {provider}", file=sys.stderr)
        sys.exit(1)


def _read_config(store_dir: str) -> dict:
    """Parse .config INI file into flat dict."""
    config = {}
    config_path = Path(store_dir) / ".config"
    if not config_path.exists():
        return config

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
            key = key.strip()
            val = val.strip()
            if section:
                config[f"{section}.{key}"] = val
            else:
                config[key] = val
    return config


def cmd_search(args):
    """Run vector or hybrid search."""
    from .similarity import search_vectors, cosine_similarity
    from .embeddings import load_manifest

    store_dir = args.store_dir
    query = args.query
    mode = args.mode
    top_k = args.top
    domain = args.domain or ""

    # Embed the query
    engine = get_engine(store_dir)
    query_vec = engine.embed([query])[0]

    if mode == "vector":
        results = search_vectors(
            query_embedding=query_vec,
            store_dir=store_dir,
            domain=domain,
            top_k=top_k,
        )
    elif mode == "hybrid":
        # Keyword pre-filter: use grep to find candidate chunks
        kw_weight = float(args.keyword_weight)
        vec_weight = float(args.vector_weight)

        kw_results = _keyword_search(store_dir, query, domain)
        candidate_paths = [path for path, _ in kw_results] if kw_results else None

        vec_results = search_vectors(
            query_embedding=query_vec,
            store_dir=store_dir,
            domain=domain,
            top_k=top_k * 3,  # over-fetch for fusion
            candidate_paths=candidate_paths,
        )

        # Score fusion
        kw_scores = {path: score for path, score in kw_results} if kw_results else {}

        # Normalize keyword scores
        max_kw = max(kw_scores.values()) if kw_scores else 1.0
        if max_kw > 0:
            kw_scores = {p: s / max_kw for p, s in kw_scores.items()}

        # Normalize vector scores
        max_vec = max(s for _, s in vec_results) if vec_results else 1.0
        if max_vec > 0:
            vec_scores = {p: s / max_vec for p, s in vec_results}
        else:
            vec_scores = {}

        # Combine all paths
        all_paths = set(kw_scores.keys()) | set(vec_scores.keys())
        fused = []
        for path in all_paths:
            ks = kw_scores.get(path, 0.0)
            vs = vec_scores.get(path, 0.0)
            score = kw_weight * ks + vec_weight * vs
            fused.append((path, score))

        fused.sort(key=lambda x: x[1], reverse=True)
        results = fused[:top_k]
    else:
        print(f"Unknown search mode: {mode}", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.json:
        _output_json(results, store_dir)
    else:
        _output_human(results, store_dir)


def _keyword_search(store_dir: str, query: str, domain: str) -> list:
    """Simple keyword search using Python (fallback from bash)."""
    import re

    store = Path(store_dir)
    search_path = store / domain if domain else store

    words = query.lower().split()
    results = []

    for txt_file in search_path.rglob("*.txt"):
        if txt_file.name.startswith("_"):
            continue
        try:
            content = txt_file.read_text(encoding="utf-8").lower()
        except Exception:
            continue

        content_len = len(content)
        if content_len == 0:
            continue

        match_count = 0
        for word in words:
            if len(word) < 2:
                continue
            match_count += content.count(word)

        if match_count > 0:
            score = match_count / content_len
            rel_path = str(txt_file.relative_to(store))
            results.append((rel_path, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _output_human(results: list, store_dir: str):
    """Print results in human-readable format."""
    store = Path(store_dir)
    for i, (path, score) in enumerate(results, 1):
        full_path = store / path
        preview = ""
        if full_path.exists():
            lines = full_path.read_text(encoding="utf-8").splitlines()[:2]
            preview = " ".join(lines)[:120]

        print(f"{i}. [{score:.4f}] {path}")
        if preview:
            print(f"   {preview}")
            print()


def _output_json(results: list, store_dir: str):
    """Print results as JSON."""
    store = Path(store_dir)
    output = []
    for path, score in results:
        full_path = store / path
        content = ""
        if full_path.exists():
            content = full_path.read_text(encoding="utf-8")

        parts = path.split("/")
        domain_name = parts[0] if len(parts) >= 3 else ""

        output.append({
            "path": path,
            "score": round(score, 6),
            "domain": domain_name,
            "content": content,
        })

    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description="ragdag search CLI")
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=["vector", "hybrid"], default="hybrid")
    parser.add_argument("--domain", default="")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--keyword-weight", type=float, default=0.3)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    cmd_search(args)


if __name__ == "__main__":
    main()
