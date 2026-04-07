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
    explain = getattr(args, 'explain', False)

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
        explain_data = {p: {"vector": s} for p, s in results} if explain else {}
    elif mode == "hybrid":
        from .bm25 import bm25_search
        from .rrf import reciprocal_rank_fusion

        config = _read_config(store_dir)

        # Phase 1: BM25 keyword search
        bm25_results = bm25_search(store_dir, query, domain, top_k=top_k * 3)

        # Phase 2: Vector search (over-fetch for fusion)
        candidate_paths = [path for path, _ in bm25_results] if bm25_results else None
        vec_results = search_vectors(
            query_embedding=query_vec,
            store_dir=store_dir,
            domain=domain,
            top_k=top_k * 3,
            candidate_paths=candidate_paths,
        )

        # Phase 3: RRF fusion
        results = reciprocal_rank_fusion(
            [bm25_results, vec_results],
            k=60,
            top_k=top_k,
        )

        # Build explain data before reranking
        explain_data = {}
        if explain:
            bm25_map = {p: s for p, s in bm25_results}
            vec_map = {p: s for p, s in vec_results}
            for path, rrf_score in results:
                explain_data[path] = {
                    "bm25": round(bm25_map.get(path, 0.0), 6),
                    "vector": round(vec_map.get(path, 0.0), 6),
                    "rrf": round(rrf_score, 6),
                }

        # Phase 4: Optional reranking
        rerank_enabled = config.get("search.rerank", "false") == "true" or getattr(args, 'rerank', False)
        if rerank_enabled and results:
            try:
                from .reranker import rerank
                store = Path(store_dir)
                candidates = []
                for path, score in results:
                    full_path = store / path
                    content = full_path.read_text(encoding="utf-8") if full_path.exists() else ""
                    candidates.append((path, score, content))
                rerank_model = config.get(
                    "search.rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                reranked = rerank(query, candidates, model_name=rerank_model, top_k=top_k)
                if explain:
                    for path, blended_score in reranked:
                        if path in explain_data:
                            explain_data[path]["reranker"] = round(blended_score, 6)
                results = reranked
            except Exception:
                pass  # graceful degradation
    else:
        print(f"Unknown search mode: {mode}", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.json:
        _output_json(results, store_dir, explain_data if explain else None)
    else:
        _output_human(results, store_dir, explain_data if explain else None)


def _output_human(results: list, store_dir: str, explain_data: dict = None):
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
        if explain_data and path in explain_data:
            parts = [f"{k}={v}" for k, v in explain_data[path].items()]
            print(f"   explain: {', '.join(parts)}")
        print()


def _output_json(results: list, store_dir: str, explain_data: dict = None):
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

        entry = {
            "path": path,
            "score": round(score, 6),
            "domain": domain_name,
            "content": content,
        }
        if explain_data and path in explain_data:
            entry["explain"] = explain_data[path]

        output.append(entry)

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
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--rerank", action="store_true")

    args = parser.parse_args()
    cmd_search(args)


if __name__ == "__main__":
    main()
