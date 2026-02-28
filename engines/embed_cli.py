#!/usr/bin/env python3
"""CLI bridge for embedding chunks — called from bash."""

import argparse
import sys
from pathlib import Path


def get_engine(provider: str, model: str, dimensions: int):
    """Factory for embedding engines."""
    if provider == "openai":
        from .openai_engine import OpenAIEngine
        return OpenAIEngine(model=model, dims=dimensions)
    elif provider == "local":
        from .local_engine import LocalEngine
        return LocalEngine(model=model, dims=dimensions)
    else:
        print(f"Unknown embedding provider: {provider}", file=sys.stderr)
        sys.exit(1)


def cmd_embed(args):
    """Embed chunks from a directory."""
    chunks_dir = Path(args.chunks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect chunk files
    chunk_files = sorted(chunks_dir.glob("*.txt"))
    if not chunk_files:
        return

    # Read texts
    texts = []
    paths = []
    for f in chunk_files:
        text = f.read_text(encoding="utf-8").strip()
        if text:
            texts.append(text)
            # Build relative path: doc_prefix/filename
            paths.append(f"{args.doc_prefix}/{f.name}" if args.doc_prefix else f.name)

    if not texts:
        return

    # Get engine and embed
    engine = get_engine(args.provider, args.model, args.dimensions)
    vectors = engine.embed(texts)

    # Write to embeddings.bin + manifest.tsv
    from .embeddings import write_embeddings
    write_embeddings(
        output_dir=str(output_dir),
        vectors=vectors,
        chunk_paths=paths,
        dimensions=engine.dimensions(),
        model_name_str=engine.model_name(),
        append=True,
    )

    print(f"Embedded {len(vectors)} chunks", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="ragdag embedding CLI")
    subparsers = parser.add_subparsers(dest="command")

    # embed command
    embed_parser = subparsers.add_parser("embed")
    embed_parser.add_argument("--chunks-dir", required=True)
    embed_parser.add_argument("--output-dir", required=True)
    embed_parser.add_argument("--provider", default="openai")
    embed_parser.add_argument("--model", default="text-embedding-3-small")
    embed_parser.add_argument("--dimensions", type=int, default=1536)
    embed_parser.add_argument("--doc-prefix", default="")

    args = parser.parse_args()

    if args.command == "embed":
        cmd_embed(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
