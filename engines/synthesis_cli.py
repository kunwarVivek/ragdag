#!/usr/bin/env python3
"""CLI bridge for synthesis operations — called from bash."""

import argparse
import json
import sys
from pathlib import Path


def _read_config(store_dir: str) -> dict:
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
            config[f"{section}.{key.strip()}"] = val.strip()
    return config


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use as a filename."""
    import re
    return re.sub(r"[^a-z0-9._-]", "", name.lower().replace(" ", "_"))[:50]


def cmd_ingest(args):
    """Run ingest-time synthesis: summary + entity extraction."""
    from .synthesis import (
        summarize_chunks,
        extract_entities,
        write_synthesis_node,
        chunk_file_hash,
        read_frontmatter,
        read_body,
    )

    store = Path(args.store_dir)
    config = _read_config(args.store_dir)
    doc_dir = store / args.doc_rel_path

    if not doc_dir.exists():
        print(f"Document directory not found: {doc_dir}", file=sys.stderr)
        return

    provider = config.get("llm.provider", "none")
    model = config.get("llm.model", "gpt-4o-mini")
    on_ingest = config.get("synthesis.on_ingest", "summary,entities")

    if provider == "none":
        return

    # Read raw chunks (non _ prefixed)
    chunk_files = sorted(f for f in doc_dir.glob("*.txt") if not f.name.startswith("_"))
    if not chunk_files:
        return

    chunks = []
    source_paths = []
    source_hashes = []
    for f in chunk_files:
        chunks.append(f.read_text(encoding="utf-8"))
        source_paths.append(str(f.relative_to(store)))
        source_hashes.append(chunk_file_hash(f))

    edges_file = store / ".edges"
    new_edges = []

    # Summary
    if "summary" in on_ingest:
        try:
            summary = summarize_chunks(chunks, provider, model)
            summary_path = doc_dir / "_summary.txt"
            write_synthesis_node(
                summary_path, summary, "summary", source_paths, source_hashes
            )
            summary_rel = str(summary_path.relative_to(store))
            for src in source_paths:
                new_edges.append(f"{summary_rel}\t{src}\tderived_from\tingest")
            print(f"  Created: {summary_rel}")
        except Exception as e:
            print(f"  Summary synthesis failed: {e}", file=sys.stderr)

    # Entity extraction
    if "entities" in on_ingest:
        try:
            entities = extract_entities(chunks, provider, model)
            for ent in entities:
                name = _sanitize_name(ent.get("name", "unknown"))
                etype = ent.get("type", "entity")
                desc = ent.get("description", "")
                node_name = f"_{etype}_{name}.txt"
                node_path = doc_dir / node_name

                # If entity node already exists, update it
                if node_path.exists():
                    existing_fm = read_frontmatter(node_path)
                    existing_body = read_body(node_path)
                    if existing_fm:
                        merged_sources = list(set(existing_fm.get("sources", []) + source_paths))
                        merged_hashes = list(set(existing_fm.get("source_hashes", []) + source_hashes))
                        merged_body = existing_body.strip() + "\n\n" + desc
                        write_synthesis_node(
                            node_path, merged_body, etype,
                            merged_sources, merged_hashes,
                        )
                    else:
                        write_synthesis_node(
                            node_path, desc, etype, source_paths, source_hashes
                        )
                else:
                    write_synthesis_node(
                        node_path, desc, etype, source_paths, source_hashes
                    )

                node_rel = str(node_path.relative_to(store))
                for src in source_paths:
                    new_edges.append(f"{node_rel}\t{src}\tderived_from\tingest")
                print(f"  Created: {node_rel}")
        except Exception as e:
            print(f"  Entity extraction failed: {e}", file=sys.stderr)

    # Append edges
    if new_edges:
        with open(edges_file, "a") as f:
            for edge in new_edges:
                f.write(edge + "\n")

    # Embed synthesis nodes
    if args.domain:
        _embed_synthesis_nodes(store, doc_dir, args.doc_rel_path, args.domain, config)


def _embed_synthesis_nodes(store, doc_dir, doc_prefix, domain, config):
    """Embed _ prefixed synthesis nodes."""
    try:
        from .embeddings import write_embeddings

        provider = config.get("embedding.provider", "none")
        model = config.get("embedding.model", "text-embedding-3-small")
        dims = int(config.get("embedding.dimensions", "1536"))

        if provider == "openai":
            from .openai_engine import OpenAIEngine
            engine = OpenAIEngine(model=model, dims=dims)
        elif provider == "local":
            from .local_engine import LocalEngine
            engine = LocalEngine(model=model, dims=dims)
        else:
            return

        texts, paths = [], []
        for f in sorted(doc_dir.glob("_*.txt")):
            from .synthesis import read_body
            t = read_body(f).strip()
            if t:
                texts.append(t)
                paths.append(f"{doc_prefix}/{f.name}")

        if not texts:
            return

        vectors = engine.embed(texts)

        embed_dir = store / domain if domain else store
        embed_dir.mkdir(parents=True, exist_ok=True)

        write_embeddings(
            output_dir=str(embed_dir),
            vectors=vectors,
            chunk_paths=paths,
            dimensions=engine.dimensions(),
            model_name_str=engine.model_name(),
            append=True,
        )
    except Exception:
        pass


def cmd_file_answer(args):
    """File an answer back into the store."""
    from .synthesis import write_synthesis_node, content_hash, chunk_file_hash

    store = Path(args.store_dir)
    config = _read_config(args.store_dir)

    queries_dir = store / "_queries"
    queries_dir.mkdir(parents=True, exist_ok=True)

    answer_hash = content_hash(args.question)
    answer_path = queries_dir / f"_answer_{answer_hash}.txt"

    sources = json.loads(args.sources) if args.sources else []
    source_hashes = []
    for src in sources:
        src_path = store / src
        if src_path.exists():
            source_hashes.append(chunk_file_hash(src_path))
        else:
            source_hashes.append("")

    write_synthesis_node(
        answer_path,
        f"Question: {args.question}\n\nAnswer: {args.answer}",
        "answer",
        sources,
        source_hashes,
    )

    # Add edges
    edges_file = store / ".edges"
    answer_rel = str(answer_path.relative_to(store))
    with open(edges_file, "a") as f:
        for src in sources:
            f.write(f"{answer_rel}\t{src}\tderived_from\tquery\n")

    print(f"  Filed: {answer_rel}")

    # Embed the answer
    _embed_answer_node(store, answer_path, answer_rel, config)


def _embed_answer_node(store, answer_path, answer_rel, config):
    """Embed a single answer node."""
    try:
        from .embeddings import write_embeddings
        from .synthesis import read_body

        provider = config.get("embedding.provider", "none")
        model = config.get("embedding.model", "text-embedding-3-small")
        dims = int(config.get("embedding.dimensions", "1536"))

        if provider == "openai":
            from .openai_engine import OpenAIEngine
            engine = OpenAIEngine(model=model, dims=dims)
        elif provider == "local":
            from .local_engine import LocalEngine
            engine = LocalEngine(model=model, dims=dims)
        else:
            return

        text = read_body(answer_path).strip()
        if not text:
            return

        vectors = engine.embed([text])

        embed_dir = store / "_queries"
        embed_dir.mkdir(parents=True, exist_ok=True)

        write_embeddings(
            output_dir=str(embed_dir),
            vectors=vectors,
            chunk_paths=[answer_rel],
            dimensions=engine.dimensions(),
            model_name_str=engine.model_name(),
            append=True,
        )
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="ragdag synthesis CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Ingest synthesis
    ingest_p = subparsers.add_parser("ingest")
    ingest_p.add_argument("--store-dir", required=True)
    ingest_p.add_argument("--doc-rel-path", required=True)
    ingest_p.add_argument("--domain", default="")

    # File answer
    file_p = subparsers.add_parser("file-answer")
    file_p.add_argument("--store-dir", required=True)
    file_p.add_argument("--question", required=True)
    file_p.add_argument("--answer", required=True)
    file_p.add_argument("--sources", default="[]")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "file-answer":
        cmd_file_answer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
