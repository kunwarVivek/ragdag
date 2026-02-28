#!/usr/bin/env python3
"""CLI bridge for RAG question answering — called from bash."""

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


def main():
    parser = argparse.ArgumentParser(description="ragdag ask CLI")
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--context-file", required=True,
                        help="File containing assembled context")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    config = _read_config(args.store_dir)
    provider = config.get("llm.provider", "none")
    model = config.get("llm.model", "gpt-4o-mini")

    context = Path(args.context_file).read_text(encoding="utf-8")

    if args.no_llm or provider == "none":
        # Return context without LLM
        if args.json:
            print(json.dumps({
                "answer": None,
                "context": context,
                "sources": _extract_sources(context),
            }))
        else:
            print("=== Context (no LLM configured) ===\n")
            print(context)
        return

    # Load custom prompt template if available
    prompt_template = None
    prompt_file = Path(args.store_dir) / "prompt.txt"
    if prompt_file.exists():
        prompt_template = prompt_file.read_text(encoding="utf-8")

    from .llm import get_answer

    try:
        answer = get_answer(
            question=args.question,
            context=context,
            provider=provider,
            model=model,
            prompt_template=prompt_template,
        )
    except Exception as e:
        print(f"LLM error: {e}", file=sys.stderr)
        if args.json:
            print(json.dumps({"error": str(e), "context": context}))
        else:
            print("=== Context (LLM failed) ===\n")
            print(context)
        return

    if args.json:
        print(json.dumps({
            "answer": answer,
            "sources": _extract_sources(context),
        }))
    else:
        print(answer)


def _extract_sources(context: str) -> list:
    """Extract source paths from context."""
    sources = []
    for line in context.splitlines():
        if line.startswith("--- Source:"):
            path = line.replace("--- Source:", "").strip()
            # Remove score part
            if " (score:" in path:
                path = path.split(" (score:")[0].strip()
            sources.append(path)
    return sources


if __name__ == "__main__":
    main()
