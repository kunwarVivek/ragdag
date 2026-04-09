"""Synthesis engine for ragdag — summarize, extract entities, synthesize clusters."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .llm import call_llm


# ---------------------------------------------------------------------------
# YAML Frontmatter utilities
# ---------------------------------------------------------------------------

def write_synthesis_node(
    path: Path,
    content: str,
    node_type: str,
    sources: List[str],
    source_hashes: List[str],
) -> None:
    """Write a synthesis node with YAML frontmatter."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    frontmatter = (
        "---\n"
        f"type: {node_type}\n"
        f"generated: {now}\n"
        f"sources: {json.dumps(sources)}\n"
        f"source_hashes: {json.dumps(source_hashes)}\n"
        "stale: false\n"
        "---\n"
    )
    path.write_text(frontmatter + content, encoding="utf-8")


def write_chunk_node(
    path: Path,
    content: str,
    meta,  # ChunkMeta — duck-typed to avoid circular import
) -> None:
    """Write a chunk file with provenance frontmatter."""
    frontmatter = (
        "---\n"
        f"type: chunk\n"
        f"source: {meta.source}\n"
        f"heading: {meta.heading}\n"
        f"position: {meta.position}\n"
        f"total: {meta.total}\n"
        f"strategy: {meta.strategy}\n"
        f"hash: {meta.hash}\n"
        "---\n"
    )
    path.write_text(frontmatter + content, encoding="utf-8")


def read_frontmatter(path: Path) -> Optional[Dict]:
    """Parse YAML frontmatter from a synthesis node. Returns None if not a synthesis node."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    if not text.startswith("---\n"):
        return None

    end = text.find("\n---\n", 4)
    if end == -1:
        return None

    fm_text = text[4:end]
    result = {}
    for line in fm_text.splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if key in ("sources", "source_hashes"):
            try:
                result[key] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                result[key] = []
        elif key == "stale":
            result[key] = val.lower() == "true"
        else:
            result[key] = val
    return result if "type" in result else None


def read_body(path: Path) -> str:
    """Read the body of a synthesis node (content after frontmatter)."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5:]
    return text


def is_synthesis_node(path: Path) -> bool:
    """Check if file has synthesis frontmatter."""
    return read_frontmatter(path) is not None


def mark_stale(path: Path) -> None:
    """Set stale: true in frontmatter."""
    text = path.read_text(encoding="utf-8")
    if "stale: false" in text:
        path.write_text(text.replace("stale: false", "stale: true", 1), encoding="utf-8")


def content_hash(text: str) -> str:
    """Short hash of content for filenames."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def chunk_file_hash(path: Path) -> str:
    """SHA256 of a chunk file for staleness tracking."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
    except Exception:
        return ""
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Synthesis functions
# ---------------------------------------------------------------------------

def generate(prompt: str, provider: str, model: str) -> str:
    """Generic LLM generation for synthesis tasks."""
    system_msg = (
        "You are a knowledge synthesis engine. Follow the instructions precisely. "
        "Be concise and factual. Do not add disclaimers or meta-commentary."
    )
    return call_llm(system_msg, prompt, provider, model)


def summarize_chunks(chunks: List[str], provider: str, model: str) -> str:
    """Generate a dense ~200-word summary from chunk texts."""
    combined = "\n\n---\n\n".join(chunks)
    prompt = (
        "Summarize the following document chunks into a single dense summary "
        "of approximately 200 words. Capture the key facts, concepts, and relationships. "
        "Write in plain prose, not bullet points.\n\n"
        f"{combined}"
    )
    return generate(prompt, provider, model)


def extract_entities(chunks: List[str], provider: str, model: str) -> List[Dict]:
    """Extract entities and concepts from chunks.

    Returns list of dicts: {"name": str, "type": "entity"|"concept", "description": str}
    """
    combined = "\n\n---\n\n".join(chunks)
    prompt = (
        "Extract the key entities (people, organizations, systems, tools) and concepts "
        "(ideas, patterns, methods, processes) from the following text. "
        "Return a JSON array where each element has:\n"
        '  {"name": "...", "type": "entity" or "concept", "description": "one-sentence description"}\n'
        "Return ONLY the JSON array, no other text.\n\n"
        f"{combined}"
    )
    raw = generate(prompt, provider, model)

    # Parse JSON from response (handle markdown code blocks)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        entities = json.loads(raw)
        if isinstance(entities, list):
            return [
                e for e in entities
                if isinstance(e, dict) and "name" in e and "type" in e
            ]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def synthesize_cluster(
    chunks: List[str], paths: List[str], provider: str, model: str
) -> Tuple[str, List[str]]:
    """Synthesize a cluster of related chunks.

    Returns (synthesis_text, list_of_contradicting_paths).
    """
    labeled = []
    for path, chunk in zip(paths, chunks):
        labeled.append(f"[Source: {path}]\n{chunk}")
    combined = "\n\n---\n\n".join(labeled)

    prompt = (
        "Analyze these related text chunks and produce a synthesis that covers:\n"
        "1. What the chunks agree on (shared facts/claims)\n"
        "2. Where they diverge (different perspectives or emphasis)\n"
        "3. Any contradictions between sources (cite the [Source: ...] paths)\n\n"
        "At the end, if there are contradictions, add a line:\n"
        "CONTRADICTS: path1, path2\n"
        "(list the source paths that contradict each other)\n\n"
        f"{combined}"
    )
    raw = generate(prompt, provider, model)

    # Extract contradiction paths
    contradicts = []
    lines = raw.strip().splitlines()
    body_lines = []
    for line in lines:
        if line.startswith("CONTRADICTS:"):
            parts = line[len("CONTRADICTS:"):].strip().split(",")
            contradicts = [p.strip() for p in parts if p.strip()]
        else:
            body_lines.append(line)

    return "\n".join(body_lines).strip(), contradicts
