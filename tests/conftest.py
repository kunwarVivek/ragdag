"""Shared fixtures for ragdag tests."""

import os
import sys
from pathlib import Path

import pytest

# Add project root and sdk to path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "sdk"))


@pytest.fixture
def tmp_store(tmp_path):
    """Create a temporary ragdag store with default config."""
    store = tmp_path / ".ragdag"
    store.mkdir()

    config = store / ".config"
    config.write_text(
        "[general]\n"
        "chunk_strategy = heading\n"
        "chunk_size = 1000\n"
        "chunk_overlap = 100\n\n"
        "[embedding]\n"
        "provider = none\n"
        "model = text-embedding-3-small\n"
        "dimensions = 1536\n\n"
        "[llm]\n"
        "provider = none\n"
        "model = gpt-4o-mini\n"
        "max_context = 8000\n\n"
        "[search]\n"
        "default_mode = hybrid\n"
        "top_k = 10\n"
        "keyword_weight = 0.3\n"
        "vector_weight = 0.7\n\n"
        "[edges]\n"
        "auto_relate = false\n"
        "relate_threshold = 0.8\n"
        "record_queries = false\n"
    )

    for f in [".edges", ".processed", ".domain-rules"]:
        (store / f).write_text(f"# {f}\n")

    return tmp_path


@pytest.fixture
def dag(tmp_store):
    """Create a RagDag instance on a temp store."""
    import ragdag
    return ragdag.open(str(tmp_store))
