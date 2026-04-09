"""Tests for chunk provenance metadata."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from ragdag.core import ChunkMeta


class TestChunkMeta:
    def test_dataclass_fields(self):
        meta = ChunkMeta(
            source="/path/to/file.md",
            heading="## Deploy > ### Docker",
            position=3,
            total=7,
            strategy="heading",
            hash="a1b2c3d4",
        )
        assert meta.source == "/path/to/file.md"
        assert meta.heading == "## Deploy > ### Docker"
        assert meta.position == 3
        assert meta.total == 7
        assert meta.strategy == "heading"
        assert meta.hash == "a1b2c3d4"

    def test_defaults(self):
        meta = ChunkMeta(source="/f.md", position=1, total=1, strategy="heading", hash="abc")
        assert meta.heading == ""


class TestChunkFrontmatter:
    def test_write_and_read_chunk_frontmatter(self, tmp_path):
        from engines.synthesis import write_chunk_node, read_frontmatter, read_body

        chunk_file = tmp_path / "01.txt"
        meta = ChunkMeta(
            source="/src/readme.md",
            heading="## Overview",
            position=1,
            total=3,
            strategy="heading",
            hash="deadbeef",
        )
        write_chunk_node(chunk_file, "This is the chunk content.", meta)

        fm = read_frontmatter(chunk_file)
        assert fm is not None
        assert fm["type"] == "chunk"
        assert fm["source"] == "/src/readme.md"
        assert fm["heading"] == "## Overview"
        assert fm["position"] == "1"
        assert fm["total"] == "3"
        assert fm["strategy"] == "heading"
        assert fm["hash"] == "deadbeef"

        body = read_body(chunk_file)
        assert body == "This is the chunk content."

    def test_read_body_no_frontmatter(self, tmp_path):
        from engines.synthesis import read_body
        chunk_file = tmp_path / "01.txt"
        chunk_file.write_text("plain chunk text")
        assert read_body(chunk_file) == "plain chunk text"

    def test_read_frontmatter_none_for_bare_file(self, tmp_path):
        from engines.synthesis import read_frontmatter
        chunk_file = tmp_path / "01.txt"
        chunk_file.write_text("no frontmatter here")
        assert read_frontmatter(chunk_file) is None
