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


class TestIngestProvenance:
    def test_add_writes_frontmatter_to_chunks(self, tmp_path):
        """After add(), each chunk .txt has provenance frontmatter."""
        import ragdag
        from engines.synthesis import read_frontmatter, read_body

        dag = ragdag.init(str(tmp_path))
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Introduction\nHello world.\n\n# Details\nMore info here.")
        dag.add(str(md_file))

        store = tmp_path / ".ragdag"
        chunk_files = sorted(
            f for f in store.rglob("*.txt")
            if f.name[0].isdigit() and not f.name.startswith("_")
        )
        assert len(chunk_files) >= 1

        for cf in chunk_files:
            fm = read_frontmatter(cf)
            assert fm is not None, f"No frontmatter in {cf.name}"
            assert fm["type"] == "chunk"
            assert fm["source"] == str(md_file.resolve())
            assert fm["strategy"] == "heading"
            assert "position" in fm
            assert "total" in fm
            assert "hash" in fm

    def test_add_heading_breadcrumb(self, tmp_path):
        """Heading strategy captures the heading text in frontmatter."""
        import ragdag
        from engines.synthesis import read_frontmatter

        dag = ragdag.init(str(tmp_path))
        md_file = tmp_path / "doc.md"
        md_file.write_text("# Chapter 1\nContent under chapter 1.\n\n# Chapter 2\nContent under chapter 2.")
        dag.add(str(md_file))

        store = tmp_path / ".ragdag"
        chunk_files = sorted(
            f for f in store.rglob("*.txt")
            if f.name[0].isdigit() and not f.name.startswith("_")
        )
        assert len(chunk_files) >= 2
        fm1 = read_frontmatter(chunk_files[0])
        assert "Chapter 1" in fm1.get("heading", "")

    def test_provenance_position_and_total(self, tmp_path):
        """position and total reflect chunk ordering."""
        import ragdag
        from engines.synthesis import read_frontmatter

        dag = ragdag.init(str(tmp_path))
        md_file = tmp_path / "multi.md"
        md_file.write_text("# A\nFirst\n\n# B\nSecond\n\n# C\nThird")
        dag.add(str(md_file))

        store = tmp_path / ".ragdag"
        chunk_files = sorted(
            f for f in store.rglob("*.txt")
            if f.name[0].isdigit() and not f.name.startswith("_")
        )
        assert len(chunk_files) == 3
        fm1 = read_frontmatter(chunk_files[0])
        fm3 = read_frontmatter(chunk_files[2])
        assert fm1["position"] == "1"
        assert fm1["total"] == "3"
        assert fm3["position"] == "3"
        assert fm3["total"] == "3"

    def test_bm25_strips_frontmatter(self, tmp_path):
        """BM25 search should strip frontmatter and only search body content."""
        import ragdag

        dag = ragdag.init(str(tmp_path))
        md_file = tmp_path / "doc.md"
        md_file.write_text("# Deployment\nUse docker-compose to deploy the application.")
        dag.add(str(md_file))

        results = dag.search("deploy", mode="keyword")
        assert len(results) >= 1
        for r in results:
            assert "---\ntype: chunk" not in r.content
