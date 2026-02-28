"""Tests for _chunk_heading, _chunk_paragraph, and _chunk_function strategies."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from ragdag.core import RagDag


def _make_dag(tmp_path):
    """Create a minimal RagDag for testing chunking methods."""
    store = tmp_path / ".ragdag"
    store.mkdir(exist_ok=True)
    (store / ".config").write_text(
        "[general]\nchunk_size = 1000\nchunk_overlap = 0\n"
    )
    (store / ".edges").write_text("")
    (store / ".processed").write_text("")
    (store / ".domain-rules").write_text("")
    return RagDag(str(tmp_path))


# ======================================================================
# _chunk_heading tests
# ======================================================================


class TestChunkHeading:
    """Tests for _chunk_heading -- split on markdown # headers."""

    def test_heading_splits_on_hash_headers(self, tmp_path):
        """Text with ## headers produces multiple chunks."""
        dag = _make_dag(tmp_path)
        text = (
            "## Introduction\n"
            "This is the intro.\n"
            "## Methods\n"
            "This is the methods section.\n"
            "## Results\n"
            "Here are results."
        )
        chunks = dag._chunk_heading(text, chunk_size=5000, overlap=0)
        assert len(chunks) == 3

    def test_heading_preserves_header_in_chunk(self, tmp_path):
        """Each chunk starts with its header text."""
        dag = _make_dag(tmp_path)
        text = (
            "## Alpha\n"
            "Alpha content.\n"
            "## Beta\n"
            "Beta content."
        )
        chunks = dag._chunk_heading(text, chunk_size=5000, overlap=0)
        assert len(chunks) == 2
        assert chunks[0].startswith("## Alpha")
        assert chunks[1].startswith("## Beta")

    def test_heading_respects_chunk_size(self, tmp_path):
        """A long section under one header gets split when it exceeds chunk_size."""
        dag = _make_dag(tmp_path)
        # One header followed by many lines that exceed chunk_size
        long_body = "\n".join(
            [f"Line {i} with some filler content here." for i in range(100)]
        )
        text = f"## Big Section\n{long_body}"
        chunks = dag._chunk_heading(text, chunk_size=200, overlap=0)
        # Should produce multiple chunks because body exceeds 200 chars
        assert len(chunks) > 1

    def test_heading_overlap_carries_tail(self, tmp_path):
        """Overlap produces shared content between adjacent chunks."""
        dag = _make_dag(tmp_path)
        text = (
            "## Section One\n"
            "Content of section one with enough text to verify overlap.\n"
            "## Section Two\n"
            "Content of section two."
        )
        chunks = dag._chunk_heading(text, chunk_size=5000, overlap=20)
        assert len(chunks) == 2
        # The second chunk should contain overlap from the tail of the first chunk
        # The overlap is taken as the last N chars of the previous chunk text
        tail_of_first = chunks[0][-20:]
        # Second chunk should start with that tail (as a separate buffer entry)
        assert tail_of_first in chunks[1]

    def test_heading_no_headers_single_chunk(self, tmp_path):
        """Text without any # headers stays as one chunk (within chunk_size)."""
        dag = _make_dag(tmp_path)
        text = "Just plain text without any headers.\nAnother line."
        chunks = dag._chunk_heading(text, chunk_size=5000, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_heading_empty_text(self, tmp_path):
        """Empty text returns an empty list."""
        dag = _make_dag(tmp_path)
        chunks = dag._chunk_heading("", chunk_size=1000, overlap=0)
        assert chunks == []


# ======================================================================
# _chunk_paragraph tests
# ======================================================================


class TestChunkParagraph:
    """Tests for _chunk_paragraph -- split on blank lines."""

    def test_paragraph_splits_on_blank_lines(self, tmp_path):
        """Text with blank lines produces separate chunks when chunk_size is small."""
        dag = _make_dag(tmp_path)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = dag._chunk_paragraph(text, chunk_size=5000, overlap=0)
        assert len(chunks) == 1  # All fit within chunk_size, so combined
        # With tiny chunk_size, they should split
        chunks = dag._chunk_paragraph(text, chunk_size=20, overlap=0)
        assert len(chunks) == 3

    def test_paragraph_combines_short_paragraphs(self, tmp_path):
        """Small paragraphs merge up to chunk_size."""
        dag = _make_dag(tmp_path)
        text = "Short A.\n\nShort B.\n\nShort C."
        chunks = dag._chunk_paragraph(text, chunk_size=5000, overlap=0)
        # All three paragraphs fit within 5000 chars, so they should be one chunk
        assert len(chunks) == 1
        assert "Short A." in chunks[0]
        assert "Short B." in chunks[0]
        assert "Short C." in chunks[0]

    def test_paragraph_respects_chunk_size(self, tmp_path):
        """Large paragraphs trigger splits when accumulated text exceeds chunk_size."""
        dag = _make_dag(tmp_path)
        p1 = "A" * 50
        p2 = "B" * 50
        p3 = "C" * 50
        text = f"{p1}\n\n{p2}\n\n{p3}"
        # chunk_size=60 means p1 (50) fits, but p1+p2 (102 with separator) exceeds
        chunks = dag._chunk_paragraph(text, chunk_size=60, overlap=0)
        assert len(chunks) == 3
        assert chunks[0] == p1
        assert chunks[1] == p2
        assert chunks[2] == p3

    def test_paragraph_overlap_carries_tail(self, tmp_path):
        """Overlap content is shared between paragraph chunks."""
        dag = _make_dag(tmp_path)
        p1 = "First paragraph with some content."
        p2 = "Second paragraph here."
        p3 = "Third paragraph text."
        text = f"{p1}\n\n{p2}\n\n{p3}"
        # Force each paragraph to be its own chunk
        chunks = dag._chunk_paragraph(text, chunk_size=40, overlap=10)
        assert len(chunks) >= 2
        # With overlap, chunks after the first should contain tail of previous
        tail = chunks[0][-10:]
        assert tail in chunks[1]

    def test_paragraph_empty_text(self, tmp_path):
        """Empty text returns an empty list."""
        dag = _make_dag(tmp_path)
        chunks = dag._chunk_paragraph("", chunk_size=1000, overlap=0)
        assert chunks == []

    def test_paragraph_whitespace_only_skipped(self, tmp_path):
        """Paragraphs that are only whitespace are ignored."""
        dag = _make_dag(tmp_path)
        text = "Real content.\n\n   \n\n  \n\nMore content."
        chunks = dag._chunk_paragraph(text, chunk_size=5000, overlap=0)
        # Whitespace-only paragraphs should be skipped
        for chunk in chunks:
            assert chunk.strip() != ""


# ======================================================================
# _chunk_function tests
# ======================================================================


class TestChunkFunction:
    """Tests for _chunk_function -- split on function/class boundaries."""

    def test_function_splits_on_python_def(self, tmp_path):
        """Python code splits on `def ` boundaries."""
        dag = _make_dag(tmp_path)
        text = (
            "def foo():\n"
            "    return 1\n"
            "\n"
            "def bar():\n"
            "    return 2\n"
            "\n"
            "def baz():\n"
            "    return 3"
        )
        chunks = dag._chunk_function(text, chunk_size=5000, overlap=0)
        assert len(chunks) == 3
        assert "def foo" in chunks[0]
        assert "def bar" in chunks[1]
        assert "def baz" in chunks[2]

    def test_function_splits_on_class(self, tmp_path):
        """Splits on `class ` boundaries."""
        dag = _make_dag(tmp_path)
        text = (
            "class Foo:\n"
            "    pass\n"
            "\n"
            "class Bar:\n"
            "    pass"
        )
        chunks = dag._chunk_function(text, chunk_size=5000, overlap=0)
        assert len(chunks) == 2
        assert "class Foo" in chunks[0]
        assert "class Bar" in chunks[1]

    def test_function_splits_on_js_function(self, tmp_path):
        """Splits on `function ` keyword."""
        dag = _make_dag(tmp_path)
        text = (
            "function hello() {\n"
            "  console.log('hello');\n"
            "}\n"
            "\n"
            "function world() {\n"
            "  console.log('world');\n"
            "}"
        )
        chunks = dag._chunk_function(text, chunk_size=5000, overlap=0)
        assert len(chunks) == 2
        assert "function hello" in chunks[0]
        assert "function world" in chunks[1]

    def test_function_splits_on_const_let_var(self, tmp_path):
        """Splits on JS const/let/var declarations."""
        dag = _make_dag(tmp_path)
        text = (
            "const add = (a, b) => a + b;\n"
            "\n"
            "let multiply = (a, b) => a * b;\n"
            "\n"
            "var divide = (a, b) => a / b;"
        )
        chunks = dag._chunk_function(text, chunk_size=5000, overlap=0)
        assert len(chunks) == 3
        assert "const add" in chunks[0]
        assert "let multiply" in chunks[1]
        assert "var divide" in chunks[2]

    def test_function_splits_on_rust_fn(self, tmp_path):
        """Splits on `fn ` and `pub fn ` for Rust code."""
        dag = _make_dag(tmp_path)
        text = (
            "fn main() {\n"
            "    println!(\"hello\");\n"
            "}\n"
            "\n"
            "pub fn helper() {\n"
            "    println!(\"help\");\n"
            "}"
        )
        chunks = dag._chunk_function(text, chunk_size=5000, overlap=0)
        assert len(chunks) == 2
        assert "fn main" in chunks[0]
        assert "pub fn helper" in chunks[1]

    def test_function_splits_on_export(self, tmp_path):
        """Splits on `export ` keyword."""
        dag = _make_dag(tmp_path)
        text = (
            "export default function App() {\n"
            "  return null;\n"
            "}\n"
            "\n"
            "export function Other() {\n"
            "  return null;\n"
            "}"
        )
        chunks = dag._chunk_function(text, chunk_size=5000, overlap=0)
        assert len(chunks) == 2
        assert "export default" in chunks[0]
        assert "export function" in chunks[1]

    def test_function_splits_on_go_func(self, tmp_path):
        """Splits on `func ` keyword for Go code."""
        dag = _make_dag(tmp_path)
        text = (
            "func main() {\n"
            "    fmt.Println(\"hello\")\n"
            "}\n"
            "\n"
            "func helper() {\n"
            "    fmt.Println(\"help\")\n"
            "}"
        )
        chunks = dag._chunk_function(text, chunk_size=5000, overlap=0)
        assert len(chunks) == 2
        assert "func main" in chunks[0]
        assert "func helper" in chunks[1]

    def test_function_respects_chunk_size(self, tmp_path):
        """Long function body gets split when it exceeds chunk_size."""
        dag = _make_dag(tmp_path)
        long_body = "\n".join([f"    x = {i}" for i in range(100)])
        text = f"def big_function():\n{long_body}"
        chunks = dag._chunk_function(text, chunk_size=200, overlap=0)
        assert len(chunks) > 1

    def test_function_empty_text(self, tmp_path):
        """Empty text returns an empty list."""
        dag = _make_dag(tmp_path)
        chunks = dag._chunk_function("", chunk_size=1000, overlap=0)
        assert chunks == []

    def test_function_no_boundaries_single_chunk(self, tmp_path):
        """Text without any function boundaries stays as one chunk."""
        dag = _make_dag(tmp_path)
        text = "# Just a comment\nx = 1\ny = 2\nprint(x + y)"
        chunks = dag._chunk_function(text, chunk_size=5000, overlap=0)
        assert len(chunks) == 1


# ======================================================================
# Auto-strategy selection tests (via add())
# ======================================================================


class TestAutoStrategy:
    """Tests for automatic strategy selection via the add() method."""

    def test_auto_strategy_markdown_uses_heading(self, tmp_path):
        """Adding a .md file uses heading strategy (splits on #)."""
        dag = _make_dag(tmp_path)
        md_file = tmp_path / "doc.md"
        md_file.write_text(
            "## Section A\n"
            "Content A.\n"
            "## Section B\n"
            "Content B."
        )
        result = dag.add(str(md_file), embed=False)
        assert result["chunks"] == 2
        # Verify stored chunks contain the headers
        store_dir = tmp_path / ".ragdag" / "doc"
        chunk1 = (store_dir / "01.txt").read_text()
        chunk2 = (store_dir / "02.txt").read_text()
        assert chunk1.startswith("## Section A")
        assert chunk2.startswith("## Section B")

    def test_auto_strategy_code_uses_function(self, tmp_path):
        """Adding a .py file uses function strategy (splits on def/class)."""
        dag = _make_dag(tmp_path)
        py_file = tmp_path / "module.py"
        py_file.write_text(
            "def alpha():\n"
            "    return 1\n"
            "\n"
            "def beta():\n"
            "    return 2\n"
        )
        result = dag.add(str(py_file), embed=False)
        assert result["chunks"] == 2
        store_dir = tmp_path / ".ragdag" / "module"
        chunk1 = (store_dir / "01.txt").read_text()
        chunk2 = (store_dir / "02.txt").read_text()
        assert "def alpha" in chunk1
        assert "def beta" in chunk2


# ======================================================================
# Chunk numbering / storage format tests
# ======================================================================


class TestChunkStorageFormat:
    """Tests for chunk file naming conventions in the store."""

    def test_chunk_numbering_format_in_store(self, tmp_path):
        """After dag.add(), stored chunk files follow the 01.txt, 02.txt format."""
        dag = _make_dag(tmp_path)
        doc = tmp_path / "numbering.md"
        doc.write_text(
            "## First\n"
            "First section content.\n"
            "## Second\n"
            "Second section content.\n"
            "## Third\n"
            "Third section content."
        )
        result = dag.add(str(doc), embed=False)
        assert result["chunks"] == 3

        store_dir = tmp_path / ".ragdag" / "numbering"
        assert store_dir.exists(), "Domain directory should exist after add()"

        chunk_files = sorted(store_dir.glob("*.txt"))
        filenames = [f.name for f in chunk_files]

        assert len(filenames) == 3
        assert filenames[0] == "01.txt"
        assert filenames[1] == "02.txt"
        assert filenames[2] == "03.txt"
