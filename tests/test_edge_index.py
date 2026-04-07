"""Tests for per-node edge index — O(degree) graph lookups."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from engines.edge_index import build_edge_index, load_edge_index, lookup_edges, append_edge


SAMPLE_EDGES = """\
# Edge file
a.txt\tb.txt\trelated_to\tsim=0.9
c.txt\ta.txt\treferences\t
b.txt\td.txt\tchunked_from\tparent=doc1
"""


class TestBuildEdgeIndex:
    """build_edge_index reads .edges and creates _edge_index.json."""

    def test_creates_index_file(self, tmp_path):
        """build_edge_index should create _edge_index.json in store_dir."""
        (tmp_path / ".edges").write_text(SAMPLE_EDGES)
        build_edge_index(str(tmp_path))
        assert (tmp_path / "_edge_index.json").exists()

    def test_index_has_correct_structure(self, tmp_path):
        """Index should have entries for all nodes (sources and targets)."""
        (tmp_path / ".edges").write_text(SAMPLE_EDGES)
        build_edge_index(str(tmp_path))
        idx = json.loads((tmp_path / "_edge_index.json").read_text())
        # All unique nodes should be present
        assert "a.txt" in idx
        assert "b.txt" in idx
        assert "c.txt" in idx
        assert "d.txt" in idx

    def test_bidirectional_entries(self, tmp_path):
        """Each edge creates outgoing from source and incoming to target."""
        edges = "a.txt\tb.txt\trelated_to\tsim=0.9\n"
        (tmp_path / ".edges").write_text(edges)
        build_edge_index(str(tmp_path))
        idx = json.loads((tmp_path / "_edge_index.json").read_text())

        # a.txt has outgoing to b.txt
        a_edges = idx["a.txt"]
        assert any(
            e["direction"] == "outgoing" and e["node"] == "b.txt" and e["edge_type"] == "related_to"
            for e in a_edges
        )
        # b.txt has incoming from a.txt
        b_edges = idx["b.txt"]
        assert any(
            e["direction"] == "incoming" and e["node"] == "a.txt" and e["edge_type"] == "related_to"
            for e in b_edges
        )

    def test_skips_comments_and_blank_lines(self, tmp_path):
        """Comments and blank lines in .edges should be ignored."""
        edges = "# comment\n\na.txt\tb.txt\trelated_to\tsim=0.9\n"
        (tmp_path / ".edges").write_text(edges)
        build_edge_index(str(tmp_path))
        idx = json.loads((tmp_path / "_edge_index.json").read_text())
        assert len(idx) == 2  # only a.txt and b.txt

    def test_metadata_preserved(self, tmp_path):
        """Metadata field should be preserved in index entries."""
        edges = "a.txt\tb.txt\trelated_to\tsim=0.9\n"
        (tmp_path / ".edges").write_text(edges)
        build_edge_index(str(tmp_path))
        idx = json.loads((tmp_path / "_edge_index.json").read_text())
        outgoing = [e for e in idx["a.txt"] if e["direction"] == "outgoing"]
        assert outgoing[0]["metadata"] == "sim=0.9"

    def test_empty_metadata(self, tmp_path):
        """Missing metadata defaults to empty string."""
        edges = "a.txt\tb.txt\trelated_to\n"
        (tmp_path / ".edges").write_text(edges)
        build_edge_index(str(tmp_path))
        idx = json.loads((tmp_path / "_edge_index.json").read_text())
        outgoing = [e for e in idx["a.txt"] if e["direction"] == "outgoing"]
        assert outgoing[0]["metadata"] == ""

    def test_no_edges_file(self, tmp_path):
        """build_edge_index with no .edges file should create empty index."""
        build_edge_index(str(tmp_path))
        idx = json.loads((tmp_path / "_edge_index.json").read_text())
        assert idx == {}


class TestLoadEdgeIndex:
    """load_edge_index returns dict or None."""

    def test_returns_dict_when_exists(self, tmp_path):
        """Should return parsed dict when _edge_index.json exists."""
        (tmp_path / "_edge_index.json").write_text('{"a.txt": []}')
        result = load_edge_index(str(tmp_path))
        assert result == {"a.txt": []}

    def test_returns_none_when_missing(self, tmp_path):
        """Should return None when no index file exists."""
        result = load_edge_index(str(tmp_path))
        assert result is None


class TestLookupEdges:
    """lookup_edges returns edges for a node or None if no index."""

    def test_returns_edges_for_node(self, tmp_path):
        """Should return list of edge dicts for existing node."""
        (tmp_path / ".edges").write_text(SAMPLE_EDGES)
        build_edge_index(str(tmp_path))
        result = lookup_edges(str(tmp_path), "a.txt")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_returns_empty_for_nonexistent_node(self, tmp_path):
        """Should return empty list for node not in index."""
        (tmp_path / ".edges").write_text(SAMPLE_EDGES)
        build_edge_index(str(tmp_path))
        result = lookup_edges(str(tmp_path), "nonexistent.txt")
        assert result == []

    def test_returns_none_without_index(self, tmp_path):
        """Should return None when no _edge_index.json exists."""
        result = lookup_edges(str(tmp_path), "a.txt")
        assert result is None

    def test_correct_outgoing_and_incoming(self, tmp_path):
        """Node a.txt should have both outgoing and incoming edges from sample."""
        (tmp_path / ".edges").write_text(SAMPLE_EDGES)
        build_edge_index(str(tmp_path))
        result = lookup_edges(str(tmp_path), "a.txt")
        directions = {e["direction"] for e in result}
        assert "outgoing" in directions  # a.txt -> b.txt
        assert "incoming" in directions  # c.txt -> a.txt


class TestAppendEdge:
    """append_edge updates index incrementally."""

    def test_adds_to_existing_index(self, tmp_path):
        """Should add new edge entries to existing index."""
        (tmp_path / ".edges").write_text(SAMPLE_EDGES)
        build_edge_index(str(tmp_path))

        append_edge(str(tmp_path), "x.txt", "y.txt", "related_to", "sim=0.7")

        result = lookup_edges(str(tmp_path), "x.txt")
        assert any(e["node"] == "y.txt" and e["direction"] == "outgoing" for e in result)

        result = lookup_edges(str(tmp_path), "y.txt")
        assert any(e["node"] == "x.txt" and e["direction"] == "incoming" for e in result)

    def test_creates_index_if_missing(self, tmp_path):
        """Should create index file if it doesn't exist."""
        append_edge(str(tmp_path), "x.txt", "y.txt", "related_to", "sim=0.7")
        assert (tmp_path / "_edge_index.json").exists()
        result = lookup_edges(str(tmp_path), "x.txt")
        assert len(result) == 1

    def test_preserves_existing_edges(self, tmp_path):
        """Appending should not remove existing edges."""
        (tmp_path / ".edges").write_text(SAMPLE_EDGES)
        build_edge_index(str(tmp_path))

        original_a = lookup_edges(str(tmp_path), "a.txt")
        original_count = len(original_a)

        append_edge(str(tmp_path), "a.txt", "z.txt", "references", "")

        updated_a = lookup_edges(str(tmp_path), "a.txt")
        assert len(updated_a) == original_count + 1

    def test_does_not_write_edges_file(self, tmp_path):
        """append_edge must NOT modify .edges (caller's responsibility)."""
        (tmp_path / ".edges").write_text(SAMPLE_EDGES)
        original = (tmp_path / ".edges").read_text()
        build_edge_index(str(tmp_path))

        append_edge(str(tmp_path), "x.txt", "y.txt", "related_to", "")

        assert (tmp_path / ".edges").read_text() == original


class TestRebuildConsistency:
    """Rebuilding from .edges produces correct bidirectional index."""

    def test_rebuild_matches_manual(self, tmp_path):
        """Rebuilding index should produce same result as building from scratch."""
        edges = "a.txt\tb.txt\trelated_to\tsim=0.9\nc.txt\ta.txt\treferences\t\n"
        (tmp_path / ".edges").write_text(edges)

        # Build once
        build_edge_index(str(tmp_path))
        first = json.loads((tmp_path / "_edge_index.json").read_text())

        # Corrupt and rebuild
        (tmp_path / "_edge_index.json").write_text("{}")
        build_edge_index(str(tmp_path))
        second = json.loads((tmp_path / "_edge_index.json").read_text())

        assert first == second

    def test_all_edge_types_handled(self, tmp_path):
        """All valid edge types should be indexed correctly."""
        types = ["chunked_from", "derived_from", "related_to", "synthesizes", "contradicts", "references"]
        lines = [f"src.txt\ttgt_{t}.txt\t{t}\tmeta_{t}" for t in types]
        (tmp_path / ".edges").write_text("\n".join(lines) + "\n")
        build_edge_index(str(tmp_path))

        result = lookup_edges(str(tmp_path), "src.txt")
        assert len(result) == len(types)
        result_types = {e["edge_type"] for e in result}
        assert result_types == set(types)
