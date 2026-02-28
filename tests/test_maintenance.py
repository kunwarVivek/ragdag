"""Tests for ragdag maintenance operations — data integrity, GC simulation,
config isolation, and orphan detection."""

import sys
from pathlib import Path

# Ensure sdk and project root are importable
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "sdk"))

import ragdag
from ragdag.core import RagDag


# ================================================================
# Helpers
# ================================================================

def _make_dag(tmp_path):
    """Initialize a fresh ragdag store in tmp_path and return (dag, store)."""
    dag = ragdag.init(str(tmp_path))
    store = tmp_path / ".ragdag"
    return dag, store


def _write_edges(store, edge_lines):
    """Write .edges file with a header + given tab-separated lines."""
    edges_file = store / ".edges"
    content = "# source\ttarget\tedge_type\tmetadata\n"
    content += "\n".join(edge_lines) + "\n"
    edges_file.write_text(content)


def _add_chunk(store, rel_path, content):
    """Manually create a chunk file at store/rel_path with given content."""
    full = store / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")


def _write_config(store, text):
    """Overwrite the .config file."""
    config_file = store / ".config"
    config_file.write_text(text)


def _read_edges(store):
    """Read all non-comment, non-empty edge lines from .edges file."""
    edges_file = store / ".edges"
    if not edges_file.exists():
        return []
    lines = []
    for line in edges_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def _filter_valid_edges(store, edge_lines):
    """Filter edge lines to only those where both source and target exist as
    chunk files in the store (or are absolute paths to existing files)."""
    valid = []
    for line in edge_lines:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        source, target = parts[0], parts[1]
        source_exists = (store / source).exists() or Path(source).exists()
        target_exists = (store / target).exists() or Path(target).exists()
        if source_exists and target_exists:
            valid.append(line)
    return valid


# ================================================================
# Maintenance Tests
# ================================================================

class TestMaintenance:
    """Tests for data integrity, GC simulation, and repair logic."""

    def test_verify_healthy_store_graph_is_consistent(self, tmp_path):
        """A store where all edges reference existing chunks has no
        inconsistencies -- graph().edges matches actual edge count."""
        dag, store = _make_dag(tmp_path)

        # Create chunks that edges will reference
        _add_chunk(store, "alpha/doc1/01.txt", "Chunk 1")
        _add_chunk(store, "alpha/doc1/02.txt", "Chunk 2")
        _add_chunk(store, "beta/doc2/01.txt", "Chunk 3")

        _write_edges(store, [
            "alpha/doc1/01.txt\talpha/doc1/02.txt\trelated_to\t",
            "alpha/doc1/02.txt\tbeta/doc2/01.txt\treferences\t",
        ])

        stats = dag.graph()
        edge_lines = _read_edges(store)
        assert stats.edges == len(edge_lines) == 2

        # All sources and targets exist in the store
        for line in edge_lines:
            parts = line.split("\t")
            source, target = parts[0], parts[1]
            assert (store / source).exists(), f"Source chunk missing: {source}"
            assert (store / target).exists(), f"Target chunk missing: {target}"

    def test_detect_orphaned_edges(self, tmp_path):
        """Edges pointing to nonexistent chunks can be detected by checking
        if source/target files exist."""
        dag, store = _make_dag(tmp_path)

        # Create only one real chunk
        _add_chunk(store, "alpha/doc1/01.txt", "Real chunk")

        # Write edges -- one valid, two orphaned
        _write_edges(store, [
            "alpha/doc1/01.txt\talpha/doc1/01.txt\trelated_to\t",   # valid (self-ref)
            "alpha/doc1/01.txt\tghost/missing/01.txt\treferences\t",  # orphaned target
            "ghost/gone/01.txt\talpha/doc1/01.txt\trelated_to\t",     # orphaned source
        ])

        edge_lines = _read_edges(store)
        orphaned = []
        for line in edge_lines:
            parts = line.split("\t")
            source, target = parts[0], parts[1]
            if not (store / source).exists() or not (store / target).exists():
                orphaned.append(line)

        assert len(orphaned) == 2, f"Expected 2 orphaned edges, found {len(orphaned)}"

    def test_detect_stale_processed(self, tmp_path):
        """Processed entries for deleted source files can be detected."""
        dag, store = _make_dag(tmp_path)

        # Add a file, which records it in .processed
        doc = tmp_path / "temp.md"
        doc.write_text("# Temp\n\nTemporary content.\n")
        dag.add(str(doc))

        # Verify it is in .processed
        processed_text = (store / ".processed").read_text()
        assert str(doc.resolve()) in processed_text

        # Delete the source file
        doc.unlink()

        # Detect stale entries: source paths that no longer exist
        stale = []
        for line in (store / ".processed").read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            source_path = parts[0]
            if not Path(source_path).exists():
                stale.append(source_path)

        assert len(stale) >= 1, "Expected at least one stale .processed entry"

    def test_gc_orphaned_edges_removable(self, tmp_path):
        """Edges to nonexistent chunks can be filtered out (GC simulation)."""
        dag, store = _make_dag(tmp_path)

        _add_chunk(store, "alpha/doc1/01.txt", "Real chunk")

        # 1 valid + 2 orphaned edges
        _write_edges(store, [
            "alpha/doc1/01.txt\talpha/doc1/01.txt\trelated_to\t",
            "alpha/doc1/01.txt\tmissing/chunk/01.txt\treferences\t",
            "missing/source/01.txt\talpha/doc1/01.txt\trelated_to\t",
        ])

        edge_lines = _read_edges(store)
        assert len(edge_lines) == 3

        valid = _filter_valid_edges(store, edge_lines)
        assert len(valid) == 1, f"Expected 1 valid edge after GC, got {len(valid)}"

    def test_gc_preserves_valid_edges(self, tmp_path):
        """Valid edges (pointing to existing chunks) are preserved after
        filtering."""
        dag, store = _make_dag(tmp_path)

        _add_chunk(store, "alpha/doc1/01.txt", "Chunk 1")
        _add_chunk(store, "alpha/doc1/02.txt", "Chunk 2")
        _add_chunk(store, "beta/doc2/01.txt", "Chunk 3")

        _write_edges(store, [
            "alpha/doc1/01.txt\talpha/doc1/02.txt\trelated_to\t",
            "alpha/doc1/02.txt\tbeta/doc2/01.txt\treferences\t",
            "beta/doc2/01.txt\talpha/doc1/01.txt\trelated_to\t",
        ])

        edge_lines = _read_edges(store)
        valid = _filter_valid_edges(store, edge_lines)
        assert len(valid) == 3, "All 3 valid edges should be preserved"

    def test_repair_removes_orphaned_edges(self, tmp_path):
        """After removing edges pointing to missing files, edge count
        decreases."""
        dag, store = _make_dag(tmp_path)

        _add_chunk(store, "alpha/doc1/01.txt", "Chunk 1")

        _write_edges(store, [
            "alpha/doc1/01.txt\talpha/doc1/01.txt\trelated_to\t",
            "alpha/doc1/01.txt\tghost/01.txt\treferences\t",
            "ghost/02.txt\talpha/doc1/01.txt\trelated_to\t",
        ])

        # Before repair
        stats_before = dag.graph()
        assert stats_before.edges == 3

        # Simulate repair: filter valid edges and rewrite
        edge_lines = _read_edges(store)
        valid = _filter_valid_edges(store, edge_lines)

        # Rewrite edges file with only valid edges
        edges_file = store / ".edges"
        header = "# source\ttarget\tedge_type\tmetadata\n"
        edges_file.write_text(header + "\n".join(valid) + "\n")

        # After repair
        stats_after = dag.graph()
        assert stats_after.edges < stats_before.edges
        assert stats_after.edges == 1

    def test_repair_preserves_valid_edges(self, tmp_path):
        """Repair does not remove edges between existing nodes."""
        dag, store = _make_dag(tmp_path)

        _add_chunk(store, "alpha/doc1/01.txt", "Chunk 1")
        _add_chunk(store, "alpha/doc1/02.txt", "Chunk 2")

        _write_edges(store, [
            "alpha/doc1/01.txt\talpha/doc1/02.txt\trelated_to\t",
            "alpha/doc1/02.txt\talpha/doc1/01.txt\treferences\t",
        ])

        # Simulate repair
        edge_lines = _read_edges(store)
        valid = _filter_valid_edges(store, edge_lines)

        edges_file = store / ".edges"
        header = "# source\ttarget\tedge_type\tmetadata\n"
        edges_file.write_text(header + "\n".join(valid) + "\n")

        stats = dag.graph()
        assert stats.edges == 2, "Both valid edges should survive repair"

    def test_config_get_returns_value(self, tmp_path):
        """_read_config returns correct value for existing key."""
        dag, store = _make_dag(tmp_path)
        _write_config(store, (
            "[general]\n"
            "chunk_size = 500\n\n"
            "[embedding]\n"
            "provider = openai\n"
        ))

        assert dag._read_config("general.chunk_size", "0") == "500"
        assert dag._read_config("embedding.provider", "none") == "openai"

    def test_config_get_missing_returns_default(self, tmp_path):
        """_read_config returns default for missing key."""
        dag, store = _make_dag(tmp_path)
        _write_config(store, "[general]\nchunk_size = 500\n")

        result = dag._read_config("general.missing_key", "my_default")
        assert result == "my_default"

    def test_config_sections_isolated(self, tmp_path):
        """Same key name in different sections returns correct value per
        section."""
        dag, store = _make_dag(tmp_path)
        _write_config(store, (
            "[general]\n"
            "name = general_val\n\n"
            "[embedding]\n"
            "name = embedding_val\n\n"
            "[llm]\n"
            "name = llm_val\n"
        ))

        assert dag._read_config("general.name", "") == "general_val"
        assert dag._read_config("embedding.name", "") == "embedding_val"
        assert dag._read_config("llm.name", "") == "llm_val"

    def test_repair_healthy_store_noop(self, tmp_path):
        """On a valid store with only valid edges, repair (filtering edges)
        produces the same edge content."""
        dag, store = _make_dag(tmp_path)

        _add_chunk(store, "alpha/doc1/01.txt", "Chunk 1")
        _add_chunk(store, "alpha/doc1/02.txt", "Chunk 2")
        _add_chunk(store, "beta/doc2/01.txt", "Chunk 3")

        _write_edges(store, [
            "alpha/doc1/01.txt\talpha/doc1/02.txt\trelated_to\t",
            "alpha/doc1/02.txt\tbeta/doc2/01.txt\treferences\t",
            "beta/doc2/01.txt\talpha/doc1/01.txt\trelated_to\t",
        ])

        edge_lines_before = _read_edges(store)
        valid = _filter_valid_edges(store, edge_lines_before)

        # All edges should survive -- no orphans
        assert len(valid) == len(edge_lines_before), (
            "Healthy store repair should not remove any edges"
        )
        assert valid == edge_lines_before, (
            "Filtered edges should be identical to original for a healthy store"
        )

    def test_detect_bad_edge_format(self, tmp_path):
        """Lines in .edges with fewer than 3 tab-separated fields are
        detected as malformed."""
        dag, store = _make_dag(tmp_path)

        # Write a mix of valid and malformed edge lines
        edges_file = store / ".edges"
        edges_file.write_text(
            "# source\ttarget\tedge_type\tmetadata\n"
            "alpha/doc1/01.txt\talpha/doc1/02.txt\trelated_to\t\n"
            "malformed_line_no_tabs\n"
            "only\tone_tab\n"
            "alpha/doc1/02.txt\tbeta/doc2/01.txt\treferences\t\n"
        )

        edge_lines = _read_edges(store)
        malformed = []
        well_formed = []
        for line in edge_lines:
            parts = line.split("\t")
            if len(parts) < 3:
                malformed.append(line)
            else:
                well_formed.append(line)

        assert len(malformed) == 2, (
            f"Expected 2 malformed lines, found {len(malformed)}: {malformed}"
        )
        assert len(well_formed) == 2, (
            f"Expected 2 well-formed lines, found {len(well_formed)}"
        )
        # Verify the malformed lines are the ones we expect
        assert "malformed_line_no_tabs" in malformed[0]
        assert "only\tone_tab" in malformed[1]
