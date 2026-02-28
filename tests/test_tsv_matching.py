"""Tests for exact TSV field matching — verifies awk-based matching doesn't do substring matching."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from ragdag.core import RagDag


def _make_store(tmp_path):
    """Create a minimal ragdag store and return RagDag instance."""
    store = tmp_path / ".ragdag"
    store.mkdir(exist_ok=True)
    (store / ".config").write_text(
        "[general]\nchunk_size = 1000\nchunk_overlap = 100\n"
        "[embedding]\nprovider = none\n"
        "[llm]\nprovider = none\n"
        "[search]\ntop_k = 10\n"
        "[edges]\nrecord_queries = false\n"
    )
    (store / ".edges").write_text("# source\ttarget\tedge_type\tmetadata\n")
    (store / ".processed").write_text("# source_path\tcontent_hash\tdomain\ttimestamp\n")
    (store / ".domain-rules").write_text("")
    return RagDag(str(tmp_path))


class TestProcessedExactMatch:
    """Tests that _is_processed does exact field matching, not substring."""

    def test_exact_path_matches(self, tmp_path):
        """Exact path + hash match returns True."""
        dag = _make_store(tmp_path)
        abs_path = Path("/docs/auth.md")
        dag._record_processed(abs_path, "abc123", "auth")
        assert dag._is_processed(abs_path, "abc123") is True

    def test_substring_path_does_not_match(self, tmp_path):
        """A path that is a substring of another should NOT match."""
        dag = _make_store(tmp_path)
        # Record /docs/auth.md
        dag._record_processed(Path("/docs/auth.md"), "abc123", "auth")
        # /docs/auth.md-extra should NOT match
        assert dag._is_processed(Path("/docs/auth.md-extra"), "abc123") is False

    def test_prefix_path_does_not_match(self, tmp_path):
        """A path that is a prefix of the recorded path should NOT match."""
        dag = _make_store(tmp_path)
        dag._record_processed(Path("/docs/auth.md"), "abc123", "auth")
        # /docs/auth should NOT match
        assert dag._is_processed(Path("/docs/auth"), "abc123") is False

    def test_different_hash_does_not_match(self, tmp_path):
        """Same path but different hash should NOT match."""
        dag = _make_store(tmp_path)
        dag._record_processed(Path("/docs/auth.md"), "abc123", "auth")
        assert dag._is_processed(Path("/docs/auth.md"), "def456") is False

    def test_re_record_overwrites_old_entry(self, tmp_path):
        """Recording same path with new hash should overwrite old entry."""
        dag = _make_store(tmp_path)
        path = Path("/docs/auth.md")
        dag._record_processed(path, "old_hash", "auth")
        dag._record_processed(path, "new_hash", "auth")
        assert dag._is_processed(path, "new_hash") is True
        assert dag._is_processed(path, "old_hash") is False


class TestEdgesExactMatch:
    """Tests that edge operations do exact field matching."""

    def test_neighbors_exact_source_match(self, tmp_path):
        """neighbors() should match exact source path, not substrings."""
        dag = _make_store(tmp_path)
        edges = dag._edges_path()
        edges.write_text(
            "auth/intro/01.txt\t/docs/auth.md\tchunked_from\t\n"
            "auth/intro-extra/01.txt\t/docs/auth-extra.md\tchunked_from\t\n"
        )
        neighbors = dag.neighbors("auth/intro/01.txt")
        # Should only find the exact match, not "auth/intro-extra/01.txt"
        paths = [n["node"] for n in neighbors]
        assert "/docs/auth.md" in paths
        assert "/docs/auth-extra.md" not in paths

    def test_neighbors_exact_target_match(self, tmp_path):
        """neighbors() with target matching should be exact."""
        dag = _make_store(tmp_path)
        edges = dag._edges_path()
        edges.write_text(
            "a/01.txt\tnode_x\treferences\t\n"
            "b/01.txt\tnode_xy\treferences\t\n"
        )
        neighbors = dag.neighbors("node_x")
        sources = [n["node"] for n in neighbors if n["direction"] == "incoming"]
        assert "a/01.txt" in sources
        assert "b/01.txt" not in sources

    def test_trace_exact_source_match(self, tmp_path):
        """trace() should follow exact path matches through provenance chain."""
        dag = _make_store(tmp_path)
        edges = dag._edges_path()
        edges.write_text(
            "auth/01.txt\t/docs/auth.md\tchunked_from\t\n"
            "auth/01.txt-backup\t/docs/backup.md\tchunked_from\t\n"
        )
        chain = dag.trace("auth/01.txt")
        assert len(chain) == 2  # chunk -> origin
        assert chain[0]["parent"] == "/docs/auth.md"
        assert chain[1]["node"] == "/docs/auth.md"

    def test_create_chunk_edges_removes_only_exact_source(self, tmp_path):
        """_create_chunk_edges should only remove edges for exact source path."""
        dag = _make_store(tmp_path)
        edges = dag._edges_path()
        edges.write_text(
            "old/01.txt\t/docs/auth.md\tchunked_from\t\n"
            "other/01.txt\t/docs/auth.md-extra\tchunked_from\t\n"
        )
        # Create new chunk
        chunk_dir = dag.store_dir / "new_doc"
        chunk_dir.mkdir(parents=True)
        (chunk_dir / "01.txt").write_text("new content")

        dag._create_chunk_edges("new_doc", "/docs/auth.md")

        content = edges.read_text()
        # Old edge for /docs/auth.md should be removed
        assert "old/01.txt" not in content
        # Edge for /docs/auth.md-extra should be preserved
        assert "/docs/auth.md-extra" in content
        # New edge should be added
        assert "new_doc/01.txt" in content


class TestGraphExpansionExactMatch:
    """Tests that ask() graph expansion uses exact matching."""

    def test_ask_graph_expansion_exact_match(self, tmp_path):
        """Graph expansion should only follow exact source matches in edges."""
        dag = _make_store(tmp_path)

        # Create chunks that will be found by keyword search
        domain_dir = dag.store_dir / "docs" / "main"
        domain_dir.mkdir(parents=True)
        (domain_dir / "01.txt").write_text("authentication login password")

        # Create a related chunk
        related_dir = dag.store_dir / "docs" / "related"
        related_dir.mkdir(parents=True)
        (related_dir / "01.txt").write_text("related auth content")

        # Create an unrelated chunk with similar path prefix
        unrelated_dir = dag.store_dir / "docs" / "unrelated"
        unrelated_dir.mkdir(parents=True)
        (unrelated_dir / "01.txt").write_text("unrelated content")

        # Set up edges: main -> related (exact), but NOT main -> unrelated
        edges = dag._edges_path()
        edges.write_text(
            "docs/main/01.txt\tdocs/related/01.txt\trelated_to\t\n"
        )

        result = dag.ask("authentication", use_llm=False)
        # Should include the related chunk via graph expansion
        assert "docs/main/01.txt" in result.sources
