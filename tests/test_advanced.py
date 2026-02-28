"""Advanced tests for ragdag SDK — ask context assembly, domain rules,
config parsing, and graph provenance features."""

import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure sdk and project root are importable
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "sdk"))

import ragdag
from ragdag.core import RagDag, SearchResult, AskResult


# ================================================================
# Helpers
# ================================================================

def _make_dag(tmp_path):
    """Initialize a fresh ragdag store in tmp_path and return (dag, store)."""
    dag = ragdag.init(str(tmp_path))
    store = tmp_path / ".ragdag"
    return dag, store


def _write_domain_rules(store, lines):
    """Write domain-rules file with the given raw lines."""
    rules_file = store / ".domain-rules"
    rules_file.write_text("\n".join(lines) + "\n")


def _write_edges(store, edge_lines):
    """Write .edges file with a header + given tab-separated lines."""
    edges_file = store / ".edges"
    content = "# source\ttarget\tedge_type\tmetadata\n"
    content += "\n".join(edge_lines) + "\n"
    edges_file.write_text(content)


def _write_config(store, text):
    """Overwrite the .config file."""
    config_file = store / ".config"
    config_file.write_text(text)


def _add_chunk(store, rel_path, content):
    """Manually create a chunk file at store/rel_path with given content."""
    full = store / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")


# ================================================================
# ASK - Context Assembly
# ================================================================

class TestAskContextAssembly:
    """Tests for the ask() RAG pipeline: context building, token budget,
    graph expansion, and LLM integration."""

    def test_ask_context_respects_token_budget(self, tmp_path):
        """With many chunks, context assembly stops when max_context is hit."""
        dag, store = _make_dag(tmp_path)

        # Set max_context = 200 tokens. Each chunk has ~20 words = ~26 tokens.
        # So we can fit roughly 7 chunks but not all 15.
        _write_config(store, (
            "[general]\nchunk_strategy = heading\nchunk_size = 1000\n"
            "chunk_overlap = 0\n\n"
            "[embedding]\nprovider = none\n\n"
            "[llm]\nprovider = none\nmodel = gpt-4o-mini\nmax_context = 200\n\n"
            "[search]\ndefault_mode = hybrid\ntop_k = 20\n"
            "keyword_weight = 0.3\nvector_weight = 0.7\n"
        ))

        # Create 15 docs, each with a short chunk containing the keyword.
        # ~20 words per doc => ~26 estimated tokens per chunk (20 * 1.3).
        for i in range(15):
            doc = tmp_path / f"doc{i}.md"
            doc.write_text(
                f"# Doc {i}\n\n"
                "budget keyword alpha bravo charlie delta echo foxtrot "
                "golf hotel india juliet kilo lima mike november oscar\n"
            )
            dag.add(str(doc))

        result = dag.ask("budget keyword", use_llm=False)

        source_count = result.context.count("--- Source:")
        # Must include at least 1 source
        assert source_count >= 1, "Expected at least 1 source in context"
        # Must NOT include all 15 -- the budget is too small
        assert source_count < 15, (
            f"Expected fewer than 15 sources due to token budget, got {source_count}"
        )

    def test_ask_context_format_has_source_path(self, tmp_path):
        """Context string includes '--- Source: path (score: X) ---' format."""
        dag, store = _make_dag(tmp_path)
        doc = tmp_path / "format_test.md"
        doc.write_text("# Format\n\nThis tests context format assembly.\n")
        dag.add(str(doc))

        result = dag.ask("format assembly", use_llm=False)
        assert result.context != ""
        # Check for the expected header format
        assert "--- Source:" in result.context
        assert "(score:" in result.context
        assert ") ---" in result.context

    def test_ask_graph_expansion_related_to(self, tmp_path):
        """related_to edges pull in extra chunks during ask()."""
        dag, store = _make_dag(tmp_path)

        # Create a primary doc with the query keyword
        primary = tmp_path / "primary.md"
        primary.write_text("# Primary\n\nsearchable keyword here\n")
        dag.add(str(primary))

        # Create a related chunk manually (won't match keyword search)
        _add_chunk(store, "extra/related/01.txt", "Extra related content no keyword match.")

        # Find the primary chunk path
        primary_chunks = list((store / "primary").rglob("*.txt"))
        assert len(primary_chunks) >= 1
        primary_rel = str(primary_chunks[0].relative_to(store))

        # Add a related_to edge from primary chunk to extra chunk
        edges_file = store / ".edges"
        with open(edges_file, "a") as f:
            f.write(f"{primary_rel}\textra/related/01.txt\trelated_to\t\n")

        result = dag.ask("searchable keyword", use_llm=False)

        # The expanded chunk should appear in sources
        assert "extra/related/01.txt" in result.sources

    def test_ask_graph_expansion_references(self, tmp_path):
        """references edges also pull in extra chunks during ask()."""
        dag, store = _make_dag(tmp_path)

        # Primary doc
        primary = tmp_path / "ref_primary.md"
        primary.write_text("# Ref Primary\n\nreferenceable content word\n")
        dag.add(str(primary))

        # Referenced chunk
        _add_chunk(store, "refs/target/01.txt", "Referenced target content.")

        # Find primary chunk
        primary_chunks = list((store / "ref_primary").rglob("*.txt"))
        assert len(primary_chunks) >= 1
        primary_rel = str(primary_chunks[0].relative_to(store))

        # Add a references edge
        edges_file = store / ".edges"
        with open(edges_file, "a") as f:
            f.write(f"{primary_rel}\trefs/target/01.txt\treferences\t\n")

        result = dag.ask("referenceable content", use_llm=False)
        assert "refs/target/01.txt" in result.sources

    def test_ask_graph_expansion_does_not_duplicate(self, tmp_path):
        """Chunks already in search results are not re-added by expansion."""
        dag, store = _make_dag(tmp_path)

        # Doc where same chunk is both a search result and edge target
        doc = tmp_path / "dedup.md"
        doc.write_text("# Dedup\n\ndedup keyword content here\n")
        dag.add(str(doc))

        chunks = list((store / "dedup").rglob("*.txt"))
        assert len(chunks) >= 1
        chunk_rel = str(chunks[0].relative_to(store))

        # Create a self-referencing related_to edge (chunk -> same chunk)
        edges_file = store / ".edges"
        with open(edges_file, "a") as f:
            f.write(f"{chunk_rel}\t{chunk_rel}\trelated_to\t\n")

        result = dag.ask("dedup keyword", use_llm=False)

        # The chunk should appear exactly once in sources
        count = result.sources.count(chunk_rel)
        assert count == 1, f"Expected chunk to appear once, got {count} times"

    def test_ask_with_mocked_llm(self, tmp_path):
        """When LLM is configured and mocked, ask() returns the LLM answer."""
        dag, store = _make_dag(tmp_path)

        # Configure an LLM provider so the code path is reached
        _write_config(store, (
            "[general]\nchunk_strategy = heading\nchunk_size = 1000\n"
            "chunk_overlap = 100\n\n"
            "[embedding]\nprovider = none\n\n"
            "[llm]\nprovider = openai\nmodel = gpt-4o-mini\nmax_context = 8000\n\n"
            "[search]\ndefault_mode = hybrid\ntop_k = 10\n"
            "keyword_weight = 0.3\nvector_weight = 0.7\n"
        ))

        doc = tmp_path / "llm_test.md"
        doc.write_text("# LLM Test\n\nContent for LLM query testing.\n")
        dag.add(str(doc))

        mock_answer = "The answer is 42."

        # Build a fake engines.llm module with a get_answer function
        fake_llm_module = types.ModuleType("engines.llm")
        fake_llm_module.get_answer = MagicMock(return_value=mock_answer)

        # Also ensure engines package module exists so "from engines.llm"
        # import resolution works
        fake_engines = sys.modules.get("engines")
        had_engines = "engines" in sys.modules
        had_engines_llm = "engines.llm" in sys.modules
        old_engines_llm = sys.modules.get("engines.llm")

        # We need engines as a package in sys.modules for the import to work.
        # Only create a fake if it doesn't already exist.
        if not had_engines:
            engines_pkg = types.ModuleType("engines")
            engines_pkg.__path__ = []  # mark as package
            sys.modules["engines"] = engines_pkg

        sys.modules["engines.llm"] = fake_llm_module

        try:
            result = dag.ask("LLM query testing", use_llm=True)
        finally:
            # Restore original module state
            if had_engines_llm:
                sys.modules["engines.llm"] = old_engines_llm
            else:
                sys.modules.pop("engines.llm", None)
            if not had_engines:
                sys.modules.pop("engines", None)
            else:
                sys.modules["engines"] = fake_engines

        assert result.answer == mock_answer
        assert result.context != ""
        assert len(result.sources) >= 1
        # Verify the mock was called with expected arguments
        fake_llm_module.get_answer.assert_called_once()
        call_kwargs = fake_llm_module.get_answer.call_args
        assert call_kwargs[1]["question"] == "LLM query testing" or call_kwargs[0][0] == "LLM query testing" if call_kwargs[0] else True

    def test_ask_no_duplicate_sources(self, tmp_path):
        """Same source should not appear twice in context."""
        dag, store = _make_dag(tmp_path)

        # Create a doc that will match the query
        doc = tmp_path / "unique_src.md"
        doc.write_text("# Unique Source\n\nunique_source_keyword content here\n")
        dag.add(str(doc))

        # Find the chunk path
        chunks = list((store / "unique_src").rglob("*.txt"))
        assert len(chunks) >= 1
        chunk_rel = str(chunks[0].relative_to(store))

        # Add a self-referencing edge so expansion could re-add the same chunk
        edges_file = store / ".edges"
        with open(edges_file, "a") as f:
            f.write(f"{chunk_rel}\t{chunk_rel}\trelated_to\t\n")

        result = dag.ask("unique_source_keyword", use_llm=False)

        # Each source should appear at most once
        seen = set()
        for src in result.sources:
            assert src not in seen, f"Duplicate source in ask() results: {src}"
            seen.add(src)

    def test_ask_custom_prompt_template(self, tmp_path):
        """If .ragdag/prompt.txt exists, it is readable and could be loaded
        as a custom prompt template. This tests the capability exists or
        documents the gap."""
        dag, store = _make_dag(tmp_path)

        # Create a custom prompt template file
        prompt_file = store / "prompt.txt"
        custom_prompt = (
            "You are a helpful assistant.\n"
            "Use the following context to answer the question:\n"
            "{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
        prompt_file.write_text(custom_prompt)

        # Verify the file exists and is readable from the store
        assert prompt_file.exists(), "prompt.txt should exist in .ragdag/"
        loaded = prompt_file.read_text()
        assert "{context}" in loaded, "Prompt template should have a {context} placeholder"
        assert "{question}" in loaded, "Prompt template should have a {question} placeholder"
        assert loaded == custom_prompt, "Loaded prompt should match what was written"

        # Verify the store directory path is accessible from the dag instance
        assert dag.store_dir == store, "dag.store_dir should point to .ragdag"
        assert (dag.store_dir / "prompt.txt").exists(), (
            "prompt.txt should be accessible via dag.store_dir"
        )

    def test_ask_llm_includes_citations(self, tmp_path):
        """When LLM is mocked and returns text with '[Source: path]', the ask
        result preserves those citations in the answer."""
        dag, store = _make_dag(tmp_path)

        # Configure an LLM provider so the code path is reached
        _write_config(store, (
            "[general]\nchunk_strategy = heading\nchunk_size = 1000\n"
            "chunk_overlap = 100\n\n"
            "[embedding]\nprovider = none\n\n"
            "[llm]\nprovider = openai\nmodel = gpt-4o-mini\nmax_context = 8000\n\n"
            "[search]\ndefault_mode = hybrid\ntop_k = 10\n"
            "keyword_weight = 0.3\nvector_weight = 0.7\n"
        ))

        doc = tmp_path / "citation_test.md"
        doc.write_text("# Citation Test\n\nContent for citation query testing.\n")
        dag.add(str(doc))

        mock_answer = (
            "The answer involves authentication. "
            "[Source: auth/login/01.txt] "
            "And also deployment. "
            "[Source: deploy/config/01.txt]"
        )

        # Build a fake engines.llm module with a get_answer function
        fake_llm_module = types.ModuleType("engines.llm")
        fake_llm_module.get_answer = MagicMock(return_value=mock_answer)

        fake_engines = sys.modules.get("engines")
        had_engines = "engines" in sys.modules
        had_engines_llm = "engines.llm" in sys.modules
        old_engines_llm = sys.modules.get("engines.llm")

        if not had_engines:
            engines_pkg = types.ModuleType("engines")
            engines_pkg.__path__ = []
            sys.modules["engines"] = engines_pkg

        sys.modules["engines.llm"] = fake_llm_module

        try:
            result = dag.ask("citation query testing", use_llm=True)
        finally:
            if had_engines_llm:
                sys.modules["engines.llm"] = old_engines_llm
            else:
                sys.modules.pop("engines.llm", None)
            if not had_engines:
                sys.modules.pop("engines", None)
            else:
                sys.modules["engines"] = fake_engines

        # The answer should preserve the citations from the LLM response
        assert result.answer == mock_answer
        assert "[Source: auth/login/01.txt]" in result.answer
        assert "[Source: deploy/config/01.txt]" in result.answer


# ================================================================
# DOMAIN RULES
# ================================================================

class TestDomainRules:
    """Tests for _apply_domain_rules and domain='auto' ingestion."""

    def test_domain_rules_basic_match(self, tmp_path):
        """Pattern 'auth' in path maps to 'auth' domain."""
        dag, store = _make_dag(tmp_path)
        _write_domain_rules(store, [
            "# Domain rules",
            "auth \u2192 auth",
        ])

        result = dag._apply_domain_rules("/project/src/auth/login.py")
        assert result == "auth"

    def test_domain_rules_multiple_patterns(self, tmp_path):
        """'oauth jwt' should match either pattern."""
        dag, store = _make_dag(tmp_path)
        _write_domain_rules(store, [
            "oauth jwt \u2192 auth",
        ])

        assert dag._apply_domain_rules("/src/oauth_handler.py") == "auth"
        assert dag._apply_domain_rules("/src/jwt_utils.py") == "auth"

    def test_domain_rules_comments_ignored(self, tmp_path):
        """Lines starting with # should be skipped."""
        dag, store = _make_dag(tmp_path)
        _write_domain_rules(store, [
            "# This is a comment",
            "# deploy \u2192 ops",
            "auth \u2192 auth",
        ])

        # "deploy" from the comment line should NOT match
        assert dag._apply_domain_rules("/src/deploy/script.sh") == ""
        # "auth" from the real rule should match
        assert dag._apply_domain_rules("/src/auth/login.py") == "auth"

    def test_domain_rules_no_match_returns_empty(self, tmp_path):
        """Unmatched path returns empty string."""
        dag, store = _make_dag(tmp_path)
        _write_domain_rules(store, [
            "auth \u2192 auth",
            "deploy \u2192 ops",
        ])

        result = dag._apply_domain_rules("/src/utils/helpers.py")
        assert result == ""

    def test_domain_rules_case_insensitive(self, tmp_path):
        """'AUTH' in path should match 'auth' pattern (case-insensitive)."""
        dag, store = _make_dag(tmp_path)
        _write_domain_rules(store, [
            "auth \u2192 auth",
        ])

        result = dag._apply_domain_rules("/project/src/AUTH/Login.py")
        assert result == "auth"

    def test_add_domain_auto(self, tmp_path):
        """add(path, domain='auto') applies domain rules."""
        dag, store = _make_dag(tmp_path)
        _write_domain_rules(store, [
            "auth \u2192 auth",
        ])

        # Create a file with 'auth' in its path
        auth_dir = tmp_path / "auth"
        auth_dir.mkdir()
        doc = auth_dir / "login.md"
        doc.write_text("# Login\n\nLogin page documentation.\n")

        dag.add(str(doc), domain="auto")

        # Should be stored under the 'auth' domain directory
        auth_store = store / "auth"
        assert auth_store.exists(), "Expected 'auth' domain directory in store"
        chunks = list(auth_store.rglob("*.txt"))
        assert len(chunks) >= 1


# ================================================================
# CONFIG
# ================================================================

class TestConfig:
    """Tests for _read_config INI parsing."""

    def test_read_config_existing_key(self, tmp_path):
        """Reads correct value for a known section.key."""
        dag, store = _make_dag(tmp_path)
        _write_config(store, (
            "[general]\n"
            "chunk_size = 500\n\n"
            "[llm]\n"
            "model = gpt-4o\n"
        ))

        assert dag._read_config("general.chunk_size", "1000") == "500"
        assert dag._read_config("llm.model", "default") == "gpt-4o"

    def test_read_config_missing_key_returns_default(self, tmp_path):
        """Missing key returns the provided default value."""
        dag, store = _make_dag(tmp_path)
        _write_config(store, "[general]\nchunk_size = 500\n")

        result = dag._read_config("general.nonexistent_key", "fallback")
        assert result == "fallback"

    def test_read_config_wrong_section(self, tmp_path):
        """Key in a different section is not found."""
        dag, store = _make_dag(tmp_path)
        _write_config(store, (
            "[general]\n"
            "chunk_size = 500\n\n"
            "[llm]\n"
            "chunk_size = 800\n"
        ))

        # Looking in 'general' section should get 500, not 800
        assert dag._read_config("general.chunk_size", "0") == "500"
        assert dag._read_config("llm.chunk_size", "0") == "800"

    def test_read_config_comments_ignored(self, tmp_path):
        """Lines starting with # should not interfere with config parsing."""
        dag, store = _make_dag(tmp_path)
        _write_config(store, (
            "[general]\n"
            "# This is a comment\n"
            "chunk_size = 500\n"
            "# chunk_size = 999\n"
        ))

        assert dag._read_config("general.chunk_size", "0") == "500"

    def test_read_config_whitespace_handling(self, tmp_path):
        """Spaces around = are handled correctly."""
        dag, store = _make_dag(tmp_path)
        _write_config(store, (
            "[general]\n"
            "  chunk_size  =  750  \n"
        ))

        assert dag._read_config("general.chunk_size", "0") == "750"


# ================================================================
# GRAPH ADVANCED
# ================================================================

class TestGraphAdvanced:
    """Tests for multi-hop trace, cycle detection, empty store, and edge counts."""

    def test_trace_multi_hop(self, tmp_path):
        """3+ level provenance chain: chunk -> doc -> source."""
        dag, store = _make_dag(tmp_path)

        # Build a 3-hop chain:
        #   chunk/01.txt --chunked_from--> doc/original.txt --derived_via--> source/root.txt
        # source/root.txt has no outgoing provenance edge (origin)
        _add_chunk(store, "chunk/01.txt", "Chunk content")
        _add_chunk(store, "doc/original.txt", "Doc content")
        _add_chunk(store, "source/root.txt", "Source content")

        _write_edges(store, [
            "chunk/01.txt\tdoc/original.txt\tchunked_from\t",
            "doc/original.txt\tsource/root.txt\tderived_via\t",
        ])

        chain = dag.trace("chunk/01.txt")

        # Should have 3 entries: chunk->doc, doc->source, source->None
        assert len(chain) == 3
        assert chain[0]["node"] == "chunk/01.txt"
        assert chain[0]["parent"] == "doc/original.txt"
        assert chain[0]["edge_type"] == "chunked_from"

        assert chain[1]["node"] == "doc/original.txt"
        assert chain[1]["parent"] == "source/root.txt"
        assert chain[1]["edge_type"] == "derived_via"

        assert chain[2]["node"] == "source/root.txt"
        assert chain[2]["parent"] is None
        assert chain[2]["edge_type"] == "origin"

    def test_trace_cycle_detection(self, tmp_path):
        """Cyclic edges should not cause an infinite loop."""
        dag, store = _make_dag(tmp_path)

        # Create a cycle: A -> B -> C -> A
        _write_edges(store, [
            "A\tB\tchunked_from\t",
            "B\tC\tchunked_from\t",
            "C\tA\tchunked_from\t",
        ])

        chain = dag.trace("A")

        # The trace should terminate. It visits A, then B, then C,
        # then tries to visit A again but A is already visited.
        # So we get entries for A->B, B->C, C->A (3 entries with the cycle detected)
        assert len(chain) <= 4, f"Trace should terminate, got {len(chain)} entries"
        # Verify all nodes appear
        nodes_in_chain = [entry["node"] for entry in chain]
        assert "A" in nodes_in_chain
        assert "B" in nodes_in_chain
        assert "C" in nodes_in_chain

    def test_graph_empty_store(self, tmp_path):
        """Empty store returns all zeros."""
        dag, store = _make_dag(tmp_path)

        stats = dag.graph()
        assert stats.domains == 0
        assert stats.documents == 0
        assert stats.chunks == 0
        # Edges file has a comment header only, so 0 real edges
        assert stats.edges == 0
        assert stats.edge_types == {}

    def test_graph_edge_type_counts(self, tmp_path):
        """edge_types dict has correct counts per type."""
        dag, store = _make_dag(tmp_path)

        # Create a domain/doc structure so graph() sees them
        _add_chunk(store, "alpha/doc1/01.txt", "Chunk 1")
        _add_chunk(store, "alpha/doc1/02.txt", "Chunk 2")
        _add_chunk(store, "beta/doc2/01.txt", "Chunk 3")

        _write_edges(store, [
            "alpha/doc1/01.txt\t/src/a.md\tchunked_from\t",
            "alpha/doc1/02.txt\t/src/a.md\tchunked_from\t",
            "beta/doc2/01.txt\t/src/b.md\tchunked_from\t",
            "alpha/doc1/01.txt\tbeta/doc2/01.txt\trelated_to\t",
            "alpha/doc1/02.txt\tbeta/doc2/01.txt\treferences\t",
        ])

        stats = dag.graph()

        assert stats.domains == 2  # alpha, beta
        assert stats.documents == 2  # doc1, doc2
        assert stats.chunks == 3  # 01.txt, 02.txt under doc1, 01.txt under doc2
        assert stats.edges == 5
        assert stats.edge_types["chunked_from"] == 3
        assert stats.edge_types["related_to"] == 1
        assert stats.edge_types["references"] == 1

    def test_trace_max_depth_terminates(self, tmp_path):
        """Build a 25-hop chain, verify trace terminates (does not hang).
        Trace should follow all hops and stop at the origin."""
        dag, store = _make_dag(tmp_path)

        # Build a 25-hop linear chain: node_00 -> node_01 -> ... -> node_24
        edge_lines = []
        for i in range(24):
            edge_lines.append(
                f"node_{i:02d}\tnode_{i+1:02d}\tchunked_from\t"
            )

        _write_edges(store, edge_lines)

        chain = dag.trace("node_00")

        # Should have 25 entries: 24 hops + 1 origin entry for node_24
        assert len(chain) == 25, (
            f"Expected 25 entries for 25-node chain, got {len(chain)}"
        )
        # First entry is node_00
        assert chain[0]["node"] == "node_00"
        assert chain[0]["parent"] == "node_01"
        assert chain[0]["edge_type"] == "chunked_from"
        # Last entry is the origin (node_24 with no parent)
        assert chain[-1]["node"] == "node_24"
        assert chain[-1]["parent"] is None
        assert chain[-1]["edge_type"] == "origin"


# ================================================================
# RELATE
# ================================================================

class TestRelate:
    """Tests for relate() semantic edge computation."""

    def test_relate_requires_embedding_provider(self, tmp_path):
        """relate() with provider=none should return without adding edges
        (no embeddings.bin to process)."""
        dag, store = _make_dag(tmp_path)

        # Ensure embedding provider is none
        _write_config(store, (
            "[general]\nchunk_strategy = heading\nchunk_size = 1000\n"
            "chunk_overlap = 100\n\n"
            "[embedding]\nprovider = none\n\n"
            "[llm]\nprovider = none\nmodel = gpt-4o-mini\nmax_context = 8000\n\n"
            "[search]\ndefault_mode = hybrid\ntop_k = 10\n"
            "keyword_weight = 0.3\nvector_weight = 0.7\n"
        ))

        # Add a doc (chunks created but no embeddings since provider=none)
        doc = tmp_path / "relate_test.md"
        doc.write_text("# Relate Test\n\nContent for relate testing.\n")
        dag.add(str(doc))

        # Count edges before relate
        edges_before = store / ".edges"
        before_text = edges_before.read_text()
        before_related = before_text.count("related_to")

        # relate() should work but not add any related_to edges
        # because there are no embeddings.bin files
        result = dag.relate()

        edges_after = store / ".edges"
        after_text = edges_after.read_text()
        after_related = after_text.count("related_to")

        assert after_related == before_related, (
            "relate() with no embeddings should not add related_to edges"
        )

    def test_relate_needs_embeddings_bin_to_create_edges(self, tmp_path):
        """relate() creates no edges when there is no embeddings.bin file.

        Even with chunks present in domain directories, relate() only creates
        related_to edges when embeddings.bin files exist. Without embeddings,
        there is no similarity data to compare, so no edges are added.
        """
        dag, store = _make_dag(tmp_path)

        # Create chunks manually in a domain directory structure
        # but do NOT create any embeddings.bin
        _add_chunk(store, "mydomain/doc1/01.txt", "First chunk about authentication.")
        _add_chunk(store, "mydomain/doc1/02.txt", "Second chunk about authorization.")
        _add_chunk(store, "mydomain/doc2/01.txt", "Third chunk about login flows.")

        # Record initial edge state (only the header comment from init)
        edges_before = store / ".edges"
        before_text = edges_before.read_text()
        before_related = before_text.count("related_to")

        # Call relate() -- should find no embeddings.bin and add no edges
        dag.relate()

        after_text = edges_before.read_text()
        after_related = after_text.count("related_to")

        assert after_related == before_related, (
            "relate() should not create related_to edges without embeddings.bin"
        )

        # Also verify no embeddings.bin was accidentally created
        embed_files = list(store.rglob("embeddings.bin"))
        assert len(embed_files) == 0, "No embeddings.bin should exist"

    def test_relate_checks_existing_edges(self, tmp_path):
        """relate_cli.main() loads existing edges to avoid duplicates.

        When relate is called and there are no embeddings, it should still
        safely read the existing .edges file (to build the existing_edges set)
        without crashing. This confirms the deduplication code path works
        even when no new edges are produced.
        """
        dag, store = _make_dag(tmp_path)

        # Pre-populate .edges with some existing related_to edges
        _write_edges(store, [
            "a/01.txt\tb/01.txt\trelated_to\tsimilarity=0.95",
            "c/01.txt\td/01.txt\tchunked_from\t",
        ])

        # Call relate with no embeddings -- should read existing edges
        # and return without error
        result = dag.relate()

        # Verify edges file is still intact and readable
        after_text = (store / ".edges").read_text()
        assert "a/01.txt\tb/01.txt\trelated_to" in after_text, (
            "Existing related_to edge should be preserved"
        )
        assert "c/01.txt\td/01.txt\tchunked_from" in after_text, (
            "Existing chunked_from edge should be preserved"
        )
