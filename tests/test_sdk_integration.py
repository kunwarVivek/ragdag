"""Integration tests for ragdag SDK — add, search, ask, graph, neighbors, trace, link."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import ragdag
from ragdag.core import RagDag, SearchResult


class TestInit:
    """Tests for store initialization."""

    def test_init_creates_store(self, tmp_path):
        """ragdag.init() should create .ragdag directory with config files."""
        dag = ragdag.init(str(tmp_path))
        store = tmp_path / ".ragdag"
        assert store.is_dir()
        assert (store / ".config").exists()
        assert (store / ".edges").exists()
        assert (store / ".processed").exists()
        assert (store / ".domain-rules").exists()

    def test_open_existing_store(self, tmp_path):
        """ragdag.open() should work on existing store."""
        ragdag.init(str(tmp_path))
        dag = ragdag.open(str(tmp_path))
        assert dag.store_dir == tmp_path / ".ragdag"

    def test_open_nonexistent_store_raises(self, tmp_path):
        """ragdag.open() on nonexistent path raises appropriate error on store operations.

        open() itself is lazy (no validation), but operations that need the store
        directory (like graph()) raise FileNotFoundError when the store doesn't exist.
        """
        dag = ragdag.open(str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError):
            dag.graph()

    def test_init_default_config_values(self, tmp_path):
        """After init, config has expected defaults (chunk_size=1000, provider=none, top_k=10)."""
        dag = ragdag.init(str(tmp_path))
        assert dag._read_config("general.chunk_size") == "1000"
        assert dag._read_config("embedding.provider") == "none"
        assert dag._read_config("search.top_k") == "10"

    def test_init_idempotent_no_overwrite(self, tmp_path):
        """Calling init twice doesn't overwrite modified config.

        _init_store() only writes config if the file doesn't already exist,
        so user modifications are preserved across repeated init calls.
        """
        dag1 = ragdag.init(str(tmp_path))

        # Modify config to a non-default value
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("chunk_size = 1000", "chunk_size = 500")
        config_path.write_text(config_text)
        assert dag1._read_config("general.chunk_size") == "500"

        # Init again — should NOT overwrite
        dag2 = ragdag.init(str(tmp_path))
        assert dag2._read_config("general.chunk_size") == "500"

    def test_init_gitignore_in_git_repo(self, tmp_path):
        """init() succeeds in a directory that has a .git/ folder.

        The SDK's _init_store() at core.py:93-127 does not currently
        implement .gitignore handling. This test documents that init()
        works without error in a git repo context. If .gitignore
        handling is added later, this test should be updated to verify
        the .ragdag/ entry is added.
        """
        # Simulate a git repo by creating .git directory
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        # init() should succeed without error
        dag = ragdag.init(str(tmp_path))
        store = tmp_path / ".ragdag"
        assert store.is_dir()
        assert (store / ".config").exists()


class TestAdd:
    """Tests for document ingestion."""

    def test_add_single_file(self, tmp_path):
        """Adding a single file creates chunks."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "test.md"
        doc.write_text("# Hello\n\nThis is a test document with some content.\n")

        result = dag.add(str(doc))
        assert result["files"] == 1
        assert result["chunks"] >= 1
        assert result["skipped"] == 0

    def test_add_idempotent(self, tmp_path):
        """Adding same file twice skips on second run."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "test.md"
        doc.write_text("# Test\n\nContent here.\n")

        dag.add(str(doc))
        result = dag.add(str(doc))
        assert result["skipped"] == 1
        assert result["files"] == 0

    def test_add_with_domain(self, tmp_path):
        """Adding with domain organizes under domain directory."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "auth.md"
        doc.write_text("# Auth\n\nAuthentication docs.\n")

        dag.add(str(doc), domain="auth")
        # Chunks should be under .ragdag/auth/
        auth_dir = tmp_path / ".ragdag" / "auth"
        assert auth_dir.exists()
        chunks = list(auth_dir.rglob("*.txt"))
        assert len(chunks) >= 1

    def test_add_directory(self, tmp_path):
        """Adding a directory processes all files."""
        dag = ragdag.init(str(tmp_path))
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "a.md").write_text("# Doc A\n\nContent A.\n")
        (docs_dir / "b.md").write_text("# Doc B\n\nContent B.\n")

        result = dag.add(str(docs_dir))
        assert result["files"] == 2

    def test_add_creates_chunked_from_edges(self, tmp_path):
        """Adding a file should create chunked_from edges."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "test.md"
        doc.write_text("# Test\n\nContent.\n")

        dag.add(str(doc))
        edges_content = (tmp_path / ".ragdag" / ".edges").read_text()
        assert "chunked_from" in edges_content

    def test_add_nonexistent_file_raises(self, tmp_path):
        """Adding a nonexistent file should raise FileNotFoundError."""
        dag = ragdag.init(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            dag.add("/nonexistent/file.md")

    def test_add_changed_file_reprocesses(self, tmp_path):
        """Changing file content and re-adding should reprocess."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "test.md"
        doc.write_text("# Version 1\n\nOriginal content.\n")
        dag.add(str(doc))

        doc.write_text("# Version 2\n\nUpdated content.\n")
        result = dag.add(str(doc))
        assert result["files"] == 1
        assert result["skipped"] == 0

    def test_reingestion_removes_old_chunks(self, tmp_path):
        """Changing file content replaces old chunk files.

        When a file is re-added with different content, the add() method
        removes all old *.txt chunks in the target directory before writing
        new ones (see core.py line 206: old.unlink()).
        """
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "multi.md"
        # Write content that produces multiple chunks (two heading sections)
        doc.write_text("# Part 1\n\nContent for part one.\n\n# Part 2\n\nContent for part two.\n")
        dag.add(str(doc))

        store = tmp_path / ".ragdag"
        # Gather chunk files (exclude dotfiles)
        chunks_before = []
        for f in store.rglob("*.txt"):
            if not any(p.startswith(".") for p in f.relative_to(store).parts):
                chunks_before.append(f)
        assert len(chunks_before) >= 2, f"Expected >= 2 chunks, got {len(chunks_before)}"

        # Now rewrite with single-section content
        doc.write_text("# Single\n\nOnly one section now.\n")
        result = dag.add(str(doc))
        assert result["files"] == 1

        chunks_after = []
        for f in store.rglob("*.txt"):
            if not any(p.startswith(".") for p in f.relative_to(store).parts):
                chunks_after.append(f)
        assert len(chunks_after) < len(chunks_before)
        # Verify new content is present
        assert any("Only one section" in f.read_text() for f in chunks_after)

    def test_add_empty_file(self, tmp_path):
        """Adding empty file produces 1 chunk with empty content.

        The chunker returns [] for empty text, but add() falls back to
        chunks = [text] when no chunks are produced (core.py line 190).
        """
        dag = ragdag.init(str(tmp_path))
        empty_doc = tmp_path / "empty.md"
        empty_doc.write_text("")

        result = dag.add(str(empty_doc))
        # Empty file still counts as 1 file processed with 1 chunk
        assert result["files"] == 1
        assert result["chunks"] == 1

    def test_reingestion_preserves_manual_edges(self, tmp_path):
        """Add file, manually create a 'references' edge, re-add same file
        with changes. The manual 'references' edge should still exist.

        _create_chunk_edges only removes lines containing
        '\\t{source_path}\\tchunked_from' so manual edges with different
        edge types are preserved.
        """
        dag = ragdag.init(str(tmp_path))
        store = tmp_path / ".ragdag"

        # Add initial file
        doc = tmp_path / "manual_edge.md"
        doc.write_text("# Version 1\n\nOriginal manual edge content.\n")
        dag.add(str(doc))

        # Find the chunk path
        chunks = list(store.rglob("*.txt"))
        chunk_paths = [
            f for f in chunks
            if not any(p.startswith(".") for p in f.relative_to(store).parts)
        ]
        assert len(chunk_paths) >= 1
        chunk_rel = str(chunk_paths[0].relative_to(store))

        # Manually add a 'references' edge
        dag.link(chunk_rel, "some/external/target.txt", "references")

        # Verify the manual edge exists
        edges_text = (store / ".edges").read_text()
        assert f"{chunk_rel}\tsome/external/target.txt\treferences" in edges_text

        # Re-add the file with changed content
        doc.write_text("# Version 2\n\nChanged manual edge content.\n")
        result = dag.add(str(doc))
        assert result["files"] == 1

        # The manual 'references' edge should still exist
        edges_after = (store / ".edges").read_text()
        assert "some/external/target.txt\treferences" in edges_after

    def test_add_embed_failure_stores_chunks(self, tmp_path):
        """Even when embedding fails (provider=none), chunks are still stored.

        With provider=none, _embed_chunks returns early without doing anything,
        but chunks should already be written to disk before embedding is attempted.
        """
        dag = ragdag.init(str(tmp_path))
        store = tmp_path / ".ragdag"

        # provider=none is the default -- embedding will be skipped
        doc = tmp_path / "embed_fail.md"
        doc.write_text("# Embed Fail\n\nChunks should exist even without embeddings.\n")
        result = dag.add(str(doc))

        assert result["files"] == 1
        assert result["chunks"] >= 1

        # Verify chunks exist on disk
        chunk_files = []
        for f in store.rglob("*.txt"):
            if not any(p.startswith(".") for p in f.relative_to(store).parts):
                chunk_files.append(f)
        assert len(chunk_files) >= 1, "Chunks should be stored even without embeddings"

        # Verify no embeddings.bin exists (since provider=none)
        embed_files = list(store.rglob("embeddings.bin"))
        assert len(embed_files) == 0, "No embedding files should exist with provider=none"

    def test_add_flat_no_domain_subdir(self, tmp_path):
        """Add a file without domain parameter stores chunks directly
        under .ragdag/<doc_name>/ without an extra domain directory level.

        When domain is not specified (None), add() uses:
            target_dir = self._store / doc_name
        instead of:
            target_dir = self._store / file_domain / doc_name
        """
        dag = ragdag.init(str(tmp_path))
        store = tmp_path / ".ragdag"

        doc = tmp_path / "flatdoc.md"
        doc.write_text("# Flat Document\n\nStored without domain.\n")
        dag.add(str(doc))  # No domain parameter

        # Chunks should be directly under .ragdag/flatdoc/
        doc_dir = store / "flatdoc"
        assert doc_dir.exists(), "Doc dir should exist at .ragdag/flatdoc/"
        chunks = list(doc_dir.glob("*.txt"))
        assert len(chunks) >= 1, "Should have at least one chunk"

        # Verify there's no extra domain level wrapping the doc dir.
        # The parent of the doc dir should be the store itself.
        assert doc_dir.parent == store, (
            f"Doc dir parent should be store root, not {doc_dir.parent}"
        )

    def test_add_domain_auto_unsorted(self, tmp_path):
        """Add a file with domain='auto' when no rules match sends it
        to 'unsorted/' domain.

        When domain='auto' and _apply_domain_rules returns empty string,
        add() falls back to 'unsorted' (core.py line 196).
        """
        dag = ragdag.init(str(tmp_path))
        store = tmp_path / ".ragdag"

        # Create a file whose path has no matching domain rules
        doc = tmp_path / "random_topic.md"
        doc.write_text("# Random Topic\n\nNothing matches any domain rule.\n")

        # domain='auto' with default empty rules -> should go to 'unsorted'
        dag.add(str(doc), domain="auto")

        unsorted_dir = store / "unsorted"
        assert unsorted_dir.exists(), "Expected 'unsorted' domain directory"
        chunks = list(unsorted_dir.rglob("*.txt"))
        assert len(chunks) >= 1, "Should have chunks under unsorted/"

    def test_add_binary_file_handled(self, tmp_path):
        """Adding a non-text file (e.g., .png) should not crash.

        The add() method calls _parse_file which falls back to
        read_text(errors='replace') for unrecognized extensions. Even
        with binary content, this should not raise an exception.
        """
        dag = ragdag.init(str(tmp_path))

        # Create a fake binary file with .png extension
        binary_file = tmp_path / "image.png"
        binary_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50 + b"\xff" * 50)

        # Should not crash -- either skips or stores as text
        result = dag.add(str(binary_file))
        # The file should be processed (read_text with errors='replace')
        assert result["files"] == 1 or result["skipped"] == 0
        # No exception means the test passes

    def test_processed_records_source_hash(self, tmp_path):
        """After adding a file, .processed contains source path and content hash.

        _record_processed writes: source_path\\tcontent_hash\\tdomain\\ttimestamp
        """
        dag = ragdag.init(str(tmp_path))
        store = tmp_path / ".ragdag"

        doc = tmp_path / "hashcheck.md"
        doc.write_text("# Hash Check\n\nVerify processed records.\n")
        dag.add(str(doc))

        processed_text = (store / ".processed").read_text()
        abs_path = str(doc.resolve())

        # Should contain the absolute source path
        assert abs_path in processed_text, (
            f"Expected source path '{abs_path}' in .processed"
        )

        # Should contain a SHA256 hash (64 hex characters)
        import re
        # Find tab-separated line with the source path
        for line in processed_text.splitlines():
            if line.startswith(abs_path):
                parts = line.split("\t")
                assert len(parts) >= 2, f"Expected tab-separated fields, got: {line}"
                content_hash = parts[1]
                assert re.match(r"^[a-f0-9]{64}$", content_hash), (
                    f"Expected SHA256 hash, got: {content_hash}"
                )
                break
        else:
            pytest.fail(f"Source path not found in .processed: {abs_path}")


class TestSearch:
    """Tests for keyword search."""

    def test_search_finds_matching_chunks(self, tmp_path):
        """Search should find chunks containing query terms."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "auth.md"
        doc.write_text("# Authentication\n\nOAuth2 login flow with JWT tokens.\n")
        dag.add(str(doc))

        results = dag.search("OAuth2 tokens", mode="keyword")
        assert len(results) >= 1
        assert any("OAuth2" in r.content for r in results)

    def test_search_no_results(self, tmp_path):
        """Search with no matches returns empty list."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "test.md"
        doc.write_text("# Hello\n\nWorld.\n")
        dag.add(str(doc))

        results = dag.search("zyxwvutsrq", mode="keyword")
        assert len(results) == 0

    def test_search_with_domain_filter(self, tmp_path):
        """Search with domain filter only searches that domain."""
        dag = ragdag.init(str(tmp_path))
        (tmp_path / "a.md").write_text("# Auth\n\npassword reset flow\n")
        (tmp_path / "b.md").write_text("# Deploy\n\npassword in env vars\n")
        dag.add(str(tmp_path / "a.md"), domain="auth")
        dag.add(str(tmp_path / "b.md"), domain="deploy")

        results = dag.search("password", mode="keyword", domain="auth")
        for r in results:
            assert r.path.startswith("auth/")

    def test_search_respects_top_k(self, tmp_path):
        """Search should return at most top_k results."""
        dag = ragdag.init(str(tmp_path))
        for i in range(20):
            f = tmp_path / f"doc{i}.md"
            f.write_text(f"# Doc {i}\n\nkeyword content here\n")
            dag.add(str(f))

        results = dag.search("keyword", mode="keyword", top=5)
        assert len(results) <= 5

    def test_keyword_search_case_insensitive(self, tmp_path):
        """'OAuth2' query matches 'oauth2' in content (case-insensitive).

        _keyword_search() lowercases both query and content before matching
        (core.py lines 651-652: query_lower = query.lower(), content_lower = content.lower()).
        """
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "auth.md"
        doc.write_text("# Auth\n\noauth2 login flow with jwt tokens.\n")
        dag.add(str(doc))

        # Search with different casing than the document
        results = dag.search("OAuth2", mode="keyword")
        assert len(results) >= 1
        assert any("oauth2" in r.content.lower() for r in results)

        # Also verify uppercase query works
        results_upper = dag.search("OAUTH2", mode="keyword")
        assert len(results_upper) >= 1

    def test_keyword_results_ordered_by_score(self, tmp_path):
        """Results come back sorted by score descending.

        _keyword_search() sorts results: results.sort(key=lambda r: r.score, reverse=True)
        (core.py line 679).
        """
        dag = ragdag.init(str(tmp_path))
        # Create docs with varying keyword density
        (tmp_path / "low.md").write_text("# Low\n\nSome other topic entirely.\nWith one keyword mention.\n")
        (tmp_path / "high.md").write_text("# High\n\nkeyword keyword keyword keyword keyword.\n")
        dag.add(str(tmp_path / "low.md"))
        dag.add(str(tmp_path / "high.md"))

        results = dag.search("keyword", mode="keyword")
        assert len(results) >= 2
        # Verify descending score order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_keyword_search_empty_store(self, tmp_path):
        """Search on store with no documents returns [].

        When no *.txt files exist in the store, rglob finds nothing and
        _keyword_search returns an empty list.
        """
        dag = ragdag.init(str(tmp_path))
        results = dag.search("anything", mode="keyword")
        assert results == []

    def test_search_result_fields(self, tmp_path):
        """SearchResult objects have .path, .score, .content, .domain attributes.

        SearchResult is a dataclass with: path (str), score (float),
        content (str), domain (str).
        """
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "fields.md"
        doc.write_text("# Fields\n\nTest search result field population.\n")
        dag.add(str(doc))

        results = dag.search("fields search", mode="keyword")
        assert len(results) >= 1
        r = results[0]

        # Verify all expected attributes exist and have correct types
        assert isinstance(r.path, str)
        assert isinstance(r.score, float)
        assert isinstance(r.content, str)
        assert isinstance(r.domain, str)
        assert r.score > 0
        assert len(r.content) > 0
        assert len(r.path) > 0

    def test_keyword_search_multiword(self, tmp_path):
        """Search with 'JWT tokens' should match a document containing both words.

        _keyword_search splits the query into words and sums occurrences of
        each word in the content (core.py line 668: match_count = sum(...)).
        """
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "jwt_doc.md"
        doc.write_text("# JWT Auth\n\nJWT tokens are used for stateless authentication.\n")
        dag.add(str(doc))

        results = dag.search("JWT tokens", mode="keyword")
        assert len(results) >= 1
        # The matching chunk should contain both words
        matched = results[0]
        assert "jwt" in matched.content.lower()
        assert "tokens" in matched.content.lower()

    def test_keyword_short_words_ignored(self, tmp_path):
        """Single-character words in query should be filtered out.

        _keyword_search filters words: words = [w for w in query_lower.split() if len(w) >= 2]
        (core.py line 652). Words shorter than 2 characters are dropped.
        """
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "auth_doc.md"
        doc.write_text("# Auth\n\nauthentication with secure passwords.\n")
        dag.add(str(doc))

        # Query with single-char words "a" and "b" that should be ignored
        results = dag.search("a b authentication", mode="keyword")
        assert len(results) >= 1
        # Should match on "authentication" only
        assert any("authentication" in r.content.lower() for r in results)

        # Verify that a query of ONLY short words returns nothing
        # (since all words are filtered out, no matches possible)
        results_short = dag.search("a b c", mode="keyword")
        assert len(results_short) == 0, (
            "Query with only single-char words should return no results"
        )


class TestAsk:
    """Tests for question answering."""

    def test_ask_returns_context_no_llm(self, tmp_path):
        """ask() with no LLM configured returns context with sources."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "auth.md"
        doc.write_text("# Auth\n\nJWT tokens expire after 24 hours.\n")
        dag.add(str(doc))

        result = dag.ask("How long do JWT tokens last?", use_llm=False)
        assert result.answer is None
        assert "JWT" in result.context
        assert len(result.sources) >= 1

    def test_ask_no_results(self, tmp_path):
        """ask() with no matching documents returns empty result."""
        dag = ragdag.init(str(tmp_path))
        result = dag.ask("nonexistent topic zyxwv")
        assert result.context == ""
        assert result.sources == []


class TestGraph:
    """Tests for graph operations."""

    def test_graph_stats(self, tmp_path):
        """graph() returns correct counts."""
        dag = ragdag.init(str(tmp_path))
        (tmp_path / "doc.md").write_text("# Test\n\nContent.\n")
        dag.add(str(tmp_path / "doc.md"), domain="test")

        stats = dag.graph()
        assert stats.domains >= 1
        assert stats.chunks >= 1
        assert stats.edges >= 1  # chunked_from edges
        assert "chunked_from" in stats.edge_types

    def test_graph_with_domain_filter(self, tmp_path):
        """graph(domain=X) only counts that domain."""
        dag = ragdag.init(str(tmp_path))
        (tmp_path / "a.md").write_text("# A\n\nContent A.\n")
        (tmp_path / "b.md").write_text("# B\n\nContent B.\n")
        dag.add(str(tmp_path / "a.md"), domain="alpha")
        dag.add(str(tmp_path / "b.md"), domain="beta")

        stats = dag.graph(domain="alpha")
        assert stats.domains == 1


class TestNeighbors:
    """Tests for neighbor lookup."""

    def test_neighbors_outgoing(self, tmp_path):
        """neighbors() finds outgoing edges."""
        dag = ragdag.init(str(tmp_path))
        dag.link("a/01.txt", "b/01.txt", "references")

        neighbors = dag.neighbors("a/01.txt")
        assert len(neighbors) >= 1
        outgoing = [n for n in neighbors if n["direction"] == "outgoing"]
        assert any(n["node"] == "b/01.txt" for n in outgoing)

    def test_neighbors_incoming(self, tmp_path):
        """neighbors() finds incoming edges."""
        dag = ragdag.init(str(tmp_path))
        dag.link("a/01.txt", "b/01.txt", "references")

        neighbors = dag.neighbors("b/01.txt")
        incoming = [n for n in neighbors if n["direction"] == "incoming"]
        assert any(n["node"] == "a/01.txt" for n in incoming)

    def test_neighbors_includes_metadata(self, tmp_path):
        """After creating an edge with metadata, neighbors() returns it.

        link() writes: source\\ttarget\\tedge_type\\t\\n (empty metadata by default).
        neighbors() parses: metadata = parts[3] if len(parts) > 3 else ""
        (core.py line 893). We use link() which writes an empty metadata field,
        then verify the neighbor dict includes the 'metadata' key.
        """
        dag = ragdag.init(str(tmp_path))
        store = tmp_path / ".ragdag"

        # link() creates edge with empty metadata field
        dag.link("src/chunk.txt", "dst/chunk.txt", "related_to")

        neighbors = dag.neighbors("src/chunk.txt")
        assert len(neighbors) >= 1

        # Find the outgoing neighbor to dst/chunk.txt
        found = False
        for n in neighbors:
            if n["node"] == "dst/chunk.txt":
                assert "metadata" in n, (
                    "Neighbor dict should include 'metadata' key"
                )
                assert isinstance(n["metadata"], str), (
                    "metadata should be a string"
                )
                found = True
                break
        assert found, "Expected to find dst/chunk.txt in neighbors"

        # Now manually write an edge with actual metadata content
        edges_file = store / ".edges"
        with open(edges_file, "a") as f:
            f.write("src/chunk.txt\tother/chunk.txt\treferences\tweight=0.9\n")

        neighbors2 = dag.neighbors("src/chunk.txt")
        meta_neighbor = [n for n in neighbors2 if n["node"] == "other/chunk.txt"]
        assert len(meta_neighbor) == 1
        assert meta_neighbor[0]["metadata"] == "weight=0.9"


class TestTrace:
    """Tests for provenance tracing."""

    def test_trace_follows_chunked_from(self, tmp_path):
        """trace() follows chunked_from edges to origin."""
        dag = ragdag.init(str(tmp_path))
        (tmp_path / "doc.md").write_text("# Test\n\nSome content.\n")
        dag.add(str(tmp_path / "doc.md"), domain="test")

        # Find a chunk path
        chunks = list((tmp_path / ".ragdag" / "test").rglob("*.txt"))
        assert len(chunks) >= 1
        chunk_rel = str(chunks[0].relative_to(tmp_path / ".ragdag"))

        chain = dag.trace(chunk_rel)
        assert len(chain) >= 2  # chunk -> origin
        assert chain[0]["edge_type"] == "chunked_from"
        assert chain[-1]["parent"] is None  # origin

    def test_trace_no_edges(self, tmp_path):
        """trace() on node with no edges returns single origin entry."""
        dag = ragdag.init(str(tmp_path))
        chain = dag.trace("nonexistent/node")
        assert len(chain) == 1
        assert chain[0]["parent"] is None


class TestLink:
    """Tests for manual edge creation."""

    def test_link_creates_edge(self, tmp_path):
        """link() appends edge to .edges file."""
        dag = ragdag.init(str(tmp_path))
        dag.link("a/01.txt", "b/01.txt", "references")

        edges_content = (tmp_path / ".ragdag" / ".edges").read_text()
        assert "a/01.txt\tb/01.txt\treferences" in edges_content

    def test_link_default_edge_type(self, tmp_path):
        """link() defaults to 'references' edge type."""
        dag = ragdag.init(str(tmp_path))
        dag.link("x", "y")

        edges_content = (tmp_path / ".ragdag" / ".edges").read_text()
        assert "x\ty\treferences" in edges_content
