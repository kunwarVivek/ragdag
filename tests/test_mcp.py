"""Tests for the MCP server at server/mcp.py.

Exercises all tool functions: ragdag_search, ragdag_ask, ragdag_add,
ragdag_graph, ragdag_neighbors, ragdag_trace.

Uses a real ragdag store in tmp directories with mock for _get_dag().

The @mcp.tool decorator wraps async functions into FunctionTool objects.
The original async function is accessible via the .fn attribute.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root and sdk to path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "sdk"))

import ragdag
import server.mcp as mcp_module

# Extract the underlying async functions from FunctionTool wrappers.
# The @mcp.tool decorator replaces each function with a FunctionTool instance
# whose .fn attribute holds the original async callable.
_search_fn = mcp_module.ragdag_search.fn
_ask_fn = mcp_module.ragdag_ask.fn
_add_fn = mcp_module.ragdag_add.fn
_graph_fn = mcp_module.ragdag_graph.fn
_neighbors_fn = mcp_module.ragdag_neighbors.fn
_trace_fn = mcp_module.ragdag_trace.fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_store(tmp_path):
    """Create a temporary ragdag store with default config."""
    ragdag.init(str(tmp_path))
    return tmp_path


@pytest.fixture()
def dag(tmp_store):
    """Return a RagDag instance on the temp store."""
    return ragdag.open(str(tmp_store))


@pytest.fixture()
def populated_store(tmp_store):
    """Store with two docs in different domains pre-added."""
    auth_doc = tmp_store / "auth.md"
    auth_doc.write_text(
        "# Authentication\n\n"
        "OAuth2 login flow with JWT tokens.\n"
        "Tokens expire after 24 hours.\n"
    )

    deploy_doc = tmp_store / "deploy.md"
    deploy_doc.write_text(
        "# Deployment\n\n"
        "Use docker-compose to deploy services.\n"
        "Environment variables go in .env file.\n"
    )

    dag = ragdag.open(str(tmp_store))
    dag.add(str(auth_doc), domain="auth")
    dag.add(str(deploy_doc), domain="deploy")

    # Add manual edge for neighbors/trace tests
    dag.link("auth/authentication/01.txt", "deploy/deployment/01.txt", "related_to")

    return tmp_store


@pytest.fixture()
def populated_dag(populated_store):
    """Return a RagDag instance backed by the populated store."""
    return ragdag.open(str(populated_store))


# ---------------------------------------------------------------------------
# 1. test_mcp_search (CRITICAL)
# ---------------------------------------------------------------------------


class TestMcpSearch:

    def test_mcp_search_keyword(self, populated_dag):
        """Search with keyword mode returns formatted results with score."""
        with patch.object(mcp_module, "_get_dag", return_value=populated_dag):
            result = _run(_search_fn(
                query="OAuth2 JWT tokens",
                mode="keyword",
            ))
        assert isinstance(result, str)
        assert "score:" in result
        # Should contain numbered results
        assert result.startswith("1.")
        # Should reference an auth-related chunk
        assert "auth" in result.lower()

    def test_mcp_search_default_mode(self, populated_dag):
        """Search with default (hybrid) mode falls back to keyword when provider=none."""
        with patch.object(mcp_module, "_get_dag", return_value=populated_dag):
            result = _run(_search_fn(
                query="docker deploy",
            ))
        assert isinstance(result, str)
        assert "score:" in result
        assert "deploy" in result.lower()

    def test_mcp_search_with_domain_filter(self, populated_dag):
        """Search with domain filter restricts results to that domain."""
        with patch.object(mcp_module, "_get_dag", return_value=populated_dag):
            result = _run(_search_fn(
                query="docker deploy",
                mode="keyword",
                domain="deploy",
            ))
        assert isinstance(result, str)
        assert "deploy" in result.lower()

    def test_mcp_search_top_param(self, populated_dag):
        """Search with top=1 limits to a single result."""
        with patch.object(mcp_module, "_get_dag", return_value=populated_dag):
            result = _run(_search_fn(
                query="OAuth2 docker deploy tokens",
                mode="keyword",
                top=1,
            ))
        assert isinstance(result, str)
        # Should have exactly 1 numbered result
        assert "1." in result
        assert "2." not in result


# ---------------------------------------------------------------------------
# 2. test_mcp_ask (CRITICAL)
# ---------------------------------------------------------------------------


class TestMcpAsk:

    def test_mcp_ask_returns_context_and_sources(self, populated_dag):
        """Ask returns answer/context with Sources section."""
        with patch.object(mcp_module, "_get_dag", return_value=populated_dag):
            result = _run(_ask_fn(
                question="How long do JWT tokens last?",
            ))
        assert isinstance(result, str)
        # With llm provider=none, answer is None so context is used.
        # The MCP function returns answer_text which falls back to result.context.
        # Sources section should be present because search finds matching docs.
        assert "**Sources:**" in result
        # Should contain relevant content about tokens
        assert "token" in result.lower() or "jwt" in result.lower()

    def test_mcp_ask_with_domain(self, populated_dag):
        """Ask with domain filter returns domain-scoped results."""
        with patch.object(mcp_module, "_get_dag", return_value=populated_dag):
            result = _run(_ask_fn(
                question="How do I deploy?",
                domain="deploy",
            ))
        assert isinstance(result, str)
        assert "deploy" in result.lower()

    def test_mcp_ask_no_results(self, populated_dag):
        """Ask with nonsense question returns empty context (no Sources)."""
        with patch.object(mcp_module, "_get_dag", return_value=populated_dag):
            result = _run(_ask_fn(
                question="zyxwvutsrqmlkjhg nonsense gibberish",
            ))
        assert isinstance(result, str)
        # With no search results, ask returns AskResult(answer=None, context="", sources=[])
        # MCP returns answer_text = result.answer or result.context = ""
        # No sources means no Sources section
        assert "**Sources:**" not in result


# ---------------------------------------------------------------------------
# 3. test_mcp_add
# ---------------------------------------------------------------------------


class TestMcpAdd:

    def test_mcp_add_ingests_file(self, dag, tmp_store):
        """Add ingests a file and returns JSON summary with files/chunks/skipped."""
        doc = tmp_store / "hello.md"
        doc.write_text("# Hello World\n\nThis is test content for ingestion.\n")

        with patch.object(mcp_module, "_get_dag", return_value=dag):
            result = _run(_add_fn(
                path=str(doc),
                domain="docs",
            ))
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["files"] == 1
        assert data["chunks"] >= 1
        assert data["skipped"] == 0

    def test_mcp_add_with_domain(self, dag, tmp_store):
        """Add with domain creates chunks under the domain directory."""
        doc = tmp_store / "notes.txt"
        doc.write_text("Some important notes about authentication.\n")

        with patch.object(mcp_module, "_get_dag", return_value=dag):
            result = _run(_add_fn(
                path=str(doc),
                domain="mynotes",
            ))
        data = json.loads(result)
        assert data["files"] == 1

        # Verify domain directory was created in the store
        domain_dir = tmp_store / ".ragdag" / "mynotes"
        assert domain_dir.exists()


# ---------------------------------------------------------------------------
# 4. test_mcp_graph
# ---------------------------------------------------------------------------


class TestMcpGraph:

    def test_mcp_graph_returns_stats(self, populated_dag):
        """Graph returns stats string with Domains/Chunks/Edges."""
        with patch.object(mcp_module, "_get_dag", return_value=populated_dag):
            result = _run(_graph_fn())
        assert isinstance(result, str)
        assert "Domains:" in result
        assert "Documents:" in result
        assert "Chunks:" in result
        assert "Edges:" in result
        assert "Edge types:" in result

    def test_mcp_graph_with_domain_filter(self, populated_dag):
        """Graph with domain filter only counts that domain."""
        with patch.object(mcp_module, "_get_dag", return_value=populated_dag):
            result = _run(_graph_fn(domain="auth"))
        assert isinstance(result, str)
        assert "Domains: 1" in result

    def test_mcp_graph_nonexistent_domain(self, populated_dag):
        """Graph with nonexistent domain returns zero counts."""
        with patch.object(mcp_module, "_get_dag", return_value=populated_dag):
            result = _run(_graph_fn(domain="nonexistent"))
        assert isinstance(result, str)
        assert "Domains: 0" in result
        assert "Chunks: 0" in result


# ---------------------------------------------------------------------------
# 5. test_mcp_neighbors
# ---------------------------------------------------------------------------


class TestMcpNeighbors:

    def test_mcp_neighbors_returns_edges(self, populated_dag):
        """Neighbors returns formatted edge list with arrows."""
        with patch.object(mcp_module, "_get_dag", return_value=populated_dag):
            result = _run(_neighbors_fn(
                node="auth/authentication/01.txt",
            ))
        assert isinstance(result, str)
        assert "Neighbors of" in result
        # Should have outgoing edge to deploy chunk (added via dag.link)
        assert "related_to" in result

    def test_mcp_neighbors_incoming(self, populated_dag):
        """Neighbors shows incoming edges for target nodes."""
        with patch.object(mcp_module, "_get_dag", return_value=populated_dag):
            result = _run(_neighbors_fn(
                node="deploy/deployment/01.txt",
            ))
        assert isinstance(result, str)
        # Should show incoming edge from auth chunk
        assert "Neighbors of" in result

    def test_mcp_neighbors_no_edges(self, dag):
        """Neighbors for unknown node returns 'No neighbors found'."""
        with patch.object(mcp_module, "_get_dag", return_value=dag):
            result = _run(_neighbors_fn(
                node="nonexistent/node",
            ))
        assert isinstance(result, str)
        assert "No neighbors found" in result


# ---------------------------------------------------------------------------
# 6. test_mcp_trace
# ---------------------------------------------------------------------------


class TestMcpTrace:

    def test_mcp_trace_returns_provenance(self, populated_dag, populated_store):
        """Trace returns provenance chain for a chunk."""
        # Find a real chunk path by looking in the store
        store = populated_store / ".ragdag"
        chunks = [
            c for c in store.rglob("*.txt")
            if not c.name.startswith(".") and not c.name.startswith("_")
        ]
        assert len(chunks) >= 1, "Populated store should have chunks"

        chunk_rel = str(chunks[0].relative_to(store))
        with patch.object(mcp_module, "_get_dag", return_value=populated_dag):
            result = _run(_trace_fn(node=chunk_rel))
        assert isinstance(result, str)
        assert "Provenance of" in result
        # Should show chunked_from edge type
        assert "chunked_from" in result

    def test_mcp_trace_no_provenance(self, dag):
        """Trace for unknown node returns origin-only chain."""
        with patch.object(mcp_module, "_get_dag", return_value=dag):
            result = _run(_trace_fn(
                node="nonexistent/node",
            ))
        assert isinstance(result, str)
        assert "Provenance of" in result
        assert "origin" in result


# ---------------------------------------------------------------------------
# 7. test_mcp_error_handling
# ---------------------------------------------------------------------------


class TestMcpErrorHandling:

    def test_mcp_search_no_results(self, populated_dag):
        """Search with no matching results returns 'No results found.'"""
        with patch.object(mcp_module, "_get_dag", return_value=populated_dag):
            result = _run(_search_fn(
                query="zyxwvutsrqmlkjhg",
                mode="keyword",
            ))
        assert result == "No results found."

    def test_mcp_search_empty_store(self, dag):
        """Search on empty store returns 'No results found.'"""
        with patch.object(mcp_module, "_get_dag", return_value=dag):
            result = _run(_search_fn(
                query="anything at all",
                mode="keyword",
            ))
        assert result == "No results found."


# ---------------------------------------------------------------------------
# 8. test_mcp_store_context
# ---------------------------------------------------------------------------


class TestMcpStoreContext:

    def test_ragdag_store_env_controls_store(self, tmp_path):
        """RAGDAG_STORE env var controls which store _get_dag() uses."""
        # Create a store at a custom location
        store_path = tmp_path / "custom_store"
        store_path.mkdir()
        ragdag.init(str(store_path))

        old_env = os.environ.get("RAGDAG_STORE")
        try:
            os.environ["RAGDAG_STORE"] = str(store_path)
            dag = mcp_module._get_dag()
            assert dag._root == store_path.resolve()
        finally:
            if old_env is None:
                os.environ.pop("RAGDAG_STORE", None)
            else:
                os.environ["RAGDAG_STORE"] = old_env

    def test_ragdag_store_default_uses_cwd(self, monkeypatch):
        """Without RAGDAG_STORE, _get_dag() defaults to current directory."""
        monkeypatch.delenv("RAGDAG_STORE", raising=False)
        dag = mcp_module._get_dag()
        assert dag._root == Path(".").resolve()
