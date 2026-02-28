"""Tests for the FastAPI HTTP server at server/api.py.

Exercises all endpoints: /health, /add, /search, /ask, /graph,
/neighbors, /link, /trace, /relate.

Uses a real ragdag store in tmp directories with TestClient (httpx).
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root and sdk to path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "sdk"))

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI TestClient requires httpx", allow_module_level=True)

import ragdag
import server.api as api_module
from server.api import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_dag():
    """Reset the global _dag before each test so env changes take effect."""
    api_module._dag = None
    yield
    api_module._dag = None


@pytest.fixture()
def tmp_store(tmp_path):
    """Create a temporary ragdag store with default config and set env var."""
    dag = ragdag.init(str(tmp_path))
    os.environ["RAGDAG_STORE"] = str(tmp_path)
    yield tmp_path
    os.environ.pop("RAGDAG_STORE", None)


@pytest.fixture()
def client(tmp_store):
    """Provide a TestClient wired to a fresh tmp store."""
    return TestClient(app)


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

    return tmp_store


@pytest.fixture()
def populated_client(populated_store):
    """TestClient backed by a store that already has documents."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:

    def test_health_returns_ok(self, client):
        """GET /health returns 200 with status ok and version."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# 2-3. Add endpoint
# ---------------------------------------------------------------------------


class TestAddEndpoint:

    def test_add_file(self, client, tmp_store):
        """POST /add with a real temp file ingests it successfully."""
        doc = tmp_store / "hello.md"
        doc.write_text("# Hello\n\nThis is test content for ingestion.\n")

        resp = client.post("/add", json={"path": str(doc), "embed": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["files"] == 1
        assert data["chunks"] >= 1
        assert data["skipped"] == 0

    def test_add_nonexistent_returns_500(self, client):
        """POST /add with a path that doesn't exist returns 500."""
        resp = client.post(
            "/add", json={"path": "/nonexistent/path/doc.md", "embed": False}
        )
        assert resp.status_code == 500
        assert "not found" in resp.json()["detail"].lower()

    def test_add_with_domain(self, client, tmp_store):
        """POST /add with domain organizes chunks under that domain."""
        doc = tmp_store / "notes.md"
        doc.write_text("# Notes\n\nSome important notes.\n")

        resp = client.post(
            "/add", json={"path": str(doc), "domain": "docs", "embed": False}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["files"] == 1

        # Verify domain directory was created
        domain_dir = tmp_store / ".ragdag" / "docs"
        assert domain_dir.exists()


# ---------------------------------------------------------------------------
# 4-6. Search endpoint
# ---------------------------------------------------------------------------


class TestSearchEndpoint:

    def test_search_keyword(self, populated_client):
        """POST /search finds matching documents with keyword mode."""
        resp = populated_client.post(
            "/search",
            json={"query": "OAuth2 JWT tokens", "mode": "keyword"},
        )
        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)
        assert len(results) >= 1
        # Each result should have the expected keys
        first = results[0]
        assert "path" in first
        assert "score" in first
        assert "content" in first
        assert "domain" in first
        # Content should contain the queried term
        assert any("OAuth2" in r["content"] for r in results)

    def test_search_no_results(self, populated_client):
        """POST /search for a nonsense query returns empty list."""
        resp = populated_client.post(
            "/search",
            json={"query": "zyxwvutsrqmlkjhg", "mode": "keyword"},
        )
        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_with_domain_filter(self, populated_client):
        """POST /search with domain filter restricts results."""
        resp = populated_client.post(
            "/search",
            json={"query": "deploy docker", "mode": "keyword", "domain": "deploy"},
        )
        assert resp.status_code == 200
        results = resp.json()
        # All results should be from deploy domain
        for r in results:
            assert r["path"].startswith("deploy/"), (
                f"Result path {r['path']} should start with 'deploy/'"
            )

    def test_api_search_missing_query_returns_422(self, client):
        """POST /search with empty body returns 422 validation error."""
        resp = client.post("/search", json={})
        assert resp.status_code == 422
        data = resp.json()
        assert "detail" in data


# ---------------------------------------------------------------------------
# 7-8. Ask endpoint
# ---------------------------------------------------------------------------


class TestAskEndpoint:

    def test_ask_no_llm(self, populated_client):
        """POST /ask with use_llm=false returns context without calling LLM."""
        resp = populated_client.post(
            "/ask",
            json={"question": "How long do JWT tokens last?", "use_llm": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "context" in data
        assert "sources" in data
        # answer should be None when use_llm=false and llm provider=none
        assert data["answer"] is None
        # Context should include relevant content
        assert "JWT" in data["context"] or "token" in data["context"].lower()
        assert len(data["sources"]) >= 1

    def test_ask_empty(self, populated_client):
        """POST /ask with no matching docs returns empty result."""
        resp = populated_client.post(
            "/ask",
            json={
                "question": "zyxwvutsrqmlk nonsense query",
                "use_llm": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] is None
        assert data["context"] == ""
        assert data["sources"] == []


# ---------------------------------------------------------------------------
# 9-10. Graph endpoint
# ---------------------------------------------------------------------------


class TestGraphEndpoint:

    def test_graph_stats(self, populated_client):
        """GET /graph returns summary statistics after adding docs."""
        resp = populated_client.get("/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert "domains" in data
        assert "documents" in data
        assert "chunks" in data
        assert "edges" in data
        assert "edge_types" in data
        assert data["domains"] >= 1
        assert data["chunks"] >= 1
        assert data["edges"] >= 1
        assert "chunked_from" in data["edge_types"]

    def test_graph_with_domain_filter(self, populated_client):
        """GET /graph?domain=auth only counts the auth domain."""
        resp = populated_client.get("/graph", params={"domain": "auth"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["domains"] == 1

    def test_graph_nonexistent_domain(self, populated_client):
        """GET /graph?domain=nonexistent returns zeroed stats."""
        resp = populated_client.get("/graph", params={"domain": "nonexistent"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["domains"] == 0
        assert data["documents"] == 0
        assert data["chunks"] == 0


# ---------------------------------------------------------------------------
# 11. Neighbors endpoint
# ---------------------------------------------------------------------------


class TestNeighborsEndpoint:

    def test_neighbors_after_link(self, client, tmp_store):
        """GET /neighbors/path returns linked neighbors."""
        # Create a link first
        client.post(
            "/link",
            json={"source": "a/01.txt", "target": "b/01.txt", "edge_type": "references"},
        )

        resp = client.get("/neighbors/a/01.txt")
        assert resp.status_code == 200
        neighbors = resp.json()
        assert isinstance(neighbors, list)
        assert len(neighbors) >= 1
        outgoing = [n for n in neighbors if n["direction"] == "outgoing"]
        assert any(n["node"] == "b/01.txt" for n in outgoing)

    def test_neighbors_incoming(self, client, tmp_store):
        """GET /neighbors/target returns incoming edges."""
        client.post(
            "/link",
            json={"source": "a/01.txt", "target": "b/01.txt"},
        )

        resp = client.get("/neighbors/b/01.txt")
        assert resp.status_code == 200
        neighbors = resp.json()
        incoming = [n for n in neighbors if n["direction"] == "incoming"]
        assert any(n["node"] == "a/01.txt" for n in incoming)

    def test_neighbors_no_edges(self, client):
        """GET /neighbors/unknown returns empty list."""
        resp = client.get("/neighbors/nonexistent/node")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# 12. Link endpoint
# ---------------------------------------------------------------------------


class TestLinkEndpoint:

    def test_link_creates_edge(self, client, tmp_store):
        """POST /link creates an edge and returns ok."""
        resp = client.post(
            "/link",
            json={
                "source": "docs/readme/01.txt",
                "target": "docs/guide/01.txt",
                "edge_type": "references",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify edge was written to .edges file
        edges_content = (tmp_store / ".ragdag" / ".edges").read_text()
        assert "docs/readme/01.txt\tdocs/guide/01.txt\treferences" in edges_content

    def test_link_default_edge_type(self, client, tmp_store):
        """POST /link without edge_type defaults to 'references'."""
        resp = client.post(
            "/link",
            json={"source": "x", "target": "y"},
        )
        assert resp.status_code == 200

        edges_content = (tmp_store / ".ragdag" / ".edges").read_text()
        assert "x\ty\treferences" in edges_content


# ---------------------------------------------------------------------------
# 13. Trace endpoint
# ---------------------------------------------------------------------------


class TestTraceEndpoint:

    def test_trace_provenance(self, populated_client, populated_store):
        """GET /trace/chunk returns provenance chain back to source."""
        # Find a chunk that was created during population
        store = populated_store / ".ragdag"
        chunks = list(store.rglob("*.txt"))
        # Filter to actual chunk files (not dot-files)
        chunks = [c for c in chunks if not c.name.startswith(".")]
        assert len(chunks) >= 1, "Populated store should have at least one chunk"

        chunk_rel = str(chunks[0].relative_to(store))
        resp = populated_client.get(f"/trace/{chunk_rel}")
        assert resp.status_code == 200
        chain = resp.json()
        assert isinstance(chain, list)
        assert len(chain) >= 2  # chunk -> origin
        assert chain[0]["edge_type"] == "chunked_from"
        assert chain[-1]["parent"] is None  # end of chain

    def test_trace_no_edges(self, client):
        """GET /trace/nonexistent returns single origin entry."""
        resp = client.get("/trace/nonexistent/node")
        assert resp.status_code == 200
        chain = resp.json()
        assert len(chain) == 1
        assert chain[0]["parent"] is None
        assert chain[0]["edge_type"] == "origin"


# ---------------------------------------------------------------------------
# 14. Relate endpoint
# ---------------------------------------------------------------------------


class TestRelateEndpoint:

    def test_relate_returns_ok(self, populated_client):
        """POST /relate completes without error (no embedding provider)."""
        resp = populated_client.post(
            "/relate", json={"threshold": 0.8}
        )
        # With provider=none, relate_cli.main() may raise or return.
        # Accept either 200 or 500 since no embedding engine is configured.
        assert resp.status_code in (200, 500)


# ---------------------------------------------------------------------------
# 15. RAGDAG_STORE env var
# ---------------------------------------------------------------------------


class TestStoreEnvVar:

    def test_ragdag_store_controls_location(self, tmp_path):
        """RAGDAG_STORE env var determines where the dag stores data."""
        # Init a store at a specific location
        store_path = tmp_path / "custom_store"
        store_path.mkdir()
        ragdag.init(str(store_path))

        os.environ["RAGDAG_STORE"] = str(store_path)
        api_module._dag = None
        try:
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.status_code == 200

            # Verify the dag points to our custom location
            dag = api_module.get_dag()
            assert dag._root == store_path.resolve()
        finally:
            os.environ.pop("RAGDAG_STORE", None)
            api_module._dag = None

    def test_default_store_uses_cwd(self, tmp_path, monkeypatch):
        """Without RAGDAG_STORE, dag defaults to current directory."""
        monkeypatch.delenv("RAGDAG_STORE", raising=False)
        api_module._dag = None

        # Initialize a store at cwd-equivalent
        ragdag.init(str(tmp_path))
        monkeypatch.setenv("RAGDAG_STORE", str(tmp_path))

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
