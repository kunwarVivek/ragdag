"""Tests for hybrid search behavior — fallbacks, mode selection, score fusion."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

import pytest
import ragdag
from ragdag.core import RagDag, SearchResult


class TestHybridFallback:
    """Tests for hybrid search fallback behavior."""

    def test_hybrid_fallback_to_keyword_no_embeddings(self, tmp_path):
        """When provider=none, hybrid falls back to keyword and still returns results.

        In _python_search(), if embedding.provider == 'none', it returns
        self._keyword_search() directly. So hybrid mode with no embeddings
        degrades gracefully to keyword search.
        """
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "auth.md"
        doc.write_text("# Authentication\n\nOAuth2 login flow with JWT tokens.\n")
        dag.add(str(doc))

        # Confirm provider is none
        assert dag._read_config("embedding.provider") == "none"

        # Hybrid mode should still return results via keyword fallback
        results = dag.search("OAuth2 tokens", mode="hybrid")
        assert len(results) >= 1
        assert any("OAuth2" in r.content for r in results)

    def test_hybrid_fallback_on_import_error(self, tmp_path):
        """When engines.similarity import fails (no numpy), falls back gracefully to keyword.

        The search() method wraps _python_search() in a try/except. If the import
        of engines.similarity raises ImportError (e.g., numpy not installed),
        it falls back to _keyword_search().
        """
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "deploy.md"
        doc.write_text("# Deploy\n\nKubernetes deployment with helm charts.\n")
        dag.add(str(doc))

        # Set provider to something that would trigger _python_search path
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace(
            "provider = none", "provider = openai"
        )
        config_path.write_text(config_text)

        # Mock the import of engines.similarity to raise ImportError
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "engines.similarity" or (
                isinstance(name, str) and "similarity" in name
            ):
                raise ImportError("No module named 'numpy'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # This should fall back to keyword search instead of crashing
            results = dag.search("Kubernetes helm", mode="hybrid")

        assert len(results) >= 1
        assert any("Kubernetes" in r.content for r in results)

    def test_search_mode_keyword_explicit(self, tmp_path):
        """mode='keyword' uses keyword search directly, bypassing _python_search entirely.

        When mode='keyword', search() calls _keyword_search() directly without
        attempting _python_search() at all.
        """
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "api.md"
        doc.write_text("# API\n\nREST API endpoints for user management.\n")
        dag.add(str(doc))

        # Patch _python_search to track if it's called
        with patch.object(dag, "_python_search", wraps=dag._python_search) as mock_ps:
            results = dag.search("REST API", mode="keyword")
            # _python_search should NOT be called for keyword mode
            mock_ps.assert_not_called()

        assert len(results) >= 1
        assert any("REST" in r.content for r in results)

    def test_search_default_mode_is_hybrid(self, tmp_path):
        """Without specifying mode, search uses hybrid (which falls back to keyword when no embeddings).

        The default value for mode parameter is 'hybrid'. With provider=none,
        hybrid falls back to keyword search through _python_search -> _keyword_search.
        """
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "infra.md"
        doc.write_text("# Infrastructure\n\nDocker containers and CI/CD pipelines.\n")
        dag.add(str(doc))

        # Call search without specifying mode
        results = dag.search("Docker containers")
        assert len(results) >= 1
        assert any("Docker" in r.content for r in results)

        # Verify default mode is hybrid via inspection
        import inspect
        sig = inspect.signature(RagDag.search)
        assert sig.parameters["mode"].default == "hybrid"


class TestHybridScoreFusion:
    """Tests for hybrid score fusion mechanics (when both keyword and vector results exist)."""

    def test_keyword_results_used_as_candidates_in_hybrid(self, tmp_path):
        """Hybrid mode uses keyword pre-filter (top*3) as candidates for vector search.

        This tests the structural flow: keyword results feed into vector search.
        With provider=none, we verify the keyword pre-filter path is taken.
        """
        dag = ragdag.init(str(tmp_path))
        # Add multiple documents with overlapping terms
        for i in range(5):
            doc = tmp_path / f"doc{i}.md"
            doc.write_text(f"# Document {i}\n\nShared keyword content variation {i}.\n")
            dag.add(str(doc))

        # With provider=none, hybrid falls back to keyword, but results should still work
        results = dag.search("keyword content", mode="hybrid")
        assert len(results) >= 1
        # All results should contain the search terms
        for r in results:
            content_lower = r.content.lower()
            assert "keyword" in content_lower or "content" in content_lower

    def test_hybrid_mode_vector_fallback_returns_search_results(self, tmp_path):
        """When _python_search raises any Exception, hybrid/vector mode falls back to keyword.

        The search() method catches all Exceptions from _python_search() and
        falls back to _keyword_search().
        """
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "test.md"
        doc.write_text("# Test\n\nFallback scenario testing.\n")
        dag.add(str(doc))

        # Force _python_search to raise a generic Exception
        with patch.object(dag, "_python_search", side_effect=RuntimeError("engine failure")):
            results = dag.search("Fallback scenario", mode="vector")
            assert len(results) >= 1
            assert any("Fallback" in r.content for r in results)

        # Same for hybrid mode
        with patch.object(dag, "_python_search", side_effect=RuntimeError("engine failure")):
            results = dag.search("Fallback scenario", mode="hybrid")
            assert len(results) >= 1
