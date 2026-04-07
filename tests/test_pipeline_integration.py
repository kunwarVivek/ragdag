"""Integration tests for the BM25+RRF search pipeline."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

import pytest
import ragdag
from ragdag.core import RagDag, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_dag(tmp_path):
    """Init a ragdag store and return the RagDag instance."""
    return ragdag.init(str(tmp_path))


def _add_doc(tmp_path, name, content):
    """Write a markdown doc under tmp_path and return its Path."""
    doc = tmp_path / name
    doc.write_text(content)
    return doc


# ===========================================================================
# TestBM25Integration
# ===========================================================================


class TestBM25Integration:
    """Verify _keyword_search delegates to BM25 and IDF ranking works."""

    def test_keyword_search_uses_bm25_not_legacy(self, tmp_path):
        """mode='keyword' should delegate to engines.bm25.bm25_search."""
        dag = _init_dag(tmp_path)
        doc = _add_doc(tmp_path, "alpha.md", "The quick brown fox jumps over the lazy dog")
        dag.add(str(doc))

        with patch("engines.bm25.bm25_search", wraps=__import__("engines.bm25", fromlist=["bm25_search"]).bm25_search) as mock_bm25:
            results = dag.search("fox", mode="keyword")
            assert mock_bm25.called, "bm25_search was not called for mode='keyword'"

    def test_bm25_idf_in_sdk_pipeline(self, tmp_path):
        """A rare term should rank higher than a common term via BM25 IDF."""
        dag = _init_dag(tmp_path)

        # 'common' appears in all docs; 'xylophone' only in doc4
        _add_doc(tmp_path, "d1.md", "common word appears here in this document about nothing")
        _add_doc(tmp_path, "d2.md", "common word also appears in this second document text")
        _add_doc(tmp_path, "d3.md", "common word yet again in the third document right here")
        _add_doc(tmp_path, "d4.md", "common word plus xylophone is a rare musical instrument")

        for name in ["d1.md", "d2.md", "d3.md", "d4.md"]:
            dag.add(str(tmp_path / name))

        results = dag.search("xylophone", mode="keyword")
        assert len(results) >= 1
        # The doc containing 'xylophone' should rank first
        assert "d4" in results[0].path

    def test_bm25_legacy_fallback(self, tmp_path):
        """When engines.bm25 import fails, search falls back to legacy."""
        dag = _init_dag(tmp_path)
        doc = _add_doc(tmp_path, "fallback.md", "important data about legacy fallback testing")
        dag.add(str(doc))

        with patch.dict("sys.modules", {"engines.bm25": None}):
            # _keyword_search will get ImportError and fall back to _keyword_search_legacy
            results = dag.search("legacy", mode="keyword")
            assert len(results) >= 1
            assert any("fallback" in r.path for r in results)


# ===========================================================================
# TestRRFIntegration
# ===========================================================================


class TestRRFIntegration:
    """Verify hybrid search uses RRF fusion when embeddings are available."""

    def test_hybrid_uses_rrf_when_embeddings_available(self, tmp_path):
        """hybrid mode with an embedding provider should call reciprocal_rank_fusion."""
        dag = _init_dag(tmp_path)
        doc = _add_doc(tmp_path, "rrf_test.md", "neural networks and deep learning concepts")
        dag.add(str(doc))

        # Set provider to 'local' so _python_search doesn't bail to keyword
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("provider = none", "provider = local", 1)
        config_path.write_text(config_text)

        # Mock the engine and vector search to return fake results
        fake_engine = MagicMock()
        fake_engine.embed.return_value = [[0.1] * 1536]
        fake_engine.dimensions.return_value = 1536
        fake_engine.model_name.return_value = "test-model"

        with patch("engines.local_engine.LocalEngine", return_value=fake_engine), \
             patch("engines.similarity.search_vectors", return_value=[("rrf_test/01.txt", 0.9)]), \
             patch("engines.rrf.reciprocal_rank_fusion", wraps=__import__("engines.rrf", fromlist=["reciprocal_rank_fusion"]).reciprocal_rank_fusion) as mock_rrf:
            results = dag.search("neural", mode="hybrid")
            assert mock_rrf.called, "reciprocal_rank_fusion was not called during hybrid search"

    def test_hybrid_fallback_to_bm25_without_embeddings(self, tmp_path):
        """With provider=none, hybrid mode should fall back to BM25 keyword search."""
        dag = _init_dag(tmp_path)
        doc = _add_doc(tmp_path, "noembedding.md", "searching without any embedding provider enabled")
        dag.add(str(doc))

        # provider=none is the default -- hybrid should fall back to keyword (BM25)
        with patch("engines.bm25.bm25_search", wraps=__import__("engines.bm25", fromlist=["bm25_search"]).bm25_search) as mock_bm25:
            results = dag.search("searching", mode="hybrid")
            assert mock_bm25.called, "BM25 was not called as fallback when provider=none"
            assert len(results) >= 1


# ===========================================================================
# TestExplainIntegration
# ===========================================================================


class TestExplainIntegration:
    """Verify explain flag populates explain dicts correctly."""

    def test_explain_keyword_returns_bm25_scores(self, tmp_path):
        """mode='keyword' with explain=True should include 'bm25' in explain dict."""
        dag = _init_dag(tmp_path)
        doc = _add_doc(tmp_path, "explain_test.md", "explanation of bm25 scoring algorithm")
        dag.add(str(doc))

        results = dag.search("bm25", mode="keyword", explain=True)
        assert len(results) >= 1
        for r in results:
            assert r.explain is not None, "explain should not be None when explain=True"
            assert "bm25" in r.explain, f"explain dict missing 'bm25' key: {r.explain}"

    def test_explain_hybrid_fallback_returns_bm25(self, tmp_path):
        """With provider=none, hybrid+explain should have 'bm25' in explain dict."""
        dag = _init_dag(tmp_path)
        doc = _add_doc(tmp_path, "hybrid_explain.md", "hybrid mode with explain enabled")
        dag.add(str(doc))

        results = dag.search("hybrid", mode="hybrid", explain=True)
        assert len(results) >= 1
        for r in results:
            assert r.explain is not None
            assert "bm25" in r.explain, f"explain dict missing 'bm25' key: {r.explain}"

    def test_explain_false_returns_none(self, tmp_path):
        """Without explain, results should have explain=None."""
        dag = _init_dag(tmp_path)
        doc = _add_doc(tmp_path, "no_explain.md", "no explanation needed here")
        dag.add(str(doc))

        results = dag.search("explanation", mode="keyword", explain=False)
        assert len(results) >= 1
        for r in results:
            assert r.explain is None, f"explain should be None but got: {r.explain}"


# ===========================================================================
# TestRerankerIntegration
# ===========================================================================


class TestRerankerIntegration:
    """Verify reranker is triggered by config and disabled by default."""

    def test_rerank_config_triggers_reranker(self, tmp_path):
        """search.rerank=true in config should invoke engines.reranker.rerank."""
        dag = _init_dag(tmp_path)
        doc = _add_doc(tmp_path, "rerank_test.md", "document for reranker testing purposes")
        dag.add(str(doc))

        # Enable reranking and set provider to 'local'
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("provider = none", "provider = local", 1)
        config_text = config_text.replace("rerank = false", "rerank = true")
        config_path.write_text(config_text)

        fake_engine = MagicMock()
        fake_engine.embed.return_value = [[0.1] * 1536]
        fake_engine.dimensions.return_value = 1536
        fake_engine.model_name.return_value = "test-model"

        mock_rerank = MagicMock(return_value=[("rerank_test/01.txt", 0.95)])

        with patch("engines.local_engine.LocalEngine", return_value=fake_engine), \
             patch("engines.similarity.search_vectors", return_value=[("rerank_test/01.txt", 0.9)]), \
             patch("engines.reranker.rerank", mock_rerank):
            results = dag.search("reranker", mode="hybrid")
            assert mock_rerank.called, "rerank was not called when search.rerank=true"

    def test_rerank_disabled_by_default(self, tmp_path):
        """Default config has rerank=false; reranker should NOT be invoked."""
        dag = _init_dag(tmp_path)
        doc = _add_doc(tmp_path, "no_rerank.md", "document without reranking enabled")
        dag.add(str(doc))

        # Set provider to local so we go through _python_search hybrid path
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("provider = none", "provider = local", 1)
        config_path.write_text(config_text)

        fake_engine = MagicMock()
        fake_engine.embed.return_value = [[0.1] * 1536]
        fake_engine.dimensions.return_value = 1536
        fake_engine.model_name.return_value = "test-model"

        mock_rerank = MagicMock(return_value=[])

        with patch("engines.local_engine.LocalEngine", return_value=fake_engine), \
             patch("engines.similarity.search_vectors", return_value=[("no_rerank/01.txt", 0.9)]), \
             patch("engines.reranker.rerank", mock_rerank):
            results = dag.search("reranking", mode="hybrid")
            assert not mock_rerank.called, "rerank should NOT be called when rerank=false"


# ===========================================================================
# TestConfigDefaults
# ===========================================================================


class TestConfigDefaults:
    """Verify ragdag.init() creates correct search config defaults."""

    def test_init_creates_rerank_config(self, tmp_path):
        """After init(), .config should contain rerank=false and rerank_model."""
        dag = _init_dag(tmp_path)

        config_path = tmp_path / ".ragdag" / ".config"
        assert config_path.exists()
        config_text = config_path.read_text()

        assert "rerank = false" in config_text, "Default config missing 'rerank = false'"
        assert "rerank_model = cross-encoder/ms-marco-MiniLM-L-6-v2" in config_text, \
            "Default config missing rerank_model"


# ===========================================================================
# TestContentCacheIntegration
# ===========================================================================


class TestContentCacheIntegration:
    """Verify content-addressable caching skips re-embedding unchanged chunks."""

    def test_readd_same_content_skips_embedding(self, tmp_path):
        """Adding the same doc twice should skip embedding on the second add.

        The _embed_chunks method calls engine.embed(texts). On the second add
        of identical content, the file-level dedup in _is_processed should
        skip the file entirely, meaning embed() is not called again.
        """
        dag = _init_dag(tmp_path)

        # Set provider to local so embedding path is exercised
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("provider = none", "provider = local", 1)
        config_path.write_text(config_text)

        doc = _add_doc(tmp_path, "cached.md", "content that should be cached after first embed")

        fake_engine = MagicMock()
        fake_engine.embed.return_value = [[0.1] * 1536]
        fake_engine.dimensions.return_value = 1536
        fake_engine.model_name.return_value = "test-model"

        with patch("engines.local_engine.LocalEngine", return_value=fake_engine):
            # First add: should embed
            result1 = dag.add(str(doc))
            first_embed_count = fake_engine.embed.call_count
            assert first_embed_count >= 1, "embed() should be called on first add"

            # Second add of same file (same content hash): should skip
            result2 = dag.add(str(doc))
            second_embed_count = fake_engine.embed.call_count
            assert second_embed_count == first_embed_count, \
                f"embed() called {second_embed_count - first_embed_count} extra times on re-add of identical content"
            assert result2["skipped"] >= 1, "Second add should report file as skipped"
