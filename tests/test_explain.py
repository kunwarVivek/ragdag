"""Tests for search explain mode."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

import pytest
import ragdag


class TestExplainMode:

    def test_explain_adds_breakdown_to_keyword_results(self, tmp_path):
        """Explain mode should attach score breakdown to keyword results."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "test.md"
        doc.write_text("# Test\n\nKeyword matching explanation mode.\n")
        dag.add(str(doc))

        results = dag.search("keyword matching", mode="keyword", explain=True)
        assert len(results) >= 1
        assert results[0].explain is not None
        assert "bm25" in results[0].explain

    def test_explain_false_has_no_breakdown(self, tmp_path):
        """Without explain, results should have explain=None."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "test.md"
        doc.write_text("# Test\n\nNo explain mode.\n")
        dag.add(str(doc))

        results = dag.search("explain", mode="keyword")
        assert len(results) >= 1
        assert results[0].explain is None

    def test_explain_default_is_false(self, tmp_path):
        """Default value for explain should be False."""
        import inspect
        from ragdag.core import RagDag
        sig = inspect.signature(RagDag.search)
        assert sig.parameters["explain"].default is False

    def test_explain_bm25_score_is_numeric(self, tmp_path):
        """BM25 score in explain should be a numeric value."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "data.md"
        doc.write_text("# Data\n\nSome interesting data content here.\n")
        dag.add(str(doc))

        results = dag.search("data content", mode="keyword", explain=True)
        assert len(results) >= 1
        bm25_score = results[0].explain["bm25"]
        assert isinstance(bm25_score, (int, float))
        assert bm25_score > 0

    def test_explain_hybrid_fallback_has_bm25(self, tmp_path):
        """Hybrid with no embeddings falls back to keyword — explain should have bm25."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "hybrid.md"
        doc.write_text("# Hybrid\n\nHybrid mode with explain.\n")
        dag.add(str(doc))

        # With provider=none, hybrid falls back to keyword
        results = dag.search("hybrid explain", mode="hybrid", explain=True)
        assert len(results) >= 1
        assert results[0].explain is not None
        assert "bm25" in results[0].explain
