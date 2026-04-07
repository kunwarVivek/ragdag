"""Tests for cross-encoder reranker with mocked model."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import engines.reranker as reranker_module
from engines.reranker import rerank


@pytest.fixture(autouse=True)
def reset_cross_encoder():
    """Reset the global cached model before each test."""
    reranker_module._cross_encoder = None
    yield
    reranker_module._cross_encoder = None


class TestRerank:
    """Tests for the rerank function."""

    def test_empty_candidates_returns_empty(self):
        """Empty input should return empty list."""
        result = rerank("some query", [])
        assert result == []

    @patch("engines.reranker._get_cross_encoder")
    def test_reorders_by_cross_encoder_scores(self, mock_get_ce):
        """Results should be reordered based on blended scores."""
        mock_model = MagicMock()
        # CE scores: doc_c highest, doc_a lowest
        mock_model.predict.return_value = [0.1, 0.5, 0.9]
        mock_get_ce.return_value = mock_model

        candidates = [
            ("doc_a.md", 0.9, "content a"),
            ("doc_b.md", 0.6, "content b"),
            ("doc_c.md", 0.3, "content c"),
        ]

        result = rerank("test query", candidates)

        paths = [path for path, _ in result]
        # doc_c has highest CE (0.9 normalized=1.0), low RRF (0.3/0.9 ~ 0.33)
        # doc_a has lowest CE (0.1 normalized=0.0), high RRF (0.9/0.9 = 1.0)
        # Blended: doc_c = 0.4*0.33 + 0.6*1.0 = 0.733
        # Blended: doc_a = 0.4*1.0 + 0.6*0.0 = 0.4
        # doc_c should be first
        assert paths[0] == "doc_c.md"

    @patch("engines.reranker._get_cross_encoder")
    def test_all_candidates_preserved(self, mock_get_ce):
        """All input candidates should appear in output."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.3, 0.8]
        mock_get_ce.return_value = mock_model

        candidates = [
            ("a.md", 0.5, "a"),
            ("b.md", 0.4, "b"),
            ("c.md", 0.3, "c"),
        ]

        result = rerank("query", candidates)
        result_paths = {path for path, _ in result}

        assert result_paths == {"a.md", "b.md", "c.md"}
        assert len(result) == 3

    @patch("engines.reranker._get_cross_encoder")
    def test_top_k_limits_output(self, mock_get_ce):
        """top_k should limit the number of results returned."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.5, 0.9]
        mock_get_ce.return_value = mock_model

        candidates = [
            ("a.md", 0.5, "a"),
            ("b.md", 0.4, "b"),
            ("c.md", 0.3, "c"),
        ]

        result = rerank("query", candidates, top_k=2)
        assert len(result) == 2

    @patch("engines.reranker._get_cross_encoder")
    def test_blending_high_rrf_low_ce_vs_low_rrf_high_ce(self, mock_get_ce):
        """With 40/60 blending, high CE should outweigh high RRF."""
        mock_model = MagicMock()
        # doc_a: high RRF (1.0), low CE (0.0)
        # doc_b: low RRF (0.1), high CE (1.0)
        mock_model.predict.return_value = [0.0, 1.0]
        mock_get_ce.return_value = mock_model

        candidates = [
            ("high_rrf.md", 1.0, "a"),
            ("high_ce.md", 0.1, "b"),
        ]

        result = rerank("query", candidates)
        # high_rrf: 0.4*1.0 + 0.6*0.0 = 0.4
        # high_ce: 0.4*0.1 + 0.6*1.0 = 0.64
        assert result[0][0] == "high_ce.md"
        assert result[1][0] == "high_rrf.md"

    @patch("engines.reranker._get_cross_encoder", side_effect=ImportError("no module"))
    def test_graceful_degradation_import_error(self, mock_get_ce):
        """ImportError should return candidates in original order without scores changed."""
        candidates = [
            ("a.md", 0.9, "a"),
            ("b.md", 0.5, "b"),
        ]

        result = rerank("query", candidates)
        assert result == [("a.md", 0.9), ("b.md", 0.5)]

    @patch("engines.reranker._get_cross_encoder", side_effect=RuntimeError("model failed"))
    def test_graceful_degradation_any_exception(self, mock_get_ce):
        """Any exception from model loading should return original order."""
        candidates = [
            ("a.md", 0.7, "a"),
            ("b.md", 0.3, "b"),
        ]

        result = rerank("query", candidates)
        assert result == [("a.md", 0.7), ("b.md", 0.3)]

    @patch("engines.reranker._get_cross_encoder")
    def test_single_candidate(self, mock_get_ce):
        """Single candidate should be returned as-is with blended score."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]
        mock_get_ce.return_value = mock_model

        candidates = [("only.md", 0.5, "only content")]

        result = rerank("query", candidates)
        assert len(result) == 1
        assert result[0][0] == "only.md"

    @patch("engines.reranker._get_cross_encoder")
    def test_normalization_equal_ce_scores(self, mock_get_ce):
        """When all CE scores are equal, ce_range=0 should not divide by zero."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.5, 0.5]
        mock_get_ce.return_value = mock_model

        candidates = [
            ("a.md", 0.9, "a"),
            ("b.md", 0.6, "b"),
            ("c.md", 0.3, "c"),
        ]

        result = rerank("query", candidates)
        # All CE normalized to 0 (since (0.5-0.5)/1.0 = 0)
        # So ordering should be purely by RRF
        assert len(result) == 3
        assert result[0][0] == "a.md"  # highest RRF
        # No ZeroDivisionError raised
