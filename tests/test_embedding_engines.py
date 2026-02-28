"""Tests for embedding engines — abstract base, OpenAI, and local engines."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root and sdk to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from engines.base import EmbeddingEngine
from engines.openai_engine import OpenAIEngine
from engines.local_engine import LocalEngine


class TestEmbeddingEngineABC:
    """Tests for the abstract base class."""

    def test_embedding_engine_is_abstract(self):
        """EmbeddingEngine cannot be instantiated directly (raises TypeError)."""
        with pytest.raises(TypeError):
            EmbeddingEngine()

    def test_embedding_engine_has_required_methods(self):
        """ABC has embed, dimensions, model_name methods."""
        # Check that the abstract methods are declared on the class
        abstract_methods = EmbeddingEngine.__abstractmethods__
        assert "embed" in abstract_methods
        assert "dimensions" in abstract_methods
        assert "model_name" in abstract_methods


class TestOpenAIEngine:
    """Tests for the OpenAI embedding engine."""

    def test_openai_engine_requires_api_key(self):
        """OpenAIEngine() without OPENAI_API_KEY raises ValueError."""
        with patch.dict(os.environ, {}, clear=False):
            # Ensure the key is not set
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIEngine()

    def test_openai_engine_embed_mocked(self):
        """With mocked openai module and API key env var, embed() returns
        vectors of correct dimensions."""
        dims = 1536

        # Create mock embedding data items
        mock_item_1 = MagicMock()
        mock_item_1.embedding = [0.1] * dims
        mock_item_2 = MagicMock()
        mock_item_2.embedding = [0.2] * dims

        mock_response = MagicMock()
        mock_response.data = [mock_item_1, mock_item_2]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}):
            engine = OpenAIEngine(model="text-embedding-3-small", dims=dims)

            with patch.dict(sys.modules, {"openai": mock_openai}):
                result = engine.embed(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == dims
        assert len(result[1]) == dims
        assert result[0] == [0.1] * dims
        assert result[1] == [0.2] * dims

        # Verify the API was called correctly
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["hello", "world"],
            dimensions=dims,
        )


class TestLocalEngine:
    """Tests for the local sentence-transformers engine."""

    def test_local_engine_embed_mocked(self):
        """With mocked sentence_transformers, embed() returns vectors."""
        import numpy as np

        dims = 384
        mock_embeddings = np.random.rand(3, dims).astype(np.float32)

        mock_model_instance = MagicMock()
        mock_model_instance.encode.return_value = mock_embeddings
        mock_model_instance.get_sentence_embedding_dimension.return_value = dims

        mock_st_module = MagicMock()
        mock_st_class = MagicMock(return_value=mock_model_instance)
        mock_st_module.SentenceTransformer = mock_st_class

        engine = LocalEngine(model="all-MiniLM-L6-v2", dims=dims)

        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            result = engine.embed(["one", "two", "three"])

        assert len(result) == 3
        for vec in result:
            assert len(vec) == dims
            assert all(isinstance(v, float) for v in vec)

        # Verify the model was loaded and encode was called
        mock_st_class.assert_called_once_with("all-MiniLM-L6-v2")
        mock_model_instance.encode.assert_called_once_with(
            ["one", "two", "three"], show_progress_bar=False
        )
