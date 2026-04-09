"""Tests for HyDE (Hypothetical Document Embeddings) query transformation."""

import hashlib
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import ragdag
from ragdag.core import RagDag


class TestHyDE:
    def _enable_hyde_and_llm(self, tmp_path):
        """Helper to enable HyDE + LLM in config."""
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        # Enable LLM provider (under [llm] section, not [embedding])
        config_text = config_text.replace(
            "[llm]\nprovider = none",
            "[llm]\nprovider = openai",
        )
        # Enable hyde in search section
        config_text = config_text.replace("hyde = false", "hyde = true")
        config_path.write_text(config_text)

    def test_hyde_expand_generates_hypothetical(self, tmp_path):
        """_hyde_expand calls LLM to generate hypothetical answer."""
        dag = ragdag.init(str(tmp_path))
        self._enable_hyde_and_llm(tmp_path)

        mock_answer = "To deploy, build the Docker image and push to the registry."
        with patch("engines.llm.call_llm", return_value=mock_answer):
            result = dag._hyde_expand("How do I deploy?")
            assert result == mock_answer

    def test_hyde_expand_returns_original_when_disabled(self, tmp_path):
        """With hyde=false (default), _hyde_expand returns the original query."""
        dag = ragdag.init(str(tmp_path))
        result = dag._hyde_expand("How do I deploy?")
        assert result == "How do I deploy?"

    def test_hyde_expand_returns_original_when_no_llm(self, tmp_path):
        """With provider=none, _hyde_expand returns original even if hyde=true."""
        dag = ragdag.init(str(tmp_path))
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("hyde = false", "hyde = true")
        config_path.write_text(config_text)

        result = dag._hyde_expand("How do I deploy?")
        assert result == "How do I deploy?"

    def test_hyde_caching(self, tmp_path):
        """Second call with same query uses cache, not LLM."""
        dag = ragdag.init(str(tmp_path))
        self._enable_hyde_and_llm(tmp_path)

        mock_answer = "Deploy using Docker."
        with patch("engines.llm.call_llm", return_value=mock_answer) as mock_llm:
            result1 = dag._hyde_expand("How do I deploy?")
            result2 = dag._hyde_expand("How do I deploy?")
            assert result1 == result2 == mock_answer
            assert mock_llm.call_count == 1  # Cached

    def test_hyde_graceful_degradation(self, tmp_path):
        """If LLM fails, returns original query."""
        dag = ragdag.init(str(tmp_path))
        self._enable_hyde_and_llm(tmp_path)

        with patch("engines.llm.call_llm", side_effect=Exception("API error")):
            result = dag._hyde_expand("How do I deploy?")
            assert result == "How do I deploy?"

    def test_hyde_config_default_false(self, tmp_path):
        """HyDE is disabled by default."""
        dag = ragdag.init(str(tmp_path))
        assert dag._read_config("search.hyde", "false") == "false"
