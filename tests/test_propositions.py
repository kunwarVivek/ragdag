"""Tests for proposition chunking strategy."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import ragdag
from ragdag.core import RagDag


class TestPropositionChunking:
    def test_proposition_with_llm_mock(self, tmp_path):
        """Proposition strategy decomposes text into atomic statements via LLM."""
        dag = ragdag.init(str(tmp_path))
        # Set LLM provider so proposition attempts LLM path
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("provider = none", "provider = openai", 1)  # first occurrence = LLM
        config_path.write_text(config_text)

        mock_response = (
            "Docker containers run in isolation from each other.\n"
            "Kubernetes orchestrates Docker containers at scale.\n"
            "For local development, docker-compose is recommended."
        )
        with patch("engines.llm.call_llm", return_value=mock_response):
            text = "Docker containers run in isolation. Kubernetes orchestrates them at scale. For local dev, use docker-compose instead."
            results = dag._chunk_proposition(text, 1000, 0)
            assert len(results) == 3
            assert "Docker containers" in results[0][0]
            assert "Kubernetes" in results[1][0]
            assert "docker-compose" in results[2][0]

    def test_proposition_fallback_no_llm(self, tmp_path):
        """With provider=none, proposition falls back to sentence splitting."""
        dag = ragdag.init(str(tmp_path))
        text = "First sentence here. Second sentence follows. Third sentence ends it."
        results = dag._chunk_proposition(text, 1000, 0)
        assert len(results) >= 2
        for chunk_text, heading in results:
            assert isinstance(chunk_text, str)
            assert len(chunk_text.strip()) > 0

    def test_proposition_strategy_in_chunk_text_with_meta(self, tmp_path):
        """_chunk_text_with_meta routes 'proposition' strategy correctly."""
        dag = ragdag.init(str(tmp_path))
        text = "One important fact here. Another important fact follows. Third important fact ends."
        results = dag._chunk_text_with_meta(text, 1000, 0, "proposition")
        assert len(results) >= 2

    def test_proposition_ingest_creates_files(self, tmp_path):
        """add() with proposition strategy creates proposition chunk files."""
        dag = ragdag.init(str(tmp_path))
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("chunk_strategy = heading", "chunk_strategy = proposition")
        config_path.write_text(config_text)

        md_file = tmp_path / "doc.md"
        md_file.write_text("The sky is very blue today. Water is always extremely wet. Fire is incredibly hot indeed.")
        dag.add(str(md_file))

        store = tmp_path / ".ragdag"
        chunk_files = sorted(f for f in store.rglob("*.txt") if f.name[0].isdigit())
        assert len(chunk_files) >= 2

    def test_proposition_preserves_heading_from_parent(self, tmp_path):
        """Propositions from a heading chunk inherit the heading."""
        dag = ragdag.init(str(tmp_path))
        text = "# Setup\nInstall Docker first. Configure networking next. Start services last."

        mock_response = (
            "Docker should be installed first.\n"
            "Networking needs to be configured next.\n"
            "Services should be started last."
        )
        # Enable LLM
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("provider = none", "provider = openai", 1)
        config_path.write_text(config_text)

        with patch("engines.llm.call_llm", return_value=mock_response):
            results = dag._chunk_text_with_meta(text, 5000, 0, "proposition")
            assert len(results) == 3
            # All propositions should inherit the "# Setup" heading
            for _, heading in results:
                assert "Setup" in heading
