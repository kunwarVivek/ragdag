"""Integration tests for the full RAG quality stack (provenance + propositions + HyDE + CRAG)."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import ragdag
from engines.synthesis import read_frontmatter, read_body


class TestFullStack:
    def _setup_dag_with_all_features(self, tmp_path):
        """Create a dag with proposition chunking enabled."""
        dag = ragdag.init(str(tmp_path))
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("chunk_strategy = heading", "chunk_strategy = proposition")
        config_path.write_text(config_text)
        return dag

    def test_provenance_survives_proposition_chunking(self, tmp_path):
        """Proposition chunks still have provenance frontmatter."""
        dag = self._setup_dag_with_all_features(tmp_path)
        doc = tmp_path / "doc.md"
        doc.write_text("# Overview\nThe sky is very blue today. Water is always extremely wet. Fire is incredibly hot indeed.")
        dag.add(str(doc))

        store = tmp_path / ".ragdag"
        chunks = sorted(f for f in store.rglob("*.txt") if f.name[0].isdigit())
        assert len(chunks) >= 2

        for cf in chunks:
            fm = read_frontmatter(cf)
            assert fm is not None
            assert fm["type"] == "chunk"
            assert fm["source"] == str(doc.resolve())
            assert fm["strategy"] == "proposition"

    def test_search_returns_clean_content(self, tmp_path):
        """Search results have content without frontmatter artifacts."""
        dag = self._setup_dag_with_all_features(tmp_path)
        doc = tmp_path / "doc.md"
        doc.write_text("# Testing\nPytest is a great testing framework for Python applications.")
        dag.add(str(doc))

        results = dag.search("pytest", mode="keyword")
        for r in results:
            assert "---\ntype:" not in r.content

    def test_backward_compat_bare_chunks(self, tmp_path):
        """Old-style bare chunks (no frontmatter) still work in search."""
        dag = ragdag.init(str(tmp_path))
        store = tmp_path / ".ragdag"

        doc_dir = store / "legacy"
        doc_dir.mkdir()
        (doc_dir / "01.txt").write_text("Kubernetes orchestrates containers at scale.")

        results = dag.search("kubernetes", mode="keyword")
        assert len(results) >= 1
        assert "Kubernetes" in results[0].content

    def test_ask_result_always_has_new_fields(self, tmp_path):
        """AskResult from ask() always includes confidence and retrieval_attempts."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "doc.md"
        doc.write_text("# Hello\nWorld content here for testing.")
        dag.add(str(doc))

        result = dag.ask("hello", use_llm=False)
        assert hasattr(result, "confidence")
        assert hasattr(result, "retrieval_attempts")
        assert result.retrieval_attempts >= 1

    def test_hyde_disabled_does_not_affect_search(self, tmp_path):
        """With hyde=false, search works exactly as before."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "doc.md"
        doc.write_text("# Deployment\nDeploy using Docker containers and Kubernetes.")
        dag.add(str(doc))

        results = dag.search("deploy", mode="keyword")
        assert len(results) >= 1

    def test_crag_disabled_does_not_affect_ask(self, tmp_path):
        """With crag=false, ask() returns results without relevance checking."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "doc.md"
        doc.write_text("# Test\nImportant testing information for validation.")
        dag.add(str(doc))

        result = dag.ask("testing", use_llm=False)
        assert result.context != ""
        assert result.confidence == "unknown"  # No CRAG check
        assert result.retrieval_attempts == 1
