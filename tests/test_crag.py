"""Tests for CRAG (Corrective RAG) loop."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import ragdag
from ragdag.core import RagDag, AskResult


class TestAskResultFields:
    def test_ask_result_has_confidence(self):
        result = AskResult(answer="test", context="ctx", sources=["a.txt"])
        assert result.confidence == "unknown"

    def test_ask_result_has_retrieval_attempts(self):
        result = AskResult(answer="test", context="ctx")
        assert result.retrieval_attempts == 1

    def test_ask_result_explicit_confidence(self):
        result = AskResult(answer="ok", context="c", confidence="sufficient", retrieval_attempts=2)
        assert result.confidence == "sufficient"
        assert result.retrieval_attempts == 2


class TestCRAG:
    def _setup_crag_dag(self, tmp_path, content="# Deploy\nUse docker-compose up to deploy the app."):
        """Create dag with CRAG + LLM enabled and a doc ingested."""
        dag = ragdag.init(str(tmp_path))
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        # Enable LLM provider (the only "provider = none" is under [llm])
        lines = config_text.splitlines()
        in_llm = False
        new_lines = []
        for line in lines:
            if line.strip() == "[llm]":
                in_llm = True
            elif line.strip().startswith("[") and line.strip().endswith("]"):
                in_llm = False
            if in_llm and line.strip() == "provider = none":
                line = "provider = openai"
            new_lines.append(line)
        config_text = "\n".join(new_lines)
        # Enable CRAG in [search] section
        config_text = config_text.replace("crag = false", "crag = true")
        config_path.write_text(config_text)

        doc = tmp_path / "doc.md"
        doc.write_text(content)
        dag.add(str(doc))
        return dag

    def test_crag_sufficient_proceeds_normally(self, tmp_path):
        """SUFFICIENT rating -> proceed to answer, confidence=sufficient."""
        dag = self._setup_crag_dag(tmp_path)

        def mock_llm(system_msg, user_msg, provider="openai", model="gpt-4o-mini"):
            if "rate" in system_msg.lower() or "relevance" in system_msg.lower():
                return "SUFFICIENT"
            return "Use docker-compose up."

        with patch("engines.llm.call_llm", side_effect=mock_llm):
            result = dag.ask("How do I deploy?")
            assert result.confidence == "sufficient"
            assert result.retrieval_attempts == 1
            assert result.answer is not None

    def test_crag_insufficient_reformulates(self, tmp_path):
        """INSUFFICIENT -> reformulate -> retry -> answer."""
        dag = self._setup_crag_dag(tmp_path)

        call_count = [0]
        def mock_llm(system_msg, user_msg, provider="openai", model="gpt-4o-mini"):
            call_count[0] += 1
            if "rate" in system_msg.lower() or "relevance" in system_msg.lower():
                # First check: insufficient, second: sufficient
                if call_count[0] <= 2:
                    return "INSUFFICIENT"
                return "SUFFICIENT"
            if "reformulate" in system_msg.lower() or "search query" in system_msg.lower() or "search query" in user_msg.lower():
                return "deploy docker setup"
            return "Run docker-compose up"

        with patch("engines.llm.call_llm", side_effect=mock_llm):
            result = dag.ask("How to set up?")
            assert result.retrieval_attempts == 2

    def test_crag_gives_up_after_max_retries(self, tmp_path):
        """After max retries, returns with confidence=insufficient."""
        dag = self._setup_crag_dag(tmp_path, "# Cooking\nRecipe for pasta.")

        def mock_llm(system_msg, user_msg, provider="openai", model="gpt-4o-mini"):
            if "rate" in system_msg.lower() or "relevance" in system_msg.lower():
                return "INSUFFICIENT"
            if "search query" in user_msg.lower() or "reformulate" in system_msg.lower():
                return "deployment kubernetes"
            return "I don't have enough information."

        with patch("engines.llm.call_llm", side_effect=mock_llm):
            result = dag.ask("How do I deploy to k8s?")
            assert result.confidence == "insufficient"
            assert result.retrieval_attempts == 2

    def test_crag_disabled_by_default(self, tmp_path):
        """Default config has crag=false."""
        dag = ragdag.init(str(tmp_path))
        assert dag._read_config("search.crag", "false") == "false"

    def test_crag_partial_triggers_sub_query(self, tmp_path):
        """PARTIAL -> extract gap -> sub-query -> merge -> answer."""
        dag = self._setup_crag_dag(tmp_path, "# Architecture\nThe system uses microservices with REST APIs.")

        call_count = [0]
        def mock_llm(system_msg, user_msg, provider="openai", model="gpt-4o-mini"):
            call_count[0] += 1
            if "rate" in system_msg.lower() or "relevance" in system_msg.lower():
                if call_count[0] <= 2:
                    return "PARTIAL: missing information about database layer"
                return "SUFFICIENT"
            if "search query" in user_msg.lower() or "reformulate" in system_msg.lower() or "missing" in user_msg.lower():
                return "database schema microservices"
            return "The system uses microservices with REST APIs."

        with patch("engines.llm.call_llm", side_effect=mock_llm):
            result = dag.ask("Describe the full architecture")
            assert result.retrieval_attempts == 2
            assert result.confidence == "sufficient"

    def test_crag_no_llm_skips_check(self, tmp_path):
        """With use_llm=False, CRAG is skipped."""
        dag = ragdag.init(str(tmp_path))
        config_path = tmp_path / ".ragdag" / ".config"
        config_text = config_path.read_text()
        config_text = config_text.replace("crag = false", "crag = true")
        config_path.write_text(config_text)

        doc = tmp_path / "doc.md"
        doc.write_text("# Test\nSome content here.")
        dag.add(str(doc))

        result = dag.ask("What is this?", use_llm=False)
        assert result.answer is None
        assert result.confidence == "unknown"

    def test_ask_without_crag_still_returns_new_fields(self, tmp_path):
        """Even without CRAG, AskResult has confidence and retrieval_attempts."""
        dag = ragdag.init(str(tmp_path))
        doc = tmp_path / "doc.md"
        doc.write_text("# Hello\nWorld content.")
        dag.add(str(doc))

        result = dag.ask("hello", use_llm=False)
        assert hasattr(result, "confidence")
        assert hasattr(result, "retrieval_attempts")
        assert result.retrieval_attempts == 1
        assert result.confidence == "unknown"
