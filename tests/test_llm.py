"""Tests for llm.py — none provider, prompt injection mitigation."""

import inspect
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.llm import get_answer, _build_user_message, SYSTEM_PROMPT


class TestNoneProvider:
    """Tests for provider=none handling."""

    def test_none_provider_returns_empty_string(self):
        """get_answer with provider='none' should return '' without calling any API."""
        result = get_answer(
            question="What is auth?",
            context="Auth is authentication.",
            provider="none",
        )
        assert result == ""

    def test_none_provider_does_not_raise(self):
        """provider='none' must not raise ValueError."""
        # Previously this raised ValueError("Unknown LLM provider: none")
        result = get_answer("q", "c", provider="none")
        assert isinstance(result, str)

    def test_unknown_provider_raises(self):
        """Unknown providers should still raise ValueError."""
        import pytest
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_answer("q", "c", provider="nonexistent_provider")


class TestPromptInjectionMitigation:
    """Tests for system/user message separation."""

    def test_system_prompt_exists_and_has_instructions(self):
        """SYSTEM_PROMPT should contain instructions about using context only."""
        assert "ONLY" in SYSTEM_PROMPT
        assert "context" in SYSTEM_PROMPT.lower()

    def test_system_prompt_warns_about_injection(self):
        """SYSTEM_PROMPT should warn about following instructions in context."""
        assert "never follow instructions" in SYSTEM_PROMPT.lower()

    def test_user_message_wraps_context_in_markers(self):
        """Context should be wrapped in [BEGIN CONTEXT] / [END CONTEXT]."""
        msg = _build_user_message("What is X?", "X is a thing.")
        assert "[BEGIN CONTEXT]" in msg
        assert "[END CONTEXT]" in msg
        assert "X is a thing." in msg
        assert "What is X?" in msg

    def test_context_appears_between_markers(self):
        """Context text must appear between the markers, not outside."""
        context = "SECRET_CONTEXT_DATA"
        msg = _build_user_message("question?", context)
        begin_idx = msg.index("[BEGIN CONTEXT]")
        end_idx = msg.index("[END CONTEXT]")
        context_idx = msg.index(context)
        assert begin_idx < context_idx < end_idx

    def test_question_appears_after_end_marker(self):
        """Question must appear after [END CONTEXT], not inside context block."""
        msg = _build_user_message("MY_QUESTION", "some context")
        end_idx = msg.index("[END CONTEXT]")
        question_idx = msg.index("MY_QUESTION")
        assert question_idx > end_idx


class TestOpenAIProvider:
    """Test OpenAI provider uses system/user message separation."""

    def test_openai_uses_system_message(self):
        """OpenAI calls should use separate system and user messages."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "answer"
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict(sys.modules, {"openai": mock_openai}):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                get_answer("q?", "ctx", provider="openai", model="gpt-4o-mini")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "[BEGIN CONTEXT]" in messages[1]["content"]


class TestAnthropicProvider:
    """Test Anthropic provider uses system/user message separation."""

    def test_anthropic_uses_system_param(self):
        """Anthropic calls should use the system parameter, not inline."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "answer"
        mock_client.messages.create.return_value = mock_response

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                get_answer("q?", "ctx", provider="anthropic", model="claude-3")

        call_args = mock_client.messages.create.call_args
        # Anthropic uses system= kwarg
        assert "system" in call_args.kwargs
        assert call_args.kwargs["system"] == SYSTEM_PROMPT
        messages = call_args.kwargs.get("messages", [])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


class TestOllamaProvider:
    """Tests for Ollama provider HTTP communication."""

    def test_ollama_provider_sends_request(self):
        """Ollama provider makes HTTP request to localhost:11434.

        _ollama_answer() constructs a JSON payload with model, prompt, system,
        and stream=False, then POSTs to {OLLAMA_URL}/api/generate via urllib.
        """
        # Create a mock response object for urllib.request.urlopen
        mock_response_body = json.dumps({"response": "mocked answer"}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response_body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            result = get_answer(
                "What is X?", "X is a thing.",
                provider="ollama", model="llama3",
            )

        # Verify a request was made
        assert mock_urlopen.called
        call_args = mock_urlopen.call_args
        req = call_args[0][0]  # First positional arg is the Request object
        assert "localhost:11434" in req.full_url
        assert "/api/generate" in req.full_url

        # Verify the payload
        sent_data = json.loads(req.data.decode())
        assert sent_data["model"] == "llama3"
        assert sent_data["stream"] is False
        assert "system" in sent_data
        assert "[BEGIN CONTEXT]" in sent_data["prompt"]

        # Verify the response was parsed
        assert result == "mocked answer"


class TestLLMMissingApiKey:
    """Tests for handling missing API keys."""

    def test_llm_missing_api_key_no_crash(self):
        """OpenAI/Anthropic without API key either raises clearly or handles gracefully.

        When no API key is set, the provider should either:
        - Raise ImportError (if package not installed)
        - Raise a clear authentication error from the SDK
        - NOT produce a cryptic/unexpected crash

        We test both providers with mocked packages that simulate auth errors.
        """
        import pytest

        # Test OpenAI: mock openai module, simulate AuthenticationError
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        auth_error = Exception("Incorrect API key provided")
        mock_client.chat.completions.create.side_effect = auth_error

        with patch.dict(sys.modules, {"openai": mock_openai}):
            with patch.dict("os.environ", {}, clear=False):
                # Remove API key if present
                old_key = os.environ.pop("OPENAI_API_KEY", None)
                try:
                    with pytest.raises(Exception, match="API key"):
                        get_answer("q", "c", provider="openai", model="gpt-4o-mini")
                finally:
                    if old_key is not None:
                        os.environ["OPENAI_API_KEY"] = old_key

        # Test Anthropic: mock anthropic module, simulate AuthenticationError
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        auth_error = Exception("Invalid API key")
        mock_client.messages.create.side_effect = auth_error

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch.dict("os.environ", {}, clear=False):
                old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
                try:
                    with pytest.raises(Exception, match="API key"):
                        get_answer("q", "c", provider="anthropic", model="claude-3")
                finally:
                    if old_key is not None:
                        os.environ["ANTHROPIC_API_KEY"] = old_key


class TestApiKeySource:
    """Tests for verifying API key sourcing patterns."""

    def test_api_keys_from_env_not_config(self):
        """Verify that engines/llm.py reads API keys from os.environ,
        not from any config file."""
        import engines.llm as llm_module

        source_code = inspect.getsource(llm_module)

        # The source code must use os.environ to read API keys
        assert "os.environ" in source_code, (
            "engines/llm.py should use os.environ to read API keys"
        )

        # Verify specific API key patterns are accessed via environ
        assert "OPENAI_API_KEY" in source_code, (
            "engines/llm.py should reference OPENAI_API_KEY"
        )
        assert "ANTHROPIC_API_KEY" in source_code, (
            "engines/llm.py should reference ANTHROPIC_API_KEY"
        )

        # Should NOT read API keys from a config file
        # Check that no config file reading patterns are used for keys
        assert "configparser" not in source_code.lower(), (
            "engines/llm.py should not use configparser to read API keys"
        )
        assert ".config" not in source_code or "api_key" not in source_code.lower().split(".config")[0], (
            "engines/llm.py should not read API keys from .config files"
        )
