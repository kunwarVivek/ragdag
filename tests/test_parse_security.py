"""Tests for file parsing and filename sanitization in ragdag core."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure SDK is importable
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "sdk"))

from ragdag.core import RagDag, _sanitize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def dag(tmp_path):
    """Create a minimal RagDag instance for testing instance methods."""
    store = tmp_path / ".ragdag"
    store.mkdir()
    config = store / ".config"
    config.write_text(
        "[general]\nchunk_strategy = heading\nchunk_size = 1000\n"
        "chunk_overlap = 100\n\n"
        "[embedding]\nprovider = none\nmodel = text-embedding-3-small\n"
        "dimensions = 1536\n\n"
        "[llm]\nprovider = none\nmodel = gpt-4o-mini\nmax_context = 8000\n\n"
        "[search]\ndefault_mode = hybrid\ntop_k = 10\n"
        "keyword_weight = 0.3\nvector_weight = 0.7\n\n"
        "[edges]\nauto_relate = false\nrelate_threshold = 0.8\n"
        "record_queries = false\n"
    )
    for f in [".edges", ".processed", ".domain-rules"]:
        (store / f).write_text(f"# {f}\n")
    return RagDag(str(tmp_path))


# ===================================================================
# FILE TYPE DETECTION  (_detect_file_type)
# ===================================================================

class TestDetectFileType:
    """Tests for RagDag._detect_file_type."""

    def test_detect_markdown(self, dag, tmp_path):
        """1. .md extension returns 'markdown'."""
        p = tmp_path / "readme.md"
        p.write_text("# Hello")
        assert dag._detect_file_type(p) == "markdown"

    def test_detect_text(self, dag, tmp_path):
        """2. .txt extension returns 'text'."""
        p = tmp_path / "notes.txt"
        p.write_text("some notes")
        assert dag._detect_file_type(p) == "text"

    def test_detect_code_python(self, dag, tmp_path):
        """3. .py extension returns 'code'."""
        p = tmp_path / "app.py"
        p.write_text("print('hello')")
        assert dag._detect_file_type(p) == "code"

    def test_detect_code_javascript(self, dag, tmp_path):
        """4. .js extension returns 'code'."""
        p = tmp_path / "index.js"
        p.write_text("console.log('hi')")
        assert dag._detect_file_type(p) == "code"

    def test_detect_csv(self, dag, tmp_path):
        """5. .csv extension returns 'csv'."""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2\n")
        assert dag._detect_file_type(p) == "csv"

    def test_detect_json(self, dag, tmp_path):
        """6. .json extension returns 'json'."""
        p = tmp_path / "config.json"
        p.write_text("{}")
        assert dag._detect_file_type(p) == "json"

    def test_detect_unknown_defaults_text(self, dag, tmp_path):
        """7. Unknown extension .xyz falls back to 'text'."""
        p = tmp_path / "mystery.xyz"
        p.write_text("???")
        assert dag._detect_file_type(p) == "text"

    def test_detect_type_pdf(self, dag, tmp_path):
        """25. .pdf extension returns 'pdf'."""
        p = tmp_path / "report.pdf"
        p.write_bytes(b"%PDF-1.4 fake content")
        assert dag._detect_file_type(p) == "pdf"

    def test_detect_type_html(self, dag, tmp_path):
        """26. .html extension returns 'html'."""
        p = tmp_path / "page.html"
        p.write_text("<html><body>Hi</body></html>")
        assert dag._detect_file_type(p) == "html"

    def test_detect_type_docx(self, dag, tmp_path):
        """27. .docx extension returns 'docx'."""
        p = tmp_path / "document.docx"
        p.write_bytes(b"PK\x03\x04 fake docx content")
        assert dag._detect_file_type(p) == "docx"


# ===================================================================
# FILE PARSING  (_parse_file, _parse_csv, _parse_html)
# ===================================================================

class TestParseFile:
    """Tests for RagDag._parse_file and its helpers."""

    # -- Markdown --

    def test_parse_markdown_strips_frontmatter(self, dag, tmp_path):
        """8. YAML frontmatter between --- delimiters is stripped."""
        p = tmp_path / "doc.md"
        p.write_text("---\ntitle: Test\ndate: 2024-01-01\n---\n# Heading\nBody text")
        result = dag._parse_file(p)
        assert "title: Test" not in result
        assert "date: 2024-01-01" not in result
        assert "# Heading" in result
        assert "Body text" in result

    def test_parse_markdown_preserves_content(self, dag, tmp_path):
        """9. Headers and body text are preserved after frontmatter strip."""
        p = tmp_path / "doc.md"
        p.write_text("---\nkey: val\n---\n# Main\n\nParagraph one.\n\n## Sub\n\nParagraph two.")
        result = dag._parse_file(p)
        assert "# Main" in result
        assert "Paragraph one." in result
        assert "## Sub" in result
        assert "Paragraph two." in result

    def test_parse_markdown_no_frontmatter(self, dag, tmp_path):
        """10. Markdown without frontmatter passes through unchanged."""
        content = "# Title\n\nSome content here."
        p = tmp_path / "plain.md"
        p.write_text(content)
        result = dag._parse_file(p)
        assert result == content

    # -- CSV --

    def test_parse_csv_to_keyvalue(self, dag, tmp_path):
        """11. CSV is converted to '--- Record N ---\\nkey: value' format."""
        p = tmp_path / "data.csv"
        with open(p, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "age", "city"])
            writer.writerow(["Alice", "30", "NYC"])
            writer.writerow(["Bob", "25", "LA"])
        result = dag._parse_file(p)
        assert "--- Record 1 ---" in result
        assert "name: Alice" in result
        assert "age: 30" in result
        assert "city: NYC" in result
        assert "--- Record 2 ---" in result
        assert "name: Bob" in result
        assert "age: 25" in result
        assert "city: LA" in result

    # -- JSON --

    def test_parse_json_flatten(self, dag, tmp_path):
        """12. Nested JSON is flattened to 'key.subkey: value' lines."""
        data = {"user": {"name": "Alice", "age": 30}, "active": True}
        p = tmp_path / "data.json"
        p.write_text(json.dumps(data))
        result = dag._parse_file(p)
        assert "user.name: Alice" in result
        assert "user.age: 30" in result
        assert "active: True" in result

    def test_parse_json_invalid_passthrough(self, dag, tmp_path):
        """13. Invalid JSON returns raw text unmodified."""
        raw = "{ this is not valid json !!!"
        p = tmp_path / "bad.json"
        p.write_text(raw)
        result = dag._parse_file(p)
        assert result == raw

    # -- Plain text --

    def test_parse_text_passthrough(self, dag, tmp_path):
        """14. .txt files return content unchanged."""
        content = "Line one\nLine two\nLine three"
        p = tmp_path / "notes.txt"
        p.write_text(content)
        result = dag._parse_file(p)
        assert result == content

    # -- HTML --

    def test_parse_html_strips_tags_fallback(self, dag, tmp_path):
        """15. HTML tags are stripped when pandoc is not available (fallback path)."""
        html = (
            "<html><head><title>Test</title>"
            "<script>alert('xss')</script>"
            "<style>body{color:red}</style>"
            "</head><body>"
            "<h1>Hello</h1><p>World</p>"
            "</body></html>"
        )
        p = tmp_path / "page.html"
        p.write_text(html)

        # Force the fallback path by making pandoc unavailable
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = dag._parse_html(p)

        # Script and style content must be removed
        assert "alert" not in result
        assert "color:red" not in result
        # Actual text content preserved
        assert "Hello" in result
        assert "World" in result
        # No HTML tags remain
        assert "<" not in result
        assert ">" not in result

    # -- Code passthrough --

    def test_parse_code_passthrough(self, dag, tmp_path):
        """28. .py file content passes through unchanged (no transformation)."""
        content = "def hello():\n    print('world')\n\nclass Foo:\n    pass"
        p = tmp_path / "script.py"
        p.write_text(content)
        result = dag._parse_file(p)
        assert result == content

    # -- Unknown extension as text --

    def test_parse_unknown_extension_as_text(self, dag, tmp_path):
        """29. .xyz or .unknown extension reads file as plain text."""
        content = "Some arbitrary content in an unknown format."
        p = tmp_path / "data.unknown"
        p.write_text(content)
        result = dag._parse_file(p)
        assert result == content

    # -- PDF mocked --

    def test_parse_pdf_mocked(self, dag, tmp_path):
        """30. _parse_pdf calls pdftotext with correct args and returns stdout."""
        p = tmp_path / "report.pdf"
        p.write_bytes(b"%PDF-1.4 fake pdf")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Extracted text from PDF document."

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = dag._parse_pdf(p)

        # Verify pdftotext was called with the correct arguments
        mock_run.assert_called_once_with(
            ["pdftotext", str(p), "-"],
            capture_output=True, text=True, timeout=30,
        )
        assert result == "Extracted text from PDF document."

    # -- PDF missing pdftotext --

    def test_parse_pdf_missing_pdftotext(self, dag, tmp_path):
        """31. When pdftotext is not installed, _parse_pdf raises ValueError."""
        p = tmp_path / "report.pdf"
        p.write_bytes(b"%PDF-1.4 fake pdf")

        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(ValueError, match="pdftotext not available"):
                dag._parse_pdf(p)

    # -- DOCX mocked --

    def test_parse_docx_mocked(self, dag, tmp_path):
        """32. _parse_docx calls pandoc with correct args and returns output."""
        p = tmp_path / "document.docx"
        p.write_bytes(b"PK\x03\x04 fake docx")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Extracted text from DOCX document."

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = dag._parse_docx(p)

        mock_run.assert_called_once_with(
            ["pandoc", "-t", "plain", str(p)],
            capture_output=True, text=True, timeout=30,
        )
        assert result == "Extracted text from DOCX document."


# ===================================================================
# SECURITY: FILENAME SANITIZATION  (_sanitize)
# ===================================================================

class TestSanitize:
    """Tests for the module-level _sanitize function."""

    def test_sanitize_lowercase(self):
        """16. Upper-case letters are lowercased."""
        assert _sanitize("MyFile") == "myfile"

    def test_sanitize_strips_spaces(self):
        """17. Spaces are removed."""
        assert _sanitize("my file") == "myfile"

    def test_sanitize_strips_backticks(self):
        """18. Backticks (command injection vector) are removed."""
        assert _sanitize("file`cmd`") == "filecmd"

    def test_sanitize_strips_dollar(self):
        """19. Dollar signs (variable expansion) are removed."""
        assert _sanitize("file$HOME") == "filehome"

    def test_sanitize_strips_semicolons(self):
        """20. Semicolons (command chaining) are removed."""
        assert _sanitize("file;rm") == "filerm"

    def test_sanitize_strips_pipes(self):
        """21. Pipe characters are removed."""
        assert _sanitize("file|cmd") == "filecmd"

    def test_sanitize_preserves_dots_dashes_underscores(self):
        """22. Dots, dashes, and underscores are preserved."""
        assert _sanitize("my-file_v1.0") == "my-file_v1.0"

    def test_sanitize_strips_parentheses(self):
        """23. Parentheses are removed."""
        assert _sanitize("file(1)") == "file1"

    def test_sanitize_empty_after_strip(self):
        """24. Input of only special characters results in empty string."""
        assert _sanitize("!@#$%^&*()") == ""
