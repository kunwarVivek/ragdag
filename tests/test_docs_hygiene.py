from pathlib import Path

_project_root = Path(__file__).parent.parent


class TestDocsHygiene:
    def test_readme_exists_for_package_metadata(self):
        readme = _project_root / "README.md"
        assert readme.exists(), "README.md is required by pyproject metadata"
        text = readme.read_text(encoding="utf-8", errors="replace")
        assert "# ragdag" in text

    def test_no_generated_report_markdown_in_repo_tree(self):
        scan_roots = [
            _project_root / ".claude" / "cache" / "agents",
            _project_root / "thoughts" / "shared" / "plans",
        ]
        violations = []
        for scan_root in scan_roots:
            if not scan_root.exists():
                continue
            for md_file in scan_root.rglob("*.md"):
                violations.append(str(md_file.relative_to(_project_root)))

        assert violations == [], (
            "Generated point-in-time markdown reports must not exist in repo tree:\n"
            + "\n".join(violations)
        )
