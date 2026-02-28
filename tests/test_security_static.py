"""Static security analysis tests for ragdag shell scripts."""

import re
import sys
from pathlib import Path

# Ensure sdk and project root are importable
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "sdk"))


# ================================================================
# Security Static Analysis
# ================================================================

class TestSecurityStatic:
    """Static analysis of bash scripts for unsafe patterns."""

    def test_no_eval_in_bash_scripts(self):
        """Scan all lib/*.sh files for eval command usage.
        Assert no eval of user content exists."""
        lib_dir = _project_root / "lib"
        assert lib_dir.exists(), f"lib/ directory not found at {lib_dir}"

        sh_files = sorted(lib_dir.glob("*.sh"))
        assert len(sh_files) > 0, "No .sh files found in lib/"

        violations = []
        # Pattern: 'eval' as a standalone command (not inside comments,
        # not as part of another word like 'evaluate')
        eval_pattern = re.compile(r"(?:^|\s|;)eval\s", re.MULTILINE)

        for sh_file in sh_files:
            content = sh_file.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.lstrip()
                # Skip comment lines
                if stripped.startswith("#"):
                    continue
                if eval_pattern.search(line):
                    violations.append(f"{sh_file.name}:{i}: {stripped.strip()}")

        assert len(violations) == 0, (
            f"Found eval usage in bash scripts (potential command injection):\n"
            + "\n".join(violations)
        )

    def test_no_backtick_substitution_in_bash(self):
        """Scan lib/*.sh for backtick command substitution with user
        variables. Backtick substitution is error-prone and harder to
        nest safely; $() is preferred."""
        lib_dir = _project_root / "lib"
        assert lib_dir.exists(), f"lib/ directory not found at {lib_dir}"

        sh_files = sorted(lib_dir.glob("*.sh"))
        assert len(sh_files) > 0, "No .sh files found in lib/"

        violations = []
        # Pattern: backtick command substitution that contains a variable
        # reference ($VAR or ${VAR})
        backtick_var_pattern = re.compile(r"`[^`]*\$\{?\w+\}?[^`]*`")

        for sh_file in sh_files:
            content = sh_file.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.lstrip()
                # Skip comment lines
                if stripped.startswith("#"):
                    continue
                matches = backtick_var_pattern.findall(line)
                for match in matches:
                    violations.append(
                        f"{sh_file.name}:{i}: {match}"
                    )

        assert len(violations) == 0, (
            f"Found backtick command substitution with variables "
            f"(use $() instead):\n" + "\n".join(violations)
        )
