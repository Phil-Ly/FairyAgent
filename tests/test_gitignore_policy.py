from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "check_gitignore_policy.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_gitignore_policy", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_gitignore(root: Path, lines: list[str]) -> Path:
    path = root / ".gitignore"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def complete_required_patterns() -> list[str]:
    checker = load_checker()
    return [
        *checker.REQUIRED_AGENT_MARKDOWN_PATTERNS,
        *checker.REQUIRED_SECRET_PATTERNS,
    ]


def test_reports_missing_agent_markdown_and_secret_patterns(tmp_path: Path) -> None:
    checker = load_checker()
    gitignore = write_gitignore(
        tmp_path,
        [
            "AGENTS.md",
            ".env",
            "!.env.example",
        ],
    )

    report = checker.evaluate_gitignore(gitignore)

    assert "CLAUDE.md" in report.missing_agent_markdown_patterns
    assert "SPEC.md" in report.missing_agent_markdown_patterns
    assert ".env.*" in report.missing_secret_patterns
    assert "*.pem" in report.missing_secret_patterns


def test_does_not_require_or_ignore_readme(tmp_path: Path) -> None:
    checker = load_checker()
    gitignore = write_gitignore(tmp_path, complete_required_patterns())

    report = checker.evaluate_gitignore(gitignore)

    assert "README.md" not in checker.REQUIRED_AGENT_MARKDOWN_PATTERNS
    assert not report.forbidden_readme_patterns
    assert not report.has_issues


def test_flags_readme_ignore_patterns(tmp_path: Path) -> None:
    checker = load_checker()
    gitignore = write_gitignore(
        tmp_path,
        [
            *complete_required_patterns(),
            "README.md",
        ],
    )

    report = checker.evaluate_gitignore(gitignore)

    assert report.forbidden_readme_patterns == ["README.md"]
    assert report.has_issues


def test_cli_hook_mode_warns_but_exits_zero(tmp_path: Path) -> None:
    write_gitignore(tmp_path, ["AGENTS.md", ".env"])

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--project", str(tmp_path), "--hook-mode"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "gitignore policy check failed" in result.stdout
    assert "CLAUDE.md" in result.stdout


def test_cli_codex_json_outputs_session_start_warning(tmp_path: Path) -> None:
    write_gitignore(tmp_path, ["AGENTS.md", ".env"])

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--project",
            str(tmp_path),
            "--hook-mode",
            "--codex-json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert "gitignore policy check failed" in payload["systemMessage"]
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "CLAUDE.md" in payload["hookSpecificOutput"]["additionalContext"]


def test_cli_codex_json_outputs_session_start_success(tmp_path: Path) -> None:
    gitignore = write_gitignore(tmp_path, complete_required_patterns())

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--project",
            str(tmp_path),
            "--hook-mode",
            "--codex-json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["systemMessage"] == f"gitignore policy check passed: {gitignore}"
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert payload["hookSpecificOutput"]["additionalContext"] == (
        f"gitignore policy check passed: {gitignore}"
    )
