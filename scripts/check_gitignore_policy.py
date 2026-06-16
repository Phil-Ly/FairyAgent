#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_AGENT_MARKDOWN_PATTERNS = [
    "agent.md",
    "Agent.md",
    "Feature.md",
    "AGENT.md",
    "AGENTS.md",
    "CLAUDE.md",
    "SPEC.md",
    "TESTING.md",
    "ARCHITECTURE.md",
    "CONVENTIONS.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "DECISIONS.md",
    "RUNBOOK.md",
    "API.md",
    "ROADMAP.md",
    "PRODUCT.md",
    "PRD.md",
    "REQUIREMENTS.md",
    "DESIGN.md",
    "CONTEXT.md",
    "PROJECT_CONTEXT.md",
    "MEMORY.md",
    "PROMPTS.md",
    "PROMPTBOOK.md",
    "WORKFLOW.md",
    "COLLABORATION.md",
    "PLAYBOOK.md",
    "PLANNING.md",
    "PLAN.md",
    "PLANS.md",
    "TASKS.md",
    "BACKLOG.md",
    "NOTES.md",
    "IDEAS.md",
    "HANDOFF.md",
    "CHECKLIST.md",
    "MILESTONES.md",
    "REVIEW.md",
    "RETROSPECTIVE.md",
    "ADR.md",
    "ADRS.md",
    "EVALUATION.md",
    "EVALS.md",
]

REQUIRED_SECRET_PATTERNS = [
    ".env",
    ".env.*",
    "!.env.example",
    "*.env",
    "*.pem",
    "*.key",
    "*.crt",
    "*.der",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "*.kubeconfig",
    "kubeconfig",
    ".aws/",
    ".azure/",
    ".gcp/",
    "secrets/",
    "credentials/",
    "credentials.json",
    "token.json",
    "service-account*.json",
    "*-service-account.json",
    "*.secret",
    "*.secrets",
]


@dataclass(frozen=True)
class GitignorePolicyReport:
    gitignore_path: Path
    missing_agent_markdown_patterns: list[str]
    missing_secret_patterns: list[str]
    forbidden_readme_patterns: list[str]

    @property
    def has_issues(self) -> bool:
        return bool(
            self.missing_agent_markdown_patterns
            or self.missing_secret_patterns
            or self.forbidden_readme_patterns
        )


def parse_gitignore_patterns(gitignore_path: Path) -> list[str]:
    if not gitignore_path.exists():
        return []

    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)
    return patterns


def normalize_pattern(pattern: str) -> str:
    negated = pattern.startswith("!")
    body = pattern[1:] if negated else pattern
    while body.startswith("./"):
        body = body[2:]
    body = body.lstrip("/")
    if body.endswith("/"):
        body = body.rstrip("/")
    return f"!{body}" if negated else body


def pattern_is_present(required_pattern: str, actual_patterns: list[str]) -> bool:
    required = normalize_pattern(required_pattern)
    return any(normalize_pattern(actual) == required for actual in actual_patterns)


def readme_ignore_patterns(actual_patterns: list[str]) -> list[str]:
    forbidden = []
    for pattern in actual_patterns:
        normalized = normalize_pattern(pattern).lower()
        if normalized in {"readme.md", "readme*"}:
            forbidden.append(pattern)
    return forbidden


def evaluate_gitignore(gitignore_path: Path) -> GitignorePolicyReport:
    actual_patterns = parse_gitignore_patterns(gitignore_path)
    missing_agent = [
        pattern
        for pattern in REQUIRED_AGENT_MARKDOWN_PATTERNS
        if not pattern_is_present(pattern, actual_patterns)
    ]
    missing_secrets = [
        pattern
        for pattern in REQUIRED_SECRET_PATTERNS
        if not pattern_is_present(pattern, actual_patterns)
    ]
    return GitignorePolicyReport(
        gitignore_path=gitignore_path,
        missing_agent_markdown_patterns=missing_agent,
        missing_secret_patterns=missing_secrets,
        forbidden_readme_patterns=readme_ignore_patterns(actual_patterns),
    )


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (candidate / ".gitignore").exists():
            return candidate
    return current


def candidate_project_from_hook_input(payload: Any) -> Path | None:
    keys = {
        "cwd",
        "project_root",
        "projectRoot",
        "git_root",
        "gitRoot",
        "current_working_directory",
        "currentWorkingDirectory",
    }

    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value:
                return Path(value)

        roots = payload.get("runtimeWorkspaceRoots") or payload.get(
            "runtime_workspace_roots"
        )
        if isinstance(roots, list):
            for value in roots:
                if isinstance(value, str) and value:
                    return Path(value)

        for value in payload.values():
            nested = candidate_project_from_hook_input(value)
            if nested is not None:
                return nested

    if isinstance(payload, list):
        for value in payload:
            nested = candidate_project_from_hook_input(value)
            if nested is not None:
                return nested

    return None


def project_from_stdin() -> Path | None:
    if sys.stdin.isatty():
        return None

    content = sys.stdin.read().strip()
    if not content:
        return None

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return candidate_project_from_hook_input(payload)


def resolve_project_path(cli_project: str | None) -> Path:
    if cli_project:
        return find_project_root(Path(cli_project))

    hook_project = project_from_stdin()
    if hook_project is not None:
        return find_project_root(hook_project)

    return find_project_root(Path(os.environ.get("PWD", os.getcwd())))


def format_report(report: GitignorePolicyReport) -> str:
    if not report.has_issues:
        return f"gitignore policy check passed: {report.gitignore_path}"

    lines = [f"gitignore policy check failed: {report.gitignore_path}"]
    if report.missing_agent_markdown_patterns:
        lines.append("missing agent/vibe markdown ignore patterns:")
        lines.extend(
            f"  - {pattern}" for pattern in report.missing_agent_markdown_patterns
        )
    if report.missing_secret_patterns:
        lines.append("missing secret ignore patterns:")
        lines.extend(f"  - {pattern}" for pattern in report.missing_secret_patterns)
    if report.forbidden_readme_patterns:
        lines.append("README must not be ignored; remove these patterns:")
        lines.extend(f"  - {pattern}" for pattern in report.forbidden_readme_patterns)
    return "\n".join(lines)


def format_codex_session_start_output(report: GitignorePolicyReport) -> str:
    message = format_report(report)
    return json.dumps(
        {
            "systemMessage": message,
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": message,
            },
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check project .gitignore for agent docs and secret patterns."
    )
    parser.add_argument("--project", help="Project directory to inspect.")
    parser.add_argument(
        "--hook-mode",
        action="store_true",
        help="Always exit 0 so Codex session_start is advisory.",
    )
    parser.add_argument(
        "--codex-json",
        action="store_true",
        help="Emit SessionStart JSON so failures surface as Codex UI warnings.",
    )
    args = parser.parse_args()

    project_root = resolve_project_path(args.project)
    report = evaluate_gitignore(project_root / ".gitignore")
    output = (
        format_codex_session_start_output(report)
        if args.codex_json
        else format_report(report)
    )
    if output:
        print(output)
    if args.hook_mode:
        return 0
    return 1 if report.has_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
