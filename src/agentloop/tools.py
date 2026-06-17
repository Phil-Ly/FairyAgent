"""LangChain tools exposed to the agent."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from agentloop.safety import ContentSource, SafetyPolicy
from agentloop.tool_runtime import ToolMetadata, ToolRiskLevel, ToolRuntime

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_EXCLUDED_SEARCH_DIRS = {
    ".git",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
_MAX_TEXT_FILE_BYTES = 1_000_000
_DEFAULT_MAX_MATCHES = 20
_MAX_MATCHES_LIMIT = 100
_DEFAULT_TREE_MAX_DEPTH = 2
_DEFAULT_TREE_MAX_ENTRIES = 200
_MAX_TREE_ENTRIES_LIMIT = 1_000


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression."""

    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate_expression(tree.body)
    except (SyntaxError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("Unsupported expression.") from exc

    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def _evaluate_expression(node: ast.AST) -> int | float:
    """Evaluate only whitelisted arithmetic AST nodes."""

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Unsupported expression.")
        return node.value

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)
        if operator_type not in _BINARY_OPERATORS:
            raise ValueError("Unsupported expression.")
        left = _evaluate_expression(node.left)
        right = _evaluate_expression(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Unsupported expression.")
        return _BINARY_OPERATORS[operator_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)
        if operator_type not in _UNARY_OPERATORS:
            raise ValueError("Unsupported expression.")
        return _UNARY_OPERATORS[operator_type](_evaluate_expression(node.operand))

    raise ValueError("Unsupported expression.")


@tool
def echo(text: str) -> str:
    """Return the provided text unchanged."""

    return text


@tool
def list_files(path: str = ".") -> str:
    """List direct children of a workspace directory."""

    workspace_root, target = _resolve_workspace_path(path)
    if not target.exists():
        raise ValueError("Path does not exist.")
    if not target.is_dir():
        raise ValueError("Path is not a directory.")

    entries = []
    for child in sorted(target.iterdir(), key=lambda item: item.name):
        suffix = "/" if child.is_dir() else ""
        entries.append(f"{child.name}{suffix}")
    return "\n".join(entries) if entries else "(empty)"


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file inside the workspace."""

    _, target = _resolve_workspace_path(path)
    if not target.exists():
        raise ValueError("Path does not exist.")
    if not target.is_file():
        raise ValueError("Path is not a file.")
    return _read_workspace_text_file(target)


@tool
def search_text(
    query: str,
    path: str = ".",
    max_matches: int = _DEFAULT_MAX_MATCHES,
) -> str:
    """Search UTF-8 text files inside the workspace."""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query cannot be empty.")
    match_limit = _validate_positive_limit(
        value=max_matches,
        field_name="max_matches",
        maximum=_MAX_MATCHES_LIMIT,
    )
    workspace_root, target = _resolve_workspace_path(path)
    if not target.exists():
        raise ValueError("Path does not exist.")
    if not target.is_file() and not target.is_dir():
        raise ValueError("Path is not a file or directory.")

    matches: list[str] = []
    query_key = normalized_query.casefold()
    for candidate in _iter_workspace_text_candidates(workspace_root, target):
        try:
            content = _read_workspace_text_file(candidate)
        except ValueError:
            continue
        relative_path = _workspace_relative_path(workspace_root, candidate)
        for line_number, line in enumerate(content.splitlines(), start=1):
            if query_key in line.casefold():
                matches.append(f"{relative_path}:{line_number}:{line}")
                if len(matches) >= match_limit:
                    return "\n".join(matches)

    return "\n".join(matches) if matches else "(no matches)"


@tool
def project_tree(
    path: str = ".",
    max_depth: int = _DEFAULT_TREE_MAX_DEPTH,
    max_entries: int = _DEFAULT_TREE_MAX_ENTRIES,
) -> str:
    """Show a depth-limited workspace directory tree."""

    workspace_root, target = _resolve_workspace_path(path)
    if not target.exists():
        raise ValueError("Path does not exist.")
    if not target.is_dir():
        raise ValueError("Path is not a directory.")
    depth_limit = _validate_non_negative_int(max_depth, "max_depth")
    entry_limit = _validate_positive_limit(
        value=max_entries,
        field_name="max_entries",
        maximum=_MAX_TREE_ENTRIES_LIMIT,
    )

    root_label = _workspace_relative_path(workspace_root, target)
    lines = [root_label if root_label == "." else f"{root_label}/"]
    entries_seen = 0
    truncated = False

    def add_children(directory: Path, depth: int) -> None:
        nonlocal entries_seen, truncated
        if depth >= depth_limit or truncated:
            return
        indent = "  " * (depth + 1)
        for child in _iter_visible_children(workspace_root, directory):
            if entries_seen >= entry_limit:
                lines.append(f"{indent}... (truncated)")
                truncated = True
                return
            label = f"{child.name}/" if child.is_dir() else child.name
            lines.append(f"{indent}{label}")
            entries_seen += 1
            if child.is_dir():
                add_children(child, depth + 1)

    add_children(target, depth=0)
    return "\n".join(lines)


def get_default_tools() -> list[BaseTool]:
    """Return the built-in LangChain tools."""

    return [calculator, echo, list_files, read_file, search_text, project_tree]


def _resolve_workspace_path(path: str) -> tuple[Path, Path]:
    """Return the workspace root and a resolved path within it."""

    policy = SafetyPolicy(workspace_root=Path.cwd())
    workspace_root = policy.workspace_root
    target = policy.resolve_workspace_path(path)
    return workspace_root, target


def _workspace_relative_path(workspace_root: Path, target: Path) -> str:
    """Return a stable POSIX path relative to the workspace root."""

    return target.relative_to(workspace_root).as_posix()


def _read_workspace_text_file(path: Path) -> str:
    """Read a bounded UTF-8 text file."""

    if path.stat().st_size > _MAX_TEXT_FILE_BYTES:
        raise ValueError("File is too large to read as a text file.")
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError("Path is not a text file.")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Path is not a text file.") from exc


def _iter_workspace_text_candidates(
    workspace_root: Path,
    target: Path,
) -> Iterator[Path]:
    """Yield candidate files in deterministic order without generated directories."""

    if not _is_within_workspace(workspace_root, target):
        return
    if target.is_file():
        yield target
        return
    for child in _iter_visible_children(workspace_root, target):
        if child.is_file():
            yield child
        elif child.is_dir():
            yield from _iter_workspace_text_candidates(workspace_root, child)


def _iter_visible_children(workspace_root: Path, directory: Path) -> Iterator[Path]:
    """Yield direct children while skipping generated dependency/cache directories."""

    for child in sorted(directory.iterdir(), key=lambda item: item.name):
        if not _is_within_workspace(workspace_root, child):
            continue
        if child.is_dir() and child.name in _EXCLUDED_SEARCH_DIRS:
            continue
        yield child


def _is_within_workspace(workspace_root: Path, path: Path) -> bool:
    """Return True when a path resolves inside the workspace."""

    return path.resolve().is_relative_to(workspace_root)


def _validate_positive_limit(value: int, field_name: str, maximum: int) -> int:
    """Validate a positive bounded integer option."""

    if value < 1:
        raise ValueError(f"{field_name} must be greater than 0.")
    return min(value, maximum)


def _validate_non_negative_int(value: int, field_name: str) -> int:
    """Validate a non-negative integer option."""

    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0.")
    return value


def get_default_tool_runtime() -> ToolRuntime:
    """Return the default low-risk local tool runtime."""

    tools = get_default_tools()
    file_tools = {"list_files", "read_file", "search_text", "project_tree"}
    metadata = {
        tool.name: ToolMetadata(
            name=tool.name,
            description=tool.description or "",
            risk_level=ToolRiskLevel.LOW,
            requires_confirmation=False,
            read_only=True,
            content_source=(
                ContentSource.FILE
                if tool.name in file_tools
                else ContentSource.TOOL
            ),
        )
        for tool in tools
    }
    return ToolRuntime.from_tools(tools, metadata=metadata)
