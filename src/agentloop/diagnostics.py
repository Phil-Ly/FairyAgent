"""Runtime diagnostics and display helpers."""

from __future__ import annotations

import importlib.util
import platform
import shutil
from collections.abc import Iterable
from pathlib import Path

from langchain_core.tools import BaseTool

from agentloop.config import ConfigurationError, load_config
from agentloop.memory_store import SQLiteMemoryStore
from agentloop.provider_registry import (
    DEFAULT_PROVIDER_REGISTRY,
    UnknownProviderError,
)
from agentloop.safety import SafetyPolicy
from agentloop.session_store import SQLiteSessionStore
from agentloop.storage import (
    get_memory_db_path,
    get_session_db_path,
    get_trace_db_path,
)
from agentloop.tool_runtime import ToolRuntime
from agentloop.tools import get_default_tool_runtime
from agentloop.trace_store import SQLiteTraceStore


def get_tool_lines(tools: Iterable[BaseTool] | ToolRuntime) -> list[str]:
    """Return display lines for tools."""

    if isinstance(tools, ToolRuntime):
        runtime = tools
    else:
        runtime = ToolRuntime.from_tools(list(tools))
    metadata = runtime.get_metadata()
    lines = []
    for tool in runtime.get_tools():
        tool_metadata = metadata[tool.name]
        lines.append(
            f"{tool.name}: "
            f"risk={tool_metadata.risk_level.value} "
            f"read_only={_bool_text(tool_metadata.read_only)} "
            f"requires_confirmation={_bool_text(tool_metadata.requires_confirmation)} "
            f"- {tool_metadata.description}"
        )
    return lines


def get_doctor_lines() -> list[str]:
    """Return local deployment readiness lines."""

    try:
        config = load_config()
    except ConfigurationError as exc:
        config = None
        config_error = str(exc)
    else:
        config_error = None

    lines = [
        "AgentLoop doctor",
        f"Python: {platform.python_version()}",
        f"uv: {'installed' if shutil.which('uv') else 'missing'}",
        f"langchain: {_package_status('langchain')}",
        f"langgraph: {_package_status('langgraph')}",
    ]
    if config is None:
        lines.append("configuration: invalid")
        lines.append(f"configuration_error: {config_error}")
    else:
        lines.append("configuration: ok")
        lines.append(f"MODEL_PROVIDER: {config.model_provider}")
        lines.append(f"MODEL_NAME: {config.model_name}")
        lines.append(
            "MODEL_BASE_URL: "
            f"{'configured' if config.model_base_url else 'missing'}"
        )
        lines.append(
            "MODEL_API_KEY: "
            f"{'configured' if config.model_api_key else 'missing'}"
        )
        lines.append(f"MAX_STEPS: {config.max_steps}")
        try:
            preset = DEFAULT_PROVIDER_REGISTRY.get(config.model_provider)
        except UnknownProviderError as exc:
            lines.append("provider: invalid")
            lines.append(f"provider_error: {exc}")
        else:
            lines.append(
                f"{preset.dependency_package}: "
                f"{_package_status(preset.dependency_module)}"
            )
            for env_var in preset.api_key_env_vars:
                value = getattr(config, env_var.lower(), None)
                lines.append(
                    f"{env_var}: {'configured' if value else 'missing'}"
                )
    lines.extend(_storage_status_lines())
    lines.extend(_safety_status_lines())
    lines.append(
        "tools: "
        + ", ".join(tool.name for tool in get_default_tool_runtime().get_tools())
    )
    return lines


def _package_status(module_name: str) -> str:
    """Return installed/missing for an importable package."""

    return "installed" if importlib.util.find_spec(module_name) else "missing"


def _storage_status_lines() -> list[str]:
    return [
        _sqlite_store_status(
            "session_store",
            get_session_db_path(),
            SQLiteSessionStore,
        ),
        _sqlite_store_status("memory_store", get_memory_db_path(), SQLiteMemoryStore),
        _sqlite_store_status("trace_store", get_trace_db_path(), SQLiteTraceStore),
    ]


def _sqlite_store_status(label: str, path: Path, store_factory) -> str:
    try:
        store = store_factory(path)
    except Exception as exc:
        return f"{label}: invalid path={path} error={type(exc).__name__}"
    store.close()
    return f"{label}: ready path={path}"


def _safety_status_lines() -> list[str]:
    policy = SafetyPolicy()
    allowlist = policy.tool_allowlist
    allowlist_text = "default" if allowlist is None else ",".join(sorted(allowlist))
    return [
        "safety_policy: ready "
        f"workspace_root={policy.workspace_root} "
        f"tool_allowlist={allowlist_text} "
        f"max_output_chars={policy.max_output_chars}"
    ]


def _bool_text(value: bool) -> str:
    return "true" if value else "false"
