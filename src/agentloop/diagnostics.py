"""Runtime diagnostics and display helpers."""

from __future__ import annotations

import importlib.util
import platform
import shutil
from collections.abc import Iterable

from langchain_core.tools import BaseTool

from agentloop.config import ConfigurationError, load_config
from agentloop.tools import get_default_tools


def get_tool_lines(tools: Iterable[BaseTool]) -> list[str]:
    """Return display lines for tools."""

    return [f"{tool.name}: {tool.description}" for tool in tools]


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
        f"langchain-openai: {_package_status('langchain_openai')}",
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
        lines.append(
            "OPENAI_API_KEY: "
            f"{'configured' if config.openai_api_key else 'missing'}"
        )
    lines.append("tools: " + ", ".join(tool.name for tool in get_default_tools()))
    return lines


def _package_status(module_name: str) -> str:
    """Return installed/missing for an importable package."""

    return "installed" if importlib.util.find_spec(module_name) else "missing"
