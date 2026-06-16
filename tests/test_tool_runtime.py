from pathlib import Path

from langchain_core.tools import tool

from agentloop.tool_runtime import (
    ToolMetadata,
    ToolResultStatus,
    ToolRiskLevel,
    ToolRuntime,
)
from agentloop.tools import calculator, get_default_tool_runtime


def test_tool_runtime_returns_structured_success_result() -> None:
    runtime = ToolRuntime.from_tools([calculator])

    result = runtime.call_tool("calculator", {"expression": "2 + 3 * 4"})

    assert result.status is ToolResultStatus.SUCCESS
    assert result.content == "14"
    assert result.tool_name == "calculator"
    assert result.error_code is None
    assert result.duration_ms >= 0
    assert result.metadata.risk_level is ToolRiskLevel.LOW
    assert result.to_message_content() == "14"


def test_tool_runtime_returns_structured_unknown_tool_result() -> None:
    runtime = ToolRuntime.from_tools([calculator])

    result = runtime.call_tool("missing_tool", {"text": "hello"})

    assert result.status is ToolResultStatus.ERROR
    assert result.tool_name == "missing_tool"
    assert result.error_code == "unknown_tool"
    assert result.content == (
        "Tool error (unknown_tool): Tool 'missing_tool' is not registered."
    )


def test_tool_runtime_returns_structured_tool_failure_result() -> None:
    runtime = ToolRuntime.from_tools([calculator])

    result = runtime.call_tool("calculator", {})

    assert result.status is ToolResultStatus.ERROR
    assert result.error_code == "tool_failed"
    assert result.tool_name == "calculator"
    assert result.error_type is not None
    assert result.error_message is not None
    assert result.to_message_content() == (
        "Tool error (tool_failed): Tool 'calculator' failed."
    )


def test_tool_runtime_requires_confirmation_before_high_risk_tool_execution() -> None:
    executions: list[str] = []

    @tool
    def risky_write() -> str:
        """Write something risky."""

        executions.append("executed")
        return "wrote"

    runtime = ToolRuntime.from_tools(
        [risky_write],
        metadata={
            "risky_write": ToolMetadata(
                name="risky_write",
                description="Write something risky.",
                risk_level=ToolRiskLevel.HIGH,
                requires_confirmation=True,
                read_only=False,
            )
        },
    )

    result = runtime.call_tool("risky_write", {})

    assert result.status is ToolResultStatus.REQUIRES_CONFIRMATION
    assert result.error_code == "confirmation_required"
    assert result.content == (
        "Tool requires confirmation (confirmation_required): "
        "Tool 'risky_write' requires user confirmation before execution."
    )
    assert executions == []


def test_default_tool_runtime_exposes_metadata_and_low_risk_tools() -> None:
    runtime = get_default_tool_runtime()

    metadata = runtime.get_metadata()

    assert [tool.name for tool in runtime.get_tools()] == [
        "calculator",
        "echo",
        "list_files",
        "read_file",
        "search_text",
        "project_tree",
    ]
    assert metadata["calculator"].risk_level is ToolRiskLevel.LOW
    assert metadata["echo"].risk_level is ToolRiskLevel.LOW
    assert metadata["list_files"].risk_level is ToolRiskLevel.LOW
    assert metadata["read_file"].risk_level is ToolRiskLevel.LOW
    assert metadata["search_text"].risk_level is ToolRiskLevel.LOW
    assert metadata["project_tree"].risk_level is ToolRiskLevel.LOW
    for tool_name in [
        "calculator",
        "echo",
        "list_files",
        "read_file",
        "search_text",
        "project_tree",
    ]:
        assert metadata[tool_name].read_only is True
        assert metadata[tool_name].requires_confirmation is False


def test_list_files_lists_workspace_relative_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "src").mkdir()

    runtime = get_default_tool_runtime()

    result = runtime.call_tool("list_files", {"path": "."})

    assert result.status is ToolResultStatus.SUCCESS
    assert result.content.splitlines() == ["notes.txt", "src/"]


def test_list_files_rejects_paths_outside_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    runtime = get_default_tool_runtime()

    result = runtime.call_tool("list_files", {"path": ".."})

    assert result.status is ToolResultStatus.ERROR
    assert result.error_code == "tool_failed"
    assert result.error_type == "ValueError"
