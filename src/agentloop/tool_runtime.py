"""Tool runtime primitives for registered agent capabilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_core.tools import BaseTool


class ToolRiskLevel(StrEnum):
    """Risk level for a tool capability."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolResultStatus(StrEnum):
    """Normalized status for a tool call."""

    SUCCESS = "success"
    ERROR = "error"
    REJECTED = "rejected"
    REQUIRES_CONFIRMATION = "requires_confirmation"


@dataclass(frozen=True)
class ToolMetadata:
    """Metadata used to reason about tool safety and display."""

    name: str
    description: str
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW
    requires_confirmation: bool = False
    read_only: bool = True
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class ToolResult:
    """Normalized result of one tool call."""

    tool_name: str
    status: ToolResultStatus
    content: str
    metadata: ToolMetadata
    duration_ms: float
    error_code: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_message_content(self) -> str:
        """Return the content that should be sent back to the model."""

        return self.content

    def to_message_kwargs(self) -> dict[str, Any]:
        """Return metadata for a LangChain ToolMessage."""

        kwargs: dict[str, Any] = {
            "status": self.status.value,
            "tool_name": self.tool_name,
            "duration_ms": self.duration_ms,
            "risk_level": self.metadata.risk_level.value,
            "requires_confirmation": self.metadata.requires_confirmation,
            "read_only": self.metadata.read_only,
        }
        if self.error_code is not None:
            kwargs["error_code"] = self.error_code
        if self.error_type is not None:
            kwargs["error_type"] = self.error_type
        if self.error_message is not None:
            kwargs["error_message"] = self.error_message
        return kwargs


class ToolRuntime:
    """Registers tools and normalizes tool execution results."""

    def __init__(
        self,
        tools: list[BaseTool],
        metadata: dict[str, ToolMetadata],
    ) -> None:
        self._tools_by_name = {tool.name: tool for tool in tools}
        self._metadata = dict(metadata)

    @classmethod
    def from_tools(
        cls,
        tools: list[BaseTool],
        metadata: dict[str, ToolMetadata] | None = None,
    ) -> ToolRuntime:
        """Create a runtime with low-risk read-only defaults."""

        metadata_by_name = {
            tool.name: ToolMetadata(
                name=tool.name,
                description=tool.description or "",
            )
            for tool in tools
        }
        if metadata is not None:
            metadata_by_name.update(metadata)
        return cls(tools=tools, metadata=metadata_by_name)

    def get_tools(self) -> list[BaseTool]:
        """Return registered LangChain tools."""

        return list(self._tools_by_name.values())

    def get_metadata(self) -> dict[str, ToolMetadata]:
        """Return registered tool metadata by tool name."""

        return dict(self._metadata)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Call a tool and return a normalized result."""

        started = time.perf_counter()
        metadata = self._metadata.get(
            name,
            ToolMetadata(name=name, description=""),
        )
        tool = self._tools_by_name.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name,
                status=ToolResultStatus.ERROR,
                content=(
                    "Tool error (unknown_tool): "
                    f"Tool '{name}' is not registered."
                ),
                metadata=metadata,
                duration_ms=_elapsed_ms(started),
                error_code="unknown_tool",
            )

        try:
            result = tool.invoke(arguments)
        except Exception as exc:
            return ToolResult(
                tool_name=name,
                status=ToolResultStatus.ERROR,
                content=f"Tool error (tool_failed): Tool '{name}' failed.",
                metadata=metadata,
                duration_ms=_elapsed_ms(started),
                error_code="tool_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        return ToolResult(
            tool_name=name,
            status=ToolResultStatus.SUCCESS,
            content=str(result),
            metadata=metadata,
            duration_ms=_elapsed_ms(started),
        )


def _elapsed_ms(started: float) -> float:
    """Return elapsed milliseconds since a perf counter value."""

    return round((time.perf_counter() - started) * 1000, 3)
