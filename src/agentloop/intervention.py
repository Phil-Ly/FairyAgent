"""Human-in-the-loop intervention protocol primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class InterventionReason(StrEnum):
    """Reasons the runtime stops automatic execution for user input."""

    HIGH_RISK_ACTION = "high_risk_action"
    REPEATED_FAILURE = "repeated_failure"
    STEP_LIMIT_RISK = "step_limit_risk"


@dataclass(frozen=True)
class InterventionRequest:
    """Structured request for user judgment before execution continues."""

    reason: InterventionReason
    message: str
    tool_name: str | None = None
    failure_count: int | None = None
    recommended_option: str = "stop"
    options: tuple[str, ...] = ("stop", "continue")
    resume_token: str = field(default_factory=lambda: uuid4().hex)

    @classmethod
    def for_high_risk_tool(cls, tool_name: str) -> InterventionRequest:
        """Build an intervention request for a high-risk tool call."""

        return cls(
            reason=InterventionReason.HIGH_RISK_ACTION,
            message=(
                "Intervention required: "
                f"Tool '{tool_name}' requires user confirmation before execution."
            ),
            tool_name=tool_name,
            recommended_option="stop",
            options=("stop", "confirm"),
        )

    @classmethod
    def for_repeated_failure(
        cls,
        tool_name: str,
        failure_count: int,
    ) -> InterventionRequest:
        """Build an intervention request for repeated tool failures."""

        return cls(
            reason=InterventionReason.REPEATED_FAILURE,
            message=(
                "Intervention required: "
                f"Tool '{tool_name}' failed {failure_count} consecutive times."
            ),
            tool_name=tool_name,
            failure_count=failure_count,
            recommended_option="stop",
            options=("stop", "continue", "skip_tool"),
        )

    def to_message_kwargs(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for message metadata."""

        data: dict[str, Any] = {
            "reason": self.reason.value,
            "message": self.message,
            "recommended_option": self.recommended_option,
            "options": list(self.options),
            "resume_token": self.resume_token,
        }
        if self.tool_name is not None:
            data["tool_name"] = self.tool_name
        if self.failure_count is not None:
            data["failure_count"] = self.failure_count
        return data
