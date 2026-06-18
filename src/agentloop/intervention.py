"""Human-in-the-loop intervention protocol primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class InterventionReason(StrEnum):
    """Reasons the runtime stops automatic execution for user input."""

    HIGH_RISK_ACTION = "high_risk_action"
    REPEATED_FAILURE = "repeated_failure"
    STEP_LIMIT_RISK = "step_limit_risk"


class InterventionAction(StrEnum):
    """User actions that can resolve or resume an intervention."""

    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    CONTINUE = "continue"
    SKIP_TOOL = "skip_tool"
    STOP = "stop"


@dataclass(frozen=True)
class InterventionRequest:
    """Structured request for user judgment before execution continues."""

    reason: InterventionReason
    message: str
    tool_name: str | None = None
    failure_count: int | None = None
    recommended_option: str = InterventionAction.STOP.value
    options: tuple[str, ...] = (InterventionAction.STOP.value,)
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
            recommended_option=InterventionAction.REJECT.value,
            options=(
                InterventionAction.APPROVE.value,
                InterventionAction.REJECT.value,
                InterventionAction.EDIT.value,
                InterventionAction.STOP.value,
            ),
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
            recommended_option=InterventionAction.STOP.value,
            options=(
                InterventionAction.CONTINUE.value,
                InterventionAction.SKIP_TOOL.value,
                InterventionAction.EDIT.value,
                InterventionAction.STOP.value,
            ),
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


@dataclass(frozen=True)
class DecisionRecord:
    """Structured user decision for an intervention request."""

    resume_token: str
    action: InterventionAction
    edited_tool_args: dict[str, Any] | None = None
    user_note: str | None = None
    decision_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="microseconds")
    )

    def to_message_kwargs(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for message metadata."""

        data: dict[str, Any] = {
            "decision_id": self.decision_id,
            "resume_token": self.resume_token,
            "action": self.action.value,
            "created_at": self.created_at,
        }
        if self.edited_tool_args is not None:
            data["edited_tool_args"] = self.edited_tool_args
        if self.user_note is not None:
            data["user_note"] = self.user_note
        return data


class InterventionWorkflow:
    """Validate user intervention actions and create decision records."""

    def create_decision(
        self,
        request: InterventionRequest,
        action: InterventionAction | str,
        edited_tool_args: dict[str, Any] | None = None,
        user_note: str | None = None,
    ) -> DecisionRecord:
        """Return a validated decision for a pending intervention request."""

        normalized_action = InterventionAction(action)
        if normalized_action.value not in request.options:
            raise ValueError(
                f"Action '{normalized_action.value}' is not allowed for "
                f"intervention reason '{request.reason.value}'."
            )
        return DecisionRecord(
            resume_token=request.resume_token,
            action=normalized_action,
            edited_tool_args=edited_tool_args,
            user_note=user_note,
        )

    def format_pending_request(self, request: InterventionRequest) -> list[str]:
        """Return CLI-friendly lines for a pending intervention."""

        lines = [
            f"pending: {request.reason.value}",
            f"message: {request.message}",
            f"resume-token: {request.resume_token}",
            f"actions: {', '.join(request.options)}",
        ]
        if request.tool_name is not None:
            lines.append(f"tool: {request.tool_name}")
        return lines
