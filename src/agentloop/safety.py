"""Safety policy primitives for tool execution and audit content."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class ContentSource(StrEnum):
    """Where untrusted content entered the runtime."""

    TOOL = "tool"
    FILE = "file"
    EXTERNAL = "external"


class TrustLevel(StrEnum):
    """Trust classification for content shown to or produced by the model."""

    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True)
class ToolAuthorization:
    """Safety decision for a tool call."""

    allowed: bool
    requires_confirmation: bool = False
    error_code: str | None = None
    message: str = ""


@dataclass(frozen=True)
class SanitizedContent:
    """Content after redaction, truncation, and trust labeling."""

    content: str
    source: ContentSource
    trust_level: TrustLevel
    truncated: bool = False


class SafetyPolicy:
    """Centralized safety rules for local-first agent execution."""

    def __init__(
        self,
        workspace_root: Path | None = None,
        tool_allowlist: set[str] | None = None,
        max_output_chars: int = 4_000,
    ) -> None:
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.tool_allowlist = (
            set(tool_allowlist) if tool_allowlist is not None else None
        )
        self.max_output_chars = max_output_chars

    def authorize_tool(
        self,
        metadata: Any,
        approved: bool = False,
    ) -> ToolAuthorization:
        """Return whether a tool call may execute under this policy."""

        if self.tool_allowlist is not None and metadata.name not in self.tool_allowlist:
            return ToolAuthorization(
                allowed=False,
                error_code="tool_not_allowed",
                message=f"Tool '{metadata.name}' is not allowed by SafetyPolicy.",
            )

        if (
            not approved
            and (
                metadata.requires_confirmation
                or str(metadata.risk_level) == "high"
            )
        ):
            return ToolAuthorization(
                allowed=False,
                requires_confirmation=True,
                error_code="confirmation_required",
                message=(
                    f"Tool '{metadata.name}' requires user confirmation before "
                    "execution."
                ),
            )

        return ToolAuthorization(allowed=True)

    def resolve_workspace_path(self, path: str) -> Path:
        """Resolve a path and reject traversal outside the workspace root."""

        if not path.strip():
            raise ValueError("Path cannot be empty.")
        target = (self.workspace_root / path).resolve()
        if not target.is_relative_to(self.workspace_root):
            raise ValueError("Path is outside the workspace.")
        return target

    def sanitize_output(
        self,
        content: str,
        source: ContentSource = ContentSource.TOOL,
    ) -> SanitizedContent:
        """Redact secrets, truncate large output, and mark it untrusted."""

        redacted = redact_secrets(content)
        truncated = False
        if len(redacted) > self.max_output_chars:
            redacted = redacted[: self.max_output_chars].rstrip()
            redacted = f"{redacted}\n...[truncated]"
            truncated = True
        return SanitizedContent(
            content=redacted,
            source=source,
            trust_level=TrustLevel.UNTRUSTED,
            truncated=truncated,
        )

    def mark_untrusted(
        self,
        content: str,
        source: ContentSource,
    ) -> SanitizedContent:
        """Return content explicitly labeled as untrusted."""

        return SanitizedContent(
            content=content,
            source=source,
            trust_level=TrustLevel.UNTRUSTED,
        )


_SECRET_PATTERNS = (
    re.compile(r"(?i)(MODEL_API_KEY|OPENAI_API_KEY|API_KEY|TOKEN)=([^\s]+)"),
    re.compile(r"(?i)(Authorization:\s*Bearer\s+)([A-Za-z0-9._\-]+)"),
    re.compile(r"\b(sk-[A-Za-z0-9][A-Za-z0-9._\-]{8,})\b"),
    re.compile(r"\b(ghp_[A-Za-z0-9]{8,})\b"),
    re.compile(r"\b(AKIA[0-9A-Z]{12,})\b"),
)


def redact_secrets(content: str) -> str:
    """Return content with common credential forms masked."""

    redacted = content
    redacted = _SECRET_PATTERNS[0].sub(r"\1=[REDACTED_SECRET]", redacted)
    redacted = _SECRET_PATTERNS[1].sub(r"\1[REDACTED_SECRET]", redacted)
    for pattern in _SECRET_PATTERNS[2:]:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted
