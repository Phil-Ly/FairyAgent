from pathlib import Path

from agentloop.safety import ContentSource, SafetyPolicy, TrustLevel
from agentloop.tool_runtime import ToolMetadata, ToolRiskLevel


def test_safety_policy_rejects_tool_not_in_allowlist(tmp_path: Path) -> None:
    policy = SafetyPolicy(workspace_root=tmp_path, tool_allowlist={"calculator"})
    metadata = ToolMetadata(
        name="run_shell",
        description="Run shell commands.",
        risk_level=ToolRiskLevel.HIGH,
        requires_confirmation=True,
        read_only=False,
    )

    decision = policy.authorize_tool(metadata, approved=True)

    assert decision.allowed is False
    assert decision.error_code == "tool_not_allowed"
    assert "not allowed" in decision.message


def test_safety_policy_requires_confirmation_for_high_risk_tool(
    tmp_path: Path,
) -> None:
    policy = SafetyPolicy(workspace_root=tmp_path, tool_allowlist={"write_file"})
    metadata = ToolMetadata(
        name="write_file",
        description="Write a file.",
        risk_level=ToolRiskLevel.HIGH,
        requires_confirmation=True,
        read_only=False,
    )

    blocked = policy.authorize_tool(metadata, approved=False)
    approved = policy.authorize_tool(metadata, approved=True)

    assert blocked.allowed is False
    assert blocked.requires_confirmation is True
    assert blocked.error_code == "confirmation_required"
    assert approved.allowed is True
    assert approved.requires_confirmation is False


def test_safety_policy_redacts_secrets_and_truncates_output(tmp_path: Path) -> None:
    policy = SafetyPolicy(
        workspace_root=tmp_path,
        tool_allowlist={"echo"},
        max_output_chars=80,
    )
    raw = (
        "MODEL_API_KEY=sk-1234567890abcdef\n"
        "Authorization: Bearer live-token-value\n"
        "normal output that should be truncated after the configured limit"
    )

    sanitized = policy.sanitize_output(raw)

    assert "sk-1234567890abcdef" not in sanitized.content
    assert "live-token-value" not in sanitized.content
    assert "MODEL_API_KEY=[REDACTED_SECRET]" in sanitized.content
    assert "Authorization: Bearer [REDACTED_SECRET]" in sanitized.content
    assert sanitized.truncated is True
    assert sanitized.content.endswith("...[truncated]")


def test_safety_policy_marks_file_and_external_content_as_untrusted(
    tmp_path: Path,
) -> None:
    policy = SafetyPolicy(workspace_root=tmp_path)

    file_content = policy.mark_untrusted("hello", source=ContentSource.FILE)
    external_content = policy.mark_untrusted("payload", source=ContentSource.EXTERNAL)

    assert file_content.trust_level is TrustLevel.UNTRUSTED
    assert file_content.source is ContentSource.FILE
    assert external_content.trust_level is TrustLevel.UNTRUSTED
    assert external_content.source is ContentSource.EXTERNAL


def test_safety_policy_resolves_workspace_paths(tmp_path: Path) -> None:
    policy = SafetyPolicy(workspace_root=tmp_path)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")

    resolved = policy.resolve_workspace_path("notes.txt")

    assert resolved == (tmp_path / "notes.txt").resolve()
