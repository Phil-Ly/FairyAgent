from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from agentloop.agent import AgentLoop
from agentloop.chat_session import ChatSession, format_ai_message
from agentloop.memory import Memory
from agentloop.tool_runtime import ToolMetadata, ToolRiskLevel, ToolRuntime
from agentloop.tools import get_default_tools
from tests.fakes import DeterministicCalculatorModel


def build_test_session() -> ChatSession:
    agent = AgentLoop(
        model=DeterministicCalculatorModel(),
        tools=get_default_tools(),
        memory=Memory(),
        max_steps=4,
    )
    return ChatSession(agent=agent)


def test_format_ai_message_extracts_text_from_content_blocks() -> None:
    message = AIMessage(
        content=[
            {"type": "thinking", "thinking": "internal"},
            {"type": "text", "text": "A normalized answer."},
        ]
    )

    assert format_ai_message(message) == ["assistant: A normalized answer."]


class HighRiskThenFinalModel:
    """Fake model that requests one high-risk tool and then returns final text."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list) -> "HighRiskThenFinalModel":
        self.tools = tools
        return self

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_risky_chat", "name": "risky_write", "args": {}}
                ],
            )
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage):
            return AIMessage(content=f"Done: {last_message.content}")
        return AIMessage(content="Done.")


@tool
def risky_write() -> str:
    """Write something risky."""

    return "wrote"


def build_high_risk_session() -> ChatSession:
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
    agent = AgentLoop(
        model=HighRiskThenFinalModel(),
        tools=runtime,
        memory=Memory(),
        max_steps=4,
    )
    return ChatSession(agent=agent)


def test_chat_session_handles_multiple_turns_with_shared_memory() -> None:
    session = build_test_session()

    first_outputs = session.handle_line("What is 2 + 3 * 4?")
    second_outputs = session.handle_line("What is 1 + 1?")
    history_outputs = session.handle_line("/history")

    assert first_outputs == [
        "tool-call: calculator {'expression': '2 + 3 * 4'}",
        "tool-result: calculator 14",
        "assistant: The answer is 14.",
    ]
    assert second_outputs == [
        "tool-call: calculator {'expression': '1 + 1'}",
        "tool-result: calculator 2",
        "assistant: The answer is 2.",
    ]
    assert "user: What is 2 + 3 * 4?" in history_outputs
    assert "assistant: The answer is 14." in history_outputs
    assert len(session.agent.memory.get_messages()) == 8


def test_chat_session_supports_clear_history_tools_and_doctor() -> None:
    session = build_test_session()

    session.handle_line("What is 2 + 2?")
    clear_outputs = session.handle_line("/clear")
    history_outputs = session.handle_line("/history")
    tools_outputs = session.handle_line("/tools")
    doctor_outputs = session.handle_line("/doctor")

    assert clear_outputs == ["system: memory cleared"]
    assert history_outputs == ["history: empty"]
    assert any("calculator:" in output for output in tools_outputs)
    assert "AgentLoop doctor" in doctor_outputs
    assert session.agent.memory.get_messages() == []


def test_chat_session_displays_context_report() -> None:
    session = build_test_session()

    empty_outputs = session.handle_line("/context")
    session.handle_line("What is 2 + 2?")
    context_outputs = session.handle_line("/context")

    assert empty_outputs == ["context: none"]
    assert context_outputs[0].startswith("context: estimated_tokens=")
    assert any("original" in output for output in context_outputs)


def test_chat_session_returns_exit_signal() -> None:
    session = build_test_session()

    outputs = session.handle_line("/exit")

    assert outputs == ["system: goodbye"]
    assert session.should_exit is True


def test_chat_session_ignores_empty_input() -> None:
    session = build_test_session()

    outputs = session.handle_line("  ")

    assert outputs == []
    assert session.should_exit is False


def test_chat_session_displays_and_approves_pending_intervention() -> None:
    session = build_high_risk_session()

    first_outputs = session.handle_line("write a file")
    pending_outputs = session.handle_line("/pending")
    approval_outputs = session.handle_line("/approve")

    assert first_outputs == [
        "tool-call: risky_write {}",
        (
            "tool-result: risky_write Tool requires confirmation "
            "(confirmation_required): Tool 'risky_write' requires user confirmation "
            "before execution."
        ),
        (
            "assistant: Intervention required: Tool 'risky_write' requires user "
            "confirmation before execution."
        ),
    ]
    assert "pending: high_risk_action" in pending_outputs
    assert "actions: approve, reject, edit, stop" in pending_outputs
    assert approval_outputs == [
        "tool-result: risky_write wrote",
        "assistant: Done: wrote",
    ]
    assert session.agent.get_pending_intervention() is None


def test_chat_session_reports_when_no_intervention_is_pending() -> None:
    session = build_test_session()

    assert session.handle_line("/pending") == ["pending: none"]
    assert session.handle_line("/approve") == ["pending: none"]
