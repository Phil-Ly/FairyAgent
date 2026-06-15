from agentloop.agent import AgentLoop
from agentloop.chat_session import ChatSession
from agentloop.memory import Memory
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
