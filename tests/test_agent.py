from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from mini_agent.agent import MiniAgent
from mini_agent.memory import Memory
from mini_agent.tools import get_default_tools


class FakeChatModel:
    """Model client that asks for one calculation, then returns a final answer."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list) -> "FakeChatModel":
        self.tools = tools
        return self

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "calculator",
                        "args": {"expression": "12 * (3 + 4)"},
                    }
                ],
            )
        return AIMessage(content="The answer is 84.")


class NeverFinalChatModel:
    """Model client that never produces a final answer."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list) -> "NeverFinalChatModel":
        self.tools = tools
        return self

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_repeat",
                    "name": "echo",
                    "args": {"text": "keep going"},
                }
            ],
        )


class UnknownToolThenFinalModel:
    """Model that asks for an unknown tool, then reports the tool error."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list) -> "UnknownToolThenFinalModel":
        self.tools = tools
        return self

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_unknown",
                        "name": "missing_tool",
                        "args": {"text": "hello"},
                    }
                ],
            )
        return AIMessage(content=f"Recovered from: {messages[-1].content}")


class BadArgumentsThenFinalModel:
    """Model that calls calculator without required arguments, then recovers."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list) -> "BadArgumentsThenFinalModel":
        self.tools = tools
        return self

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_bad_args", "name": "calculator", "args": {}}
                ],
            )
        return AIMessage(content=f"Recovered from: {messages[-1].content}")


class FailingToolThenFinalModel:
    """Model that calls a tool which raises, then recovers."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list) -> "FailingToolThenFinalModel":
        self.tools = tools
        return self

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{"id": "call_broken", "name": "broken", "args": {}}],
            )
        return AIMessage(content=f"Recovered from: {messages[-1].content}")


@tool
def broken() -> str:
    """Always fail for runtime hardening tests."""

    raise RuntimeError("boom")


def test_agent_completes_tool_call_loop() -> None:
    memory = Memory()
    model = FakeChatModel()
    agent = MiniAgent(
        model=model,
        tools=get_default_tools(),
        memory=memory,
        max_steps=4,
    )

    result = agent.run("What is 12 * (3 + 4)?")

    assert result == "The answer is 84."
    assert model.calls == 2
    messages = memory.get_messages()
    assert len(messages) == 4
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "What is 12 * (3 + 4)?"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].tool_calls == [
        {
            "id": "call_1",
            "name": "calculator",
            "args": {"expression": "12 * (3 + 4)"},
            "type": "tool_call",
        }
    ]
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].content == "84"
    assert messages[2].tool_call_id == "call_1"
    assert isinstance(messages[3], AIMessage)
    assert messages[3].content == "The answer is 84."


def test_agent_stops_at_max_steps() -> None:
    model = NeverFinalChatModel()
    agent = MiniAgent(
        model=model,
        tools=get_default_tools(),
        memory=Memory(),
        max_steps=2,
    )

    result = agent.run("loop forever")

    assert result == "Agent stopped because it reached the maximum number of steps."
    assert model.calls == 2


def test_agent_returns_tool_message_for_unknown_tool() -> None:
    memory = Memory()
    agent = MiniAgent(
        model=UnknownToolThenFinalModel(),
        tools=get_default_tools(),
        memory=memory,
        max_steps=3,
    )

    result = agent.run("call a missing tool")

    assert result == (
        "Recovered from: Tool error (unknown_tool): Tool 'missing_tool' "
        "is not registered."
    )
    tool_message = memory.get_messages()[2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "call_unknown"
    assert tool_message.content == (
        "Tool error (unknown_tool): Tool 'missing_tool' is not registered."
    )


def test_agent_returns_tool_message_for_bad_tool_arguments() -> None:
    memory = Memory()
    agent = MiniAgent(
        model=BadArgumentsThenFinalModel(),
        tools=get_default_tools(),
        memory=memory,
        max_steps=3,
    )

    result = agent.run("call calculator incorrectly")

    assert result.startswith("Recovered from: Tool error (tool_failed):")
    assert "Tool 'calculator' failed." in result
    tool_message = memory.get_messages()[2]
    assert isinstance(tool_message, ToolMessage)
    assert str(tool_message.content).startswith("Tool error (tool_failed):")


def test_agent_returns_tool_message_for_tool_runtime_error() -> None:
    memory = Memory()
    agent = MiniAgent(
        model=FailingToolThenFinalModel(),
        tools=[broken],
        memory=memory,
        max_steps=3,
    )

    result = agent.run("call broken")

    assert result == "Recovered from: Tool error (tool_failed): Tool 'broken' failed."
    tool_message = memory.get_messages()[2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Tool error (tool_failed): Tool 'broken' failed."
