from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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

    def bind_tools(self, tools: list) -> "NeverFinalChatModel":
        self.tools = tools
        return self

    def invoke(self, messages: list) -> AIMessage:
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
    agent = MiniAgent(
        model=NeverFinalChatModel(),
        tools=get_default_tools(),
        memory=Memory(),
        max_steps=2,
    )

    result = agent.run("loop forever")

    assert result == "Agent stopped because it reached the maximum number of steps."
