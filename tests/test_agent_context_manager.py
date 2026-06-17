from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agentloop.agent import AgentLoop
from agentloop.context_manager import ContextBudget, ContextManager
from agentloop.memory import Memory
from agentloop.tools import get_default_tools


class ContextRecordingModel:
    """Fake model that records the exact context it receives."""

    def __init__(self) -> None:
        self.seen_messages: list[list[BaseMessage]] = []

    def bind_tools(self, tools: list) -> "ContextRecordingModel":
        self.tools = tools
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.seen_messages.append(list(messages))
        return AIMessage(content="ok")


def test_agent_uses_context_manager_before_model_call() -> None:
    memory = Memory()
    memory.set_messages(
        [
            SystemMessage(content="System instruction must stay."),
            HumanMessage(content="old user detail " * 90),
            AIMessage(content="old assistant detail " * 90),
        ]
    )
    model = ContextRecordingModel()
    agent = AgentLoop(
        model=model,
        tools=get_default_tools(),
        memory=memory,
        max_steps=4,
        context_manager=ContextManager(
            ContextBudget(
                max_tokens=190,
                reserved_output_tokens=20,
                summary_max_tokens=40,
            )
        ),
    )

    result = agent.run("current request")

    assert result == "ok"
    first_model_messages = model.seen_messages[0]
    assert str(first_model_messages[0].content) == "System instruction must stay."
    assert any(
        "Running summary" in str(message.content)
        for message in first_model_messages
    )
    assert str(first_model_messages[-1].content) == "current request"
    report = agent.get_context_report()
    assert report is not None
    assert any("summary" in line for line in report.to_display_lines())
