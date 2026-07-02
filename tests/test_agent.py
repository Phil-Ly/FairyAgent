from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agentloop.agent import AgentLoop
from agentloop.intervention import InterventionAction, InterventionWorkflow
from agentloop.memory import Memory
from agentloop.tool_runtime import ToolMetadata, ToolRiskLevel, ToolRuntime
from agentloop.tools import get_default_tool_runtime, get_default_tools


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


class ContentBlockChatModel:
    """Model client that returns provider-style content blocks."""

    def bind_tools(self, tools: list) -> "ContentBlockChatModel":
        self.tools = tools
        return self

    def invoke(self, messages: list) -> AIMessage:
        return AIMessage(
            content=[
                {"type": "thinking", "thinking": "internal"},
                {"type": "text", "text": "A normalized answer."},
            ]
        )


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


class HighRiskToolModel:
    """Model that requests one high-risk tool call."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list) -> "HighRiskToolModel":
        self.tools = tools
        return self

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_risky",
                        "name": "risky_write",
                        "args": {},
                    }
                ],
            )
        return AIMessage(content="should not continue without confirmation")


class HighRiskToolThenFinalModel:
    """Model that requests one high-risk tool, then summarizes its result."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list) -> "HighRiskToolThenFinalModel":
        self.tools = tools
        return self

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_risky_resume",
                        "name": "risky_write",
                        "args": {},
                    }
                ],
            )
        return AIMessage(content=f"Approved result: {messages[-1].content}")


class RepeatedFailingToolModel:
    """Model that repeats the same failing tool call."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list) -> "RepeatedFailingToolModel":
        self.tools = tools
        return self

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"call_bad_args_{self.calls}",
                    "name": "calculator",
                    "args": {},
                }
            ],
        )


class RepeatedFailingThenFinalModel:
    """Model that repeats bad calls until the user allows continuation."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list) -> "RepeatedFailingThenFinalModel":
        self.tools = tools
        return self

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        if self.calls <= 3:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"call_bad_args_resume_{self.calls}",
                        "name": "calculator",
                        "args": {},
                    }
                ],
            )
        return AIMessage(content="Continuing after intervention.")


@tool
def broken() -> str:
    """Always fail for runtime hardening tests."""

    raise RuntimeError("boom")


@tool
def risky_write() -> str:
    """Write something risky."""

    return "wrote"


def test_agent_completes_tool_call_loop() -> None:
    memory = Memory()
    model = FakeChatModel()
    agent = AgentLoop(
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
    assert messages[2].additional_kwargs["status"] == "success"
    assert messages[2].additional_kwargs["risk_level"] == "low"
    assert messages[2].additional_kwargs["duration_ms"] >= 0
    assert isinstance(messages[3], AIMessage)
    assert messages[3].content == "The answer is 84."


def test_agent_returns_text_from_provider_content_blocks() -> None:
    agent = AgentLoop(
        model=ContentBlockChatModel(),
        tools=get_default_tools(),
        memory=Memory(),
        max_steps=2,
    )

    result = agent.run("Answer with content blocks.")

    assert result == "A normalized answer."


def test_agent_accepts_tool_runtime_and_binds_langchain_tools() -> None:
    model = FakeChatModel()

    AgentLoop(
        model=model,
        tools=get_default_tool_runtime(),
        memory=Memory(),
        max_steps=4,
    )

    assert [tool.name for tool in model.tools] == [
        "calculator",
        "echo",
        "list_files",
        "read_file",
        "search_text",
        "project_tree",
    ]


def test_agent_stops_at_max_steps() -> None:
    memory = Memory()
    model = NeverFinalChatModel()
    agent = AgentLoop(
        model=model,
        tools=get_default_tools(),
        memory=memory,
        max_steps=2,
    )

    result = agent.run("loop forever")

    assert result == "Agent stopped because it reached the maximum number of steps."
    assert model.calls == 2
    assert memory.get_messages()[-1].additional_kwargs["stop_reason"] == (
        "step_limit_risk"
    )


def test_agent_returns_tool_message_for_unknown_tool() -> None:
    memory = Memory()
    agent = AgentLoop(
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
    assert tool_message.additional_kwargs["status"] == "error"
    assert tool_message.additional_kwargs["error_code"] == "unknown_tool"


def test_agent_returns_tool_message_for_bad_tool_arguments() -> None:
    memory = Memory()
    agent = AgentLoop(
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
    agent = AgentLoop(
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


def test_agent_stops_for_high_risk_tool_confirmation() -> None:
    memory = Memory()
    model = HighRiskToolModel()
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
        model=model,
        tools=runtime,
        memory=memory,
        max_steps=4,
    )

    result = agent.run("write a file")

    assert model.calls == 1
    assert result == (
        "Intervention required: Tool 'risky_write' requires user confirmation "
        "before execution."
    )
    messages = memory.get_messages()
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].additional_kwargs["status"] == "requires_confirmation"
    assert isinstance(messages[-1], AIMessage)
    intervention = messages[-1].additional_kwargs["intervention_request"]
    assert intervention["reason"] == "high_risk_action"
    assert intervention["tool_name"] == "risky_write"
    assert intervention["recommended_option"] == "reject"
    assert intervention["options"] == ["approve", "reject", "edit", "stop"]


def test_agent_stops_after_repeated_tool_failures() -> None:
    memory = Memory()
    model = RepeatedFailingToolModel()
    agent = AgentLoop(
        model=model,
        tools=get_default_tools(),
        memory=memory,
        max_steps=8,
    )

    result = agent.run("keep trying calculator incorrectly")

    assert model.calls == 3
    assert result == (
        "Intervention required: Tool 'calculator' failed 3 consecutive times."
    )
    tool_messages = [
        message
        for message in memory.get_messages()
        if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 3
    assert isinstance(memory.get_messages()[-1], AIMessage)
    intervention = memory.get_messages()[-1].additional_kwargs[
        "intervention_request"
    ]
    assert intervention["reason"] == "repeated_failure"
    assert intervention["tool_name"] == "calculator"
    assert intervention["failure_count"] == 3


def test_agent_resumes_high_risk_tool_after_approval() -> None:
    memory = Memory()
    model = HighRiskToolThenFinalModel()
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
        model=model,
        tools=runtime,
        memory=memory,
        max_steps=4,
    )
    workflow = InterventionWorkflow()

    agent.run("write a file")
    pending = agent.get_pending_intervention()
    assert pending is not None
    decision = workflow.create_decision(
        pending.request,
        action=InterventionAction.APPROVE,
    )

    result = agent.resolve_intervention(decision)

    assert result == "Approved result: wrote"
    assert model.calls == 2
    assert agent.get_pending_intervention() is None
    tool_messages = [
        message
        for message in memory.get_messages()
        if isinstance(message, ToolMessage)
    ]
    assert tool_messages[-1].content == "wrote"
    assert tool_messages[-1].additional_kwargs["status"] == "success"


def test_agent_resumes_after_repeated_failure_continue() -> None:
    memory = Memory()
    model = RepeatedFailingThenFinalModel()
    agent = AgentLoop(
        model=model,
        tools=get_default_tools(),
        memory=memory,
        max_steps=8,
    )
    workflow = InterventionWorkflow()

    agent.run("keep trying calculator incorrectly")
    pending = agent.get_pending_intervention()
    assert pending is not None
    decision = workflow.create_decision(
        pending.request,
        action=InterventionAction.CONTINUE,
    )

    result = agent.resolve_intervention(decision)

    assert result == "Continuing after intervention."
    assert model.calls == 4
    assert agent.get_pending_intervention() is None
