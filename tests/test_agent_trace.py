from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from agentloop.agent import AgentLoop
from agentloop.intervention import InterventionAction, InterventionWorkflow
from agentloop.memory import Memory
from agentloop.tool_runtime import ToolMetadata, ToolRiskLevel, ToolRuntime
from agentloop.tools import get_default_tools
from agentloop.trace_store import SQLiteTraceStore, TraceEventType, TraceRunStatus
from tests.fakes import DeterministicCalculatorModel


def test_agent_writes_trace_events_for_successful_tool_loop(tmp_path: Path) -> None:
    trace_store = SQLiteTraceStore(tmp_path / "trace.sqlite3")
    agent = AgentLoop(
        model=DeterministicCalculatorModel(),
        tools=get_default_tools(),
        memory=Memory(),
        max_steps=4,
        trace_store=trace_store,
        session_id="session_1",
    )

    result = agent.run("What is 2 + 3 * 4?")

    runs = trace_store.list_runs("session_1")
    events = trace_store.list_events(runs[0].run_id)
    assert result == "The answer is 14."
    assert len(runs) == 1
    assert runs[0].status is TraceRunStatus.COMPLETED
    assert runs[0].stop_reason == "final_answer"
    assert [event.event_type for event in events] == [
        TraceEventType.MODEL_CALL,
        TraceEventType.TOOL_CALL,
        TraceEventType.TOOL_RESULT,
        TraceEventType.MODEL_CALL,
    ]
    assert events[1].payload["tool_name"] == "calculator"
    assert events[2].payload["status"] == "success"


class HighRiskThenFinalModel:
    """Fake model that requests a high-risk tool and summarizes its result."""

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
                    {"id": "call_risky_trace", "name": "risky_write", "args": {}}
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


def test_agent_writes_intervention_and_decision_trace_events(
    tmp_path: Path,
) -> None:
    trace_store = SQLiteTraceStore(tmp_path / "trace.sqlite3")
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
        trace_store=trace_store,
        session_id="session_2",
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

    runs = trace_store.list_runs("session_2")
    events = trace_store.list_events(runs[0].run_id)
    assert result == "Done: wrote"
    assert runs[0].status is TraceRunStatus.COMPLETED
    assert TraceEventType.INTERVENTION_REQUEST in [
        event.event_type for event in events
    ]
    assert TraceEventType.DECISION in [event.event_type for event in events]
    assert events[-1].event_type is TraceEventType.MODEL_CALL
