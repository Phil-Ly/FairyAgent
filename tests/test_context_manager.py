from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agentloop.context_manager import (
    ContextBudget,
    ContextComponentKind,
    ContextManager,
)
from agentloop.memory_store import MemoryNamespace, MemoryRecord, MemoryStatus


def test_context_manager_trims_old_messages_into_running_summary() -> None:
    manager = ContextManager(
        ContextBudget(
            max_tokens=180,
            reserved_output_tokens=20,
            summary_max_tokens=40,
        )
    )
    messages = [
        SystemMessage(content="You are a careful coding agent."),
        HumanMessage(content="old user detail " * 60),
        AIMessage(content="old assistant detail " * 60),
        HumanMessage(content="current task: implement context manager"),
    ]

    managed = manager.prepare(messages)

    assert (
        managed.composition.estimated_tokens
        <= managed.composition.input_token_budget
    )
    assert any(
        "Running summary" in str(message.content)
        for message in managed.messages
    )
    assert (
        str(managed.messages[-1].content)
        == "current task: implement context manager"
    )
    assert ContextComponentKind.SUMMARY in [
        component.kind for component in managed.composition.components
    ]


def test_context_manager_preserves_tool_call_chain_when_trimming() -> None:
    manager = ContextManager(
        ContextBudget(
            max_tokens=220,
            reserved_output_tokens=20,
            summary_max_tokens=30,
        )
    )
    tool_calling_message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_keep",
                "name": "calculator",
                "args": {"expression": "2 + 2"},
            }
        ],
    )
    messages = [
        HumanMessage(content="old user detail " * 80),
        AIMessage(content="old assistant detail " * 80),
        HumanMessage(content="please calculate"),
        tool_calling_message,
        ToolMessage(content="4", tool_call_id="call_keep"),
        HumanMessage(content="use the tool result"),
    ]

    managed = manager.prepare(messages)

    retained_tool_messages = [
        message for message in managed.messages if isinstance(message, ToolMessage)
    ]
    assert len(retained_tool_messages) == 1
    retained_tool_ids = {
        tool_call["id"]
        for message in managed.messages
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    }
    assert retained_tool_messages[0].tool_call_id in retained_tool_ids


def test_context_manager_drops_orphan_tool_messages_from_model_context() -> None:
    manager = ContextManager(
        ContextBudget(
            max_tokens=120,
            reserved_output_tokens=20,
            summary_max_tokens=20,
        )
    )
    messages = [
        HumanMessage(content="hello"),
        ToolMessage(content="orphan result", tool_call_id="missing_parent"),
        HumanMessage(content="latest user request"),
    ]

    managed = manager.prepare(messages)

    assert not any(isinstance(message, ToolMessage) for message in managed.messages)
    assert str(managed.messages[-1].content) == "latest user request"


def test_context_manager_reports_original_summary_and_long_term_memory() -> None:
    manager = ContextManager(
        ContextBudget(
            max_tokens=220,
            reserved_output_tokens=20,
            summary_max_tokens=40,
        )
    )
    memory = MemoryRecord(
        memory_id="memory_1",
        namespace=MemoryNamespace.PROJECT,
        status=MemoryStatus.ACTIVE,
        content="Project prefers SQLite-backed local persistence.",
        source_session_id="session_old",
        source_run_id="run_old",
        confirmed_by_user=True,
        metadata={},
        created_at="2026-06-17T00:00:00Z",
        updated_at="2026-06-17T00:00:00Z",
    )
    messages = [
        HumanMessage(content="old planning detail " * 80),
        AIMessage(content="old implementation detail " * 80),
        HumanMessage(content="ship the next slice"),
    ]

    managed = manager.prepare(messages, long_term_memories=[memory])
    display_lines = managed.composition.to_display_lines()

    assert ContextComponentKind.LONG_TERM_MEMORY in [
        component.kind for component in managed.composition.components
    ]
    assert ContextComponentKind.SUMMARY in [
        component.kind for component in managed.composition.components
    ]
    assert ContextComponentKind.ORIGINAL in [
        component.kind for component in managed.composition.components
    ]
    assert any("long_term_memory" in line for line in display_lines)
    assert any("summary" in line for line in display_lines)
    assert any("original" in line for line in display_lines)
