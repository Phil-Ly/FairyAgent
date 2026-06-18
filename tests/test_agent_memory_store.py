from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from agentloop.agent import AgentLoop
from agentloop.memory import Memory
from agentloop.memory_store import MemoryNamespace, SQLiteMemoryStore
from agentloop.tools import get_default_tools


class MemoryAwareModel:
    """Fake model that records the messages it receives."""

    def __init__(self) -> None:
        self.seen_messages: list[list[BaseMessage]] = []

    def bind_tools(self, tools: list) -> "MemoryAwareModel":
        self.tools = tools
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.seen_messages.append(list(messages))
        return AIMessage(content="ok")


def test_agent_injects_active_confirmed_memories_for_new_session(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    active = store.create_memory(
        namespace=MemoryNamespace.USER,
        content="User prefers concise Chinese responses.",
        source_session_id="session_old",
        source_run_id="run_old",
        confirmed_by_user=True,
    )
    disabled = store.create_memory(
        namespace=MemoryNamespace.PROJECT,
        content="Do not inject this disabled memory.",
        source_session_id="session_old",
        source_run_id="run_old",
        confirmed_by_user=True,
    )
    store.disable_memory(disabled.memory_id)
    memory = Memory()
    model = MemoryAwareModel()
    agent = AgentLoop(
        model=model,
        tools=get_default_tools(),
        memory=memory,
        max_steps=4,
        memory_store=store,
    )

    result = agent.run("hello")

    assert result == "ok"
    first_model_messages = model.seen_messages[0]
    assert isinstance(first_model_messages[0], SystemMessage)
    assert "Long-term memory" in str(first_model_messages[0].content)
    assert active.content in str(first_model_messages[0].content)
    assert disabled.content not in str(first_model_messages[0].content)


def test_agent_does_not_duplicate_memory_injection_on_later_turns(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    store.create_memory(
        namespace=MemoryNamespace.PROJECT,
        content="Project uses SQLite.",
        source_session_id="session_old",
        source_run_id="run_old",
        confirmed_by_user=True,
    )
    memory = Memory()
    model = MemoryAwareModel()
    agent = AgentLoop(
        model=model,
        tools=get_default_tools(),
        memory=memory,
        max_steps=4,
        memory_store=store,
    )

    agent.run("first")
    agent.run("second")

    system_messages = [
        message
        for message in memory.get_messages()
        if isinstance(message, SystemMessage)
    ]
    assert len(system_messages) == 1
