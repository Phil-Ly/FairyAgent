from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentloop.memory import Memory


def test_add_user_message() -> None:
    memory = Memory()

    memory.add_user_message("hello")

    assert memory.get_messages() == [HumanMessage(content="hello")]


def test_add_assistant_message() -> None:
    memory = Memory()

    memory.add_assistant_message("hi")

    assert memory.get_messages() == [AIMessage(content="hi")]


def test_add_tool_message() -> None:
    memory = Memory()

    memory.add_tool_message("call_123", "14")

    assert memory.get_messages() == [ToolMessage(content="14", tool_call_id="call_123")]


def test_set_messages_replaces_messages() -> None:
    memory = Memory()
    messages = [HumanMessage(content="hello"), AIMessage(content="hi")]

    memory.set_messages(messages)

    assert memory.get_messages() == messages


def test_clear_removes_all_messages() -> None:
    memory = Memory()
    memory.add_user_message("hello")
    memory.add_assistant_message("hi")

    memory.clear()

    assert memory.get_messages() == []
