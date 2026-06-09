from mini_agent.memory import Memory


def test_add_user_message() -> None:
    memory = Memory()

    memory.add_user_message("hello")

    assert memory.get_messages() == [{"role": "user", "content": "hello"}]


def test_add_assistant_message() -> None:
    memory = Memory()

    memory.add_assistant_message("hi")

    assert memory.get_messages() == [{"role": "assistant", "content": "hi"}]


def test_add_tool_message() -> None:
    memory = Memory()

    memory.add_tool_message("call_123", "14")

    assert memory.get_messages() == [
        {"role": "tool", "tool_call_id": "call_123", "content": "14"}
    ]


def test_clear_removes_all_messages() -> None:
    memory = Memory()
    memory.add_user_message("hello")
    memory.add_assistant_message("hi")

    memory.clear()

    assert memory.get_messages() == []
