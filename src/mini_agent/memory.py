"""Short-term LangChain message memory."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


class Memory:
    """Stores messages for a single agent run."""

    def __init__(self) -> None:
        self.messages: list[BaseMessage] = []

    def add_user_message(self, content: str) -> None:
        """Append a user message."""

        self.messages.append(HumanMessage(content=content))

    def add_assistant_message(self, content: str) -> None:
        """Append a final assistant message."""

        self.messages.append(AIMessage(content=content))

    def add_tool_message(self, tool_call_id: str, content: str) -> None:
        """Append a tool result message."""

        self.messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))

    def get_messages(self) -> list[BaseMessage]:
        """Return a shallow copy of the current message list."""

        return list(self.messages)

    def set_messages(self, messages: list[BaseMessage]) -> None:
        """Replace the current message list."""

        self.messages = list(messages)

    def clear(self) -> None:
        """Remove all messages."""

        self.messages.clear()
