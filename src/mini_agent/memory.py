"""Short-term in-memory conversation state."""

from __future__ import annotations

import json


class Memory:
    """Stores messages for a single agent run."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def add_user_message(self, content: str) -> None:
        """Append a user message."""

        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Append a final assistant message."""

        self.messages.append({"role": "assistant", "content": content})

    def add_assistant_tool_calls(self, tool_calls: list[dict]) -> None:
        """Append assistant tool calls in OpenAI-compatible message format."""

        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["arguments"]),
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
        )

    def add_tool_message(self, tool_call_id: str, content: str) -> None:
        """Append a tool result message."""

        self.messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )

    def get_messages(self) -> list[dict]:
        """Return a shallow copy of the current message list."""

        return list(self.messages)

    def clear(self) -> None:
        """Remove all messages."""

        self.messages.clear()
