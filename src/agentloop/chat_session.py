"""Line-oriented chat session logic."""

from __future__ import annotations

from langchain_core.messages import BaseMessage

from agentloop.agent import AgentLoop
from agentloop.diagnostics import get_doctor_lines, get_tool_lines


class ChatSession:
    """Handles chat control commands and agent turns for one process."""

    def __init__(self, agent: AgentLoop) -> None:
        self.agent = agent
        self.should_exit = False

    def handle_line(self, line: str) -> list[str]:
        """Handle one user input line and return display lines."""

        user_input = line.strip()
        if not user_input:
            return []
        if user_input == "/exit":
            self.should_exit = True
            return ["system: goodbye"]
        if user_input == "/clear":
            self.agent.memory.clear()
            return ["system: memory cleared"]
        if user_input == "/history":
            return self._history_lines()
        if user_input == "/tools":
            return get_tool_lines(self.agent.tools)
        if user_input == "/doctor":
            return get_doctor_lines()

        message_count = len(self.agent.memory.get_messages())
        self.agent.run(user_input)
        return format_turn_messages(
            self.agent.memory.get_messages()[message_count:],
        )

    def _history_lines(self) -> list[str]:
        """Return the current in-memory chat history."""

        messages = self.agent.memory.get_messages()
        if not messages:
            return ["history: empty"]

        lines: list[str] = []
        for message in messages:
            role = getattr(message, "type", "message")
            if role == "human":
                lines.append(f"user: {message.content}")
            elif role == "ai":
                lines.extend(format_ai_message(message))
            elif role == "tool":
                lines.append(format_tool_message(message))
        return lines


def format_turn_messages(messages: list[BaseMessage]) -> list[str]:
    """Return assistant and tool events produced by one chat turn."""

    lines: list[str] = []
    for message in messages:
        role = getattr(message, "type", "message")
        if role == "ai":
            lines.extend(format_ai_message(message))
        elif role == "tool":
            lines.append(format_tool_message(message))
    return lines


def format_ai_message(message: BaseMessage) -> list[str]:
    """Return display lines for AI content and tool calls."""

    lines: list[str] = []
    for tool_call in getattr(message, "tool_calls", []):
        lines.append(f"tool-call: {tool_call['name']} {tool_call.get('args', {})}")
    content = str(message.content)
    if content:
        lines.append(f"assistant: {content}")
    return lines


def format_tool_message(message: BaseMessage) -> str:
    """Return one display line for a tool result."""

    tool_name = message.additional_kwargs.get("tool_name", "unknown")
    return f"tool-result: {tool_name} {message.content}"
