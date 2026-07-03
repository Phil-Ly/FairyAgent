"""Line-oriented chat session logic."""

from __future__ import annotations

import json

from langchain_core.messages import BaseMessage

from agentloop.agent import AgentLoop
from agentloop.diagnostics import get_doctor_lines, get_tool_lines
from agentloop.intervention import InterventionAction, InterventionWorkflow
from agentloop.session_store import SQLiteSessionStore

ACTION_COMMANDS = {
    "/approve": InterventionAction.APPROVE,
    "/reject": InterventionAction.REJECT,
    "/continue": InterventionAction.CONTINUE,
    "/skip_tool": InterventionAction.SKIP_TOOL,
    "/stop": InterventionAction.STOP,
}


class ChatSession:
    """Handles chat control commands and agent turns for one process."""

    def __init__(
        self,
        agent: AgentLoop,
        session_store: SQLiteSessionStore | None = None,
        session_id: str | None = None,
    ) -> None:
        self.agent = agent
        self.session_store = session_store
        self.session_id = session_id
        self.should_exit = False
        self.intervention_workflow = InterventionWorkflow()

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
            return get_tool_lines(self.agent.tool_runtime)
        if user_input == "/doctor":
            return get_doctor_lines()
        if user_input == "/context":
            return self._context_lines()
        if user_input == "/pending":
            return self._pending_lines()
        if user_input in ACTION_COMMANDS:
            return self._handle_intervention_action(ACTION_COMMANDS[user_input])
        if user_input.startswith("/edit"):
            return self._handle_edit_command(user_input)

        message_count = len(self.agent.memory.get_messages())
        self.agent.run(user_input)
        new_messages = self.agent.memory.get_messages()[message_count:]
        self._persist_messages(new_messages)
        return format_turn_messages(new_messages)

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

    def _pending_lines(self) -> list[str]:
        """Return the currently pending intervention, if any."""

        pending = self.agent.get_pending_intervention()
        if pending is None:
            return ["pending: none"]
        return self.intervention_workflow.format_pending_request(pending.request)

    def _context_lines(self) -> list[str]:
        """Return the most recent model context composition."""

        report = self.agent.get_context_report()
        if report is None:
            return ["context: none"]
        return report.to_display_lines()

    def _handle_intervention_action(
        self,
        action: InterventionAction,
        edited_tool_args: dict | None = None,
    ) -> list[str]:
        """Resolve a pending intervention from a slash command."""

        pending = self.agent.get_pending_intervention()
        if pending is None:
            return ["pending: none"]
        base_message_count = len(pending.messages)
        decision = self.intervention_workflow.create_decision(
            pending.request,
            action=action,
            edited_tool_args=edited_tool_args,
        )
        self.agent.resolve_intervention(decision)
        return format_turn_messages(
            self.agent.memory.get_messages()[base_message_count:],
        )

    def _handle_edit_command(self, user_input: str) -> list[str]:
        """Resolve a pending intervention with edited tool arguments."""

        _, _, raw_payload = user_input.partition(" ")
        if not raw_payload.strip():
            return ["system: /edit requires a JSON object"]
        try:
            edited_tool_args = json.loads(raw_payload)
        except json.JSONDecodeError:
            return ["system: /edit requires a JSON object"]
        if not isinstance(edited_tool_args, dict):
            return ["system: /edit requires a JSON object"]
        return self._handle_intervention_action(
            InterventionAction.EDIT,
            edited_tool_args=edited_tool_args,
        )

    def _persist_messages(self, messages: list[BaseMessage]) -> None:
        if self.session_store is None or self.session_id is None:
            return
        for message in messages:
            self.session_store.append_message(self.session_id, message)


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
    content = str(message.text)
    if content:
        lines.append(f"assistant: {content}")
    return lines


def format_tool_message(message: BaseMessage) -> str:
    """Return one display line for a tool result."""

    tool_name = message.additional_kwargs.get("tool_name", "unknown")
    return f"tool-result: {tool_name} {message.content}"
