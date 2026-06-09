"""Minimal agent loop."""

from __future__ import annotations

from mini_agent.memory import Memory
from mini_agent.models import ModelClient
from mini_agent.tools import ToolRegistry


MAX_STEPS_MESSAGE = "Agent stopped because it reached the maximum number of steps."


class MiniAgent:
    """Runs a model/tool loop until a final answer is produced."""

    def __init__(
        self,
        model_client: ModelClient,
        tool_registry: ToolRegistry,
        memory: Memory,
        max_steps: int,
    ) -> None:
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.memory = memory
        self.max_steps = max_steps

    def run(self, user_input: str) -> str:
        """Run the agent for a single user input."""

        self.memory.add_user_message(user_input)

        for _ in range(self.max_steps):
            response = self.model_client.generate(
                self.memory.get_messages(),
                self.tool_registry.get_tool_schemas(),
            )

            if response["type"] == "final":
                content = response.get("content") or ""
                self.memory.add_assistant_message(content)
                return content

            if response["type"] == "tool_call":
                tool_calls = response.get("tool_calls", [])
                self.memory.add_assistant_tool_calls(tool_calls)
                for tool_call in tool_calls:
                    result = self.tool_registry.call_tool(
                        tool_call["name"],
                        tool_call["arguments"],
                    )
                    self.memory.add_tool_message(tool_call["id"], result)
                continue

            raise ValueError(f"Unknown model response type: {response['type']}")

        return MAX_STEPS_MESSAGE
