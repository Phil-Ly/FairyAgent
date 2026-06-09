from mini_agent.agent import MiniAgent
from mini_agent.memory import Memory
from mini_agent.tools import build_default_registry


class FakeModelClient:
    """Model client that asks for one calculation, then returns a final answer."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {
                "type": "tool_call",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "calculator",
                        "arguments": {"expression": "12 * (3 + 4)"},
                    }
                ],
            }
        return {"type": "final", "content": "The answer is 84.", "tool_calls": []}


class NeverFinalModelClient:
    """Model client that never produces a final answer."""

    def generate(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> dict:
        return {
            "type": "tool_call",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_repeat",
                    "name": "echo",
                    "arguments": {"text": "keep going"},
                }
            ],
        }


def test_agent_completes_tool_call_loop() -> None:
    memory = Memory()
    registry = build_default_registry()
    model_client = FakeModelClient()
    agent = MiniAgent(
        model_client=model_client,
        tool_registry=registry,
        memory=memory,
        max_steps=4,
    )

    result = agent.run("What is 12 * (3 + 4)?")

    assert result == "The answer is 84."
    assert model_client.calls == 2
    assert memory.get_messages() == [
        {"role": "user", "content": "What is 12 * (3 + 4)?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "12 * (3 + 4)"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "84"},
        {"role": "assistant", "content": "The answer is 84."},
    ]


def test_agent_stops_at_max_steps() -> None:
    agent = MiniAgent(
        model_client=NeverFinalModelClient(),
        tool_registry=build_default_registry(),
        memory=Memory(),
        max_steps=2,
    )

    result = agent.run("loop forever")

    assert result == "Agent stopped because it reached the maximum number of steps."
