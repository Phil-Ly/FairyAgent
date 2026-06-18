"""Run a local chat session without provider credentials."""

from __future__ import annotations

from local_model import ExampleCalculatorModel

from agentloop.agent import AgentLoop
from agentloop.chat_session import ChatSession
from agentloop.cli import run_chat_session
from agentloop.memory import Memory
from agentloop.tools import get_default_tools


def main() -> None:
    """Run a local interactive chat example."""

    agent = AgentLoop(
        model=ExampleCalculatorModel(),
        tools=get_default_tools(),
        memory=Memory(),
        max_steps=4,
    )
    run_chat_session(ChatSession(agent=agent))


if __name__ == "__main__":
    main()
