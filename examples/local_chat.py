"""Run a local chat session without provider credentials."""

from __future__ import annotations

from local_model import ExampleCalculatorModel

from mini_agent.agent import MiniAgent
from mini_agent.chat_session import ChatSession
from mini_agent.cli import run_chat_session
from mini_agent.memory import Memory
from mini_agent.tools import get_default_tools


def main() -> None:
    """Run a local interactive chat example."""

    agent = MiniAgent(
        model=ExampleCalculatorModel(),
        tools=get_default_tools(),
        memory=Memory(),
        max_steps=4,
    )
    run_chat_session(ChatSession(agent=agent))


if __name__ == "__main__":
    main()
