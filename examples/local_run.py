"""Run the agent loop locally without provider credentials."""

from __future__ import annotations

import sys

from local_model import ExampleCalculatorModel

from agentloop.agent import AgentLoop
from agentloop.memory import Memory
from agentloop.tools import get_default_tools


def main(argv: list[str] | None = None) -> None:
    """Run one local example turn."""

    prompt_parts = sys.argv[1:] if argv is None else argv
    if not prompt_parts:
        raise SystemExit('Usage: python examples/local_run.py "What is 2 + 2?"')

    agent = AgentLoop(
        model=ExampleCalculatorModel(),
        tools=get_default_tools(),
        memory=Memory(),
        max_steps=4,
    )
    print(agent.run(" ".join(prompt_parts)))


if __name__ == "__main__":
    main()
