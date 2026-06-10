"""Command line entrypoint for the mini agent."""

from __future__ import annotations

import argparse

from mini_agent.agent import MiniAgent
from mini_agent.config import load_config
from mini_agent.memory import Memory
from mini_agent.models import build_model
from mini_agent.tools import get_default_tools


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run the mini agent harness.")
    parser.add_argument("prompt", help="User input to send to the agent.")
    return parser


def main() -> None:
    """Run the command line agent."""

    args = build_parser().parse_args()
    config = load_config()
    if not config.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required. Copy .env.example to .env.")

    agent = MiniAgent(
        model=build_model(config),
        tools=get_default_tools(),
        memory=Memory(),
        max_steps=config.max_steps,
    )
    print(agent.run(args.prompt))


if __name__ == "__main__":
    main()
