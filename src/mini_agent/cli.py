"""Command line entrypoint for the mini agent."""

from __future__ import annotations

import argparse

from mini_agent.agent import MiniAgent
from mini_agent.config import load_config
from mini_agent.memory import Memory
from mini_agent.models import OpenAIModelClient
from mini_agent.tools import build_default_registry


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
        model_client=OpenAIModelClient(
            api_key=config.openai_api_key,
            model_name=config.model_name,
        ),
        tool_registry=build_default_registry(),
        memory=Memory(),
        max_steps=config.max_steps,
    )
    print(agent.run(args.prompt))


if __name__ == "__main__":
    main()
