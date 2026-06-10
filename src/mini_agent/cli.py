"""Command line entrypoint for the mini agent."""

from __future__ import annotations

import argparse
import importlib.util
import sys

from mini_agent.agent import MiniAgent
from mini_agent.config import load_config
from mini_agent.demo import DemoChatModel
from mini_agent.memory import Memory
from mini_agent.models import ModelConfigurationError, build_model
from mini_agent.tools import get_default_tools


COMMANDS = {"run", "demo", "tools", "doctor"}


def normalize_argv(argv: list[str] | None = None) -> list[str]:
    """Keep the legacy prompt-only CLI form as an alias for run."""

    normalized = list(sys.argv[1:] if argv is None else argv)
    if not normalized or normalized[0] in COMMANDS or normalized[0].startswith("-"):
        return normalized
    return ["run", *normalized]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run the mini agent harness.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run with the configured model.")
    run_parser.add_argument("prompt", nargs="+", help="User input to send.")

    demo_parser = subparsers.add_parser("demo", help="Run a local deterministic demo.")
    demo_parser.add_argument("prompt", nargs="+", help="User input to send.")

    subparsers.add_parser("tools", help="List available tools.")
    subparsers.add_parser("doctor", help="Check local runtime configuration.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the command line agent."""

    args = build_parser().parse_args(normalize_argv(argv))
    if args.command == "tools":
        _print_tools()
        return

    if args.command == "doctor":
        _print_doctor()
        return

    prompt = " ".join(args.prompt)
    if args.command == "demo":
        agent = MiniAgent(
            model=DemoChatModel(),
            tools=get_default_tools(),
            memory=Memory(),
            max_steps=4,
        )
        print(agent.run(prompt))
        return

    config = load_config()
    if not config.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required. Copy .env.example to .env.")

    try:
        model = build_model(config)
    except ModelConfigurationError as exc:
        raise SystemExit(str(exc)) from exc

    agent = MiniAgent(
        model=model,
        tools=get_default_tools(),
        memory=Memory(),
        max_steps=config.max_steps,
    )
    print(agent.run(prompt))


def _print_tools() -> None:
    """Print available tool names and descriptions."""

    for agent_tool in get_default_tools():
        print(f"{agent_tool.name}: {agent_tool.description}")


def _print_doctor() -> None:
    """Print local deployment readiness checks."""

    config = load_config()
    has_openai_key = bool(config.openai_api_key)
    has_openai_package = importlib.util.find_spec("langchain_openai") is not None
    print("Mini Agent doctor")
    print(f"MODEL_NAME: {config.model_name}")
    print(f"MAX_STEPS: {config.max_steps}")
    print(f"OPENAI_API_KEY: {'configured' if has_openai_key else 'missing'}")
    print(
        "langchain-openai: "
        f"{'installed' if has_openai_package else 'missing'}"
    )
    print("tools: " + ", ".join(tool.name for tool in get_default_tools()))


if __name__ == "__main__":
    main()
