"""Command line entrypoint for the mini agent."""

from __future__ import annotations

import argparse
import sys

from mini_agent.agent import MiniAgent
from mini_agent.chat_session import ChatSession
from mini_agent.config import ConfigurationError, load_config
from mini_agent.diagnostics import get_doctor_lines, get_tool_lines
from mini_agent.memory import Memory
from mini_agent.models import ModelConfigurationError, build_model
from mini_agent.tools import get_default_tools

COMMANDS = {"run", "chat", "tools", "doctor"}


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

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat.")
    chat_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the maximum number of LLM calls per turn.",
    )
    chat_parser.add_argument(
        "--model-name",
        default=None,
        help="Override MODEL_NAME for provider-backed chat.",
    )

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

    if args.command == "chat":
        agent = _build_agent(
            max_steps=args.max_steps,
            model_name=args.model_name,
        )
        run_chat_session(ChatSession(agent=agent))
        return

    prompt = " ".join(args.prompt)
    agent = _build_agent()
    print(agent.run(prompt))


def run_chat_session(session: ChatSession) -> None:
    """Run an interactive chat session around a MiniAgent."""

    print("system: chat started; use /exit, /clear, /history, /tools, /doctor")
    while not session.should_exit:
        try:
            user_input = input("user> ")
        except (EOFError, KeyboardInterrupt):
            print("system: goodbye")
            return

        for line in session.handle_line(user_input):
            print(line)


def _build_agent(
    max_steps: int | None = None,
    model_name: str | None = None,
) -> MiniAgent:
    """Build an agent from configured provider settings."""

    try:
        config = load_config()
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc

    updates = {}
    if max_steps is not None:
        updates["max_steps"] = max_steps
    if model_name is not None:
        updates["model_name"] = model_name
    if updates:
        config = config.model_copy(update=updates)

    if not config.resolved_model_api_key:
        raise SystemExit("MODEL_API_KEY is required. Copy .env.example to .env.")

    try:
        model = build_model(config)
    except ModelConfigurationError as exc:
        raise SystemExit(str(exc)) from exc

    return MiniAgent(
        model=model,
        tools=get_default_tools(),
        memory=Memory(),
        max_steps=config.max_steps,
    )


def _print_tools() -> None:
    """Print available tool names and descriptions."""

    for line in get_tool_lines(get_default_tools()):
        print(line)


def _print_doctor() -> None:
    """Print local deployment readiness checks."""

    for line in get_doctor_lines():
        print(line)


if __name__ == "__main__":
    main()
