"""Command line entrypoint for AgentLoop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agentloop.agent import AgentLoop
from agentloop.chat_session import ChatSession
from agentloop.config import ConfigurationError, load_config
from agentloop.context_manager import ContextBudget, ContextManager
from agentloop.diagnostics import get_doctor_lines, get_tool_lines
from agentloop.memory import Memory
from agentloop.memory_store import SQLiteMemoryStore
from agentloop.models import ModelConfigurationError, build_model
from agentloop.session_store import SQLiteSessionStore
from agentloop.storage import (
    get_memory_db_path,
    get_session_db_path,
    get_trace_db_path,
)
from agentloop.tools import get_default_tool_runtime, get_default_tools
from agentloop.trace_store import SQLiteTraceStore

COMMANDS = {"run", "chat", "sessions", "memory", "tools", "doctor"}


def normalize_argv(argv: list[str] | None = None) -> list[str]:
    """Keep the legacy prompt-only CLI form as an alias for run."""

    normalized = list(sys.argv[1:] if argv is None else argv)
    if not normalized or normalized[0] in COMMANDS or normalized[0].startswith("-"):
        return normalized
    return ["run", *normalized]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run the AgentLoop harness.")
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
    chat_parser.add_argument(
        "--session-id",
        default=None,
        help="Resume an existing persisted session id.",
    )
    chat_parser.add_argument(
        "--session-db",
        type=Path,
        default=None,
        help="SQLite database path for persisted sessions.",
    )
    chat_parser.add_argument(
        "--memory-db",
        type=Path,
        default=None,
        help="SQLite database path for long-term memories.",
    )
    chat_parser.add_argument(
        "--trace-db",
        type=Path,
        default=None,
        help="SQLite database path for trace events.",
    )

    sessions_parser = subparsers.add_parser(
        "sessions",
        help="Create, list, resume, archive, or delete sessions.",
    )
    session_subparsers = sessions_parser.add_subparsers(
        dest="sessions_command",
        required=True,
    )
    sessions_create = session_subparsers.add_parser("create", help="Create a session.")
    _add_session_db_argument(sessions_create)
    sessions_create.add_argument(
        "--title",
        default=None,
        help="Optional session title.",
    )

    sessions_list = session_subparsers.add_parser("list", help="List sessions.")
    _add_session_db_argument(sessions_list)

    sessions_resume = session_subparsers.add_parser(
        "resume",
        help="Show persisted session history.",
    )
    _add_session_db_argument(sessions_resume)
    sessions_resume.add_argument("session_id")

    sessions_archive = session_subparsers.add_parser(
        "archive",
        help="Archive a session.",
    )
    _add_session_db_argument(sessions_archive)
    sessions_archive.add_argument("session_id")

    sessions_delete = session_subparsers.add_parser(
        "delete",
        help="Delete a session.",
    )
    _add_session_db_argument(sessions_delete)
    sessions_delete.add_argument("session_id")

    memory_parser = subparsers.add_parser(
        "memory",
        help="Manage user-confirmed long-term memories.",
    )
    memory_subparsers = memory_parser.add_subparsers(
        dest="memory_command",
        required=True,
    )
    memory_add = memory_subparsers.add_parser("add", help="Add confirmed memory.")
    _add_memory_db_argument(memory_add)
    memory_add.add_argument(
        "--namespace",
        required=True,
        choices=["user", "project", "decision"],
    )
    memory_add.add_argument("--content", required=True)
    memory_add.add_argument("--source-session-id", required=True)
    memory_add.add_argument("--source-run-id", required=True)
    memory_add.add_argument("--confirm", action="store_true")

    memory_list = memory_subparsers.add_parser("list", help="List memories.")
    _add_memory_db_argument(memory_list)
    memory_list.add_argument("--namespace", choices=["user", "project", "decision"])
    memory_list.add_argument(
        "--status",
        choices=["active", "disabled", "deleted"],
        default=None,
    )

    memory_get = memory_subparsers.add_parser("get", help="Show one memory.")
    _add_memory_db_argument(memory_get)
    memory_get.add_argument("memory_id")

    memory_search = memory_subparsers.add_parser("search", help="Search memories.")
    _add_memory_db_argument(memory_search)
    memory_search.add_argument("query")
    memory_search.add_argument("--namespace", choices=["user", "project", "decision"])
    memory_search.add_argument("--limit", type=int, default=20)

    memory_disable = memory_subparsers.add_parser("disable", help="Disable memory.")
    _add_memory_db_argument(memory_disable)
    memory_disable.add_argument("memory_id")

    memory_delete = memory_subparsers.add_parser("delete", help="Delete memory.")
    _add_memory_db_argument(memory_delete)
    memory_delete.add_argument("memory_id")

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

    if args.command == "sessions":
        _handle_sessions_command(args)
        return

    if args.command == "memory":
        _handle_memory_command(args)
        return

    if args.command == "chat":
        session_store = (
            SQLiteSessionStore(args.session_db or get_session_db_path())
            if args.session_id is not None
            else None
        )
        agent = _build_agent(
            max_steps=args.max_steps,
            model_name=args.model_name,
            session_id=args.session_id,
            session_store=session_store,
            memory_db=args.memory_db,
            trace_db=args.trace_db,
        )
        run_chat_session(
            ChatSession(
                agent=agent,
                session_store=session_store,
                session_id=args.session_id,
            )
        )
        return

    prompt = " ".join(args.prompt)
    agent = _build_agent()
    print(agent.run(prompt))


def run_chat_session(session: ChatSession) -> None:
    """Run an interactive chat session around an AgentLoop."""

    print(
        "system: chat started; use /exit, /clear, /history, /tools, /doctor, "
        "/context, /pending, /approve, /reject, /edit, /continue, /skip_tool, /stop"
    )
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
    session_id: str | None = None,
    session_store: SQLiteSessionStore | None = None,
    memory_db: Path | None = None,
    trace_db: Path | None = None,
) -> AgentLoop:
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

    try:
        model = build_model(config)
    except ModelConfigurationError as exc:
        raise SystemExit(str(exc)) from exc

    memory = Memory()
    if session_store is not None and session_id is not None:
        memory.set_messages(session_store.get_messages(session_id))

    memory_store = SQLiteMemoryStore(memory_db or get_memory_db_path())
    trace_store = SQLiteTraceStore(trace_db or get_trace_db_path())
    return AgentLoop(
        model=model,
        tools=get_default_tools(),
        memory=memory,
        max_steps=config.max_steps,
        trace_store=trace_store,
        session_id=session_id,
        memory_store=memory_store,
        context_manager=ContextManager(
            ContextBudget(
                max_tokens=config.context_max_tokens,
                reserved_output_tokens=config.context_reserved_output_tokens,
                summary_max_tokens=config.context_summary_max_tokens,
            )
        ),
    )


def _print_tools() -> None:
    """Print available tool names and descriptions."""

    for line in get_tool_lines(get_default_tool_runtime()):
        print(line)


def _print_doctor() -> None:
    """Print local deployment readiness checks."""

    for line in get_doctor_lines():
        print(line)


def _handle_sessions_command(args: argparse.Namespace) -> None:
    store = SQLiteSessionStore(args.db or get_session_db_path())
    if args.sessions_command == "create":
        session = store.create_session(title=args.title)
        print(f"session {session.session_id} created {_format_session_record(session)}")
        return
    if args.sessions_command == "list":
        sessions = store.list_sessions()
        if not sessions:
            print("sessions: empty")
            return
        for session in sessions:
            print(_format_session_record(session, include_id=True))
        return
    if args.sessions_command == "resume":
        session = store.get_session(args.session_id)
        print(_format_session_record(session, include_id=True))
        messages = store.get_messages(args.session_id)
        if not messages:
            print("history: empty")
            return
        for message in messages:
            for line in _format_message_lines(message):
                print(line)
        return
    if args.sessions_command == "archive":
        store.archive_session(args.session_id)
        print(f"session {args.session_id} archived")
        return
    if args.sessions_command == "delete":
        store.delete_session(args.session_id)
        print(f"session {args.session_id} deleted")
        return
    raise SystemExit(f"Unknown sessions command: {args.sessions_command}")


def _handle_memory_command(args: argparse.Namespace) -> None:
    store = SQLiteMemoryStore(args.db or get_memory_db_path())
    if args.memory_command == "add":
        if not args.confirm:
            raise SystemExit("memory add requires --confirm")
        memory = store.create_memory(
            namespace=args.namespace,
            content=args.content,
            source_session_id=args.source_session_id,
            source_run_id=args.source_run_id,
            confirmed_by_user=True,
        )
        print(f"memory {memory.memory_id} created {_format_memory_record(memory)}")
        return
    if args.memory_command == "list":
        memories = store.list_memories(
            namespace=args.namespace,
            status=args.status,
        )
        _print_memory_records(memories)
        return
    if args.memory_command == "get":
        print(_format_memory_record(store.get_memory(args.memory_id), include_id=True))
        return
    if args.memory_command == "search":
        namespaces = (args.namespace,) if args.namespace else None
        memories = store.search_memories(
            args.query,
            namespaces=namespaces,
            limit=args.limit,
        )
        _print_memory_records(memories)
        return
    if args.memory_command == "disable":
        store.disable_memory(args.memory_id)
        print(f"memory {args.memory_id} disabled")
        return
    if args.memory_command == "delete":
        store.delete_memory(args.memory_id)
        print(f"memory {args.memory_id} deleted")
        return
    raise SystemExit(f"Unknown memory command: {args.memory_command}")


def _print_memory_records(memories) -> None:
    if not memories:
        print("memory: empty")
        return
    for memory in memories:
        print(_format_memory_record(memory, include_id=True))


def _format_session_record(session, include_id: bool = False) -> str:
    prefix = f"session {session.session_id} " if include_id else ""
    title = session.title if session.title is not None else ""
    return (
        f"{prefix}title={title!r} "
        f"status={session.status.value} "
        f"messages={session.message_count}"
    )


def _format_memory_record(memory, include_id: bool = False) -> str:
    prefix = f"{memory.memory_id} " if include_id else ""
    return (
        f"{prefix}namespace={memory.namespace.value} "
        f"status={memory.status.value} "
        f"source_session_id={memory.source_session_id} "
        f"source_run_id={memory.source_run_id} "
        f"content={memory.content}"
    )


def _format_message_lines(message: BaseMessage) -> list[str]:
    if isinstance(message, HumanMessage):
        return [f"user: {message.content}"]
    if isinstance(message, AIMessage):
        lines = []
        for tool_call in message.tool_calls:
            lines.append(f"tool-call: {tool_call['name']} {tool_call.get('args', {})}")
        if str(message.content):
            lines.append(f"assistant: {message.content}")
        return lines
    if isinstance(message, ToolMessage):
        tool_name = message.additional_kwargs.get("tool_name", "unknown")
        return [f"tool-result: {tool_name} {message.content}"]
    return [f"{getattr(message, 'type', 'message')}: {message.content}"]


def _add_session_db_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite session database path.",
    )


def _add_memory_db_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite memory database path.",
    )


if __name__ == "__main__":
    main()
