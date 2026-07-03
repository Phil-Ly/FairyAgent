import pytest
from langchain_core.messages import HumanMessage

from agentloop.cli import build_parser, main, normalize_argv
from agentloop.memory_store import MemoryStatus, SQLiteMemoryStore
from agentloop.models import ModelConfigurationError
from agentloop.session_store import SQLiteSessionStore


def test_normalize_argv_preserves_subcommands() -> None:
    assert normalize_argv(["run", "What is 2 + 2?"]) == ["run", "What is 2 + 2?"]
    assert normalize_argv(["chat"]) == ["chat"]
    assert normalize_argv(["sessions", "list"]) == ["sessions", "list"]
    assert normalize_argv(["memory", "list"]) == ["memory", "list"]


def test_normalize_argv_treats_plain_prompt_as_run_command() -> None:
    assert normalize_argv(["What is 2 + 2?"]) == ["run", "What is 2 + 2?"]
    assert normalize_argv(["demo", "What is 2 + 2?"]) == [
        "run",
        "demo",
        "What is 2 + 2?",
    ]


def test_tools_command_lists_default_tools(capsys: pytest.CaptureFixture[str]) -> None:
    main(["tools"])

    output = capsys.readouterr().out
    assert "calculator: risk=low read_only=true requires_confirmation=false" in output
    assert "read_file: risk=low read_only=true requires_confirmation=false" in output


def test_run_command_reports_missing_model_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def raise_missing_dependency(config):
        raise ModelConfigurationError(
            "langchain-openai is required for the run command."
        )

    monkeypatch.setattr("agentloop.cli.build_model", raise_missing_dependency)

    with pytest.raises(SystemExit, match="langchain-openai is required"):
        main(["run", "hello"])


def test_run_command_reports_missing_model_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(
        SystemExit,
        match="MODEL_API_KEY or OPENAI_API_KEY is required",
    ):
        main(["run", "hello"])


def test_chat_command_reports_missing_model_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(
        SystemExit,
        match="MODEL_API_KEY or OPENAI_API_KEY is required",
    ):
        main(["chat"])


def test_run_command_accepts_provider_specific_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("MODEL_NAME", "claude-sonnet-4-6")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    def confirm_provider_key_reaches_model_layer(config):
        assert config.anthropic_api_key == "anthropic-key"
        raise ModelConfigurationError("model-layer-sentinel")

    monkeypatch.setattr(
        "agentloop.cli.build_model",
        confirm_provider_key_reaches_model_layer,
    )

    with pytest.raises(SystemExit, match="model-layer-sentinel"):
        main(["run", "hello"])


def test_doctor_command_prints_runtime_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["doctor"])

    output = capsys.readouterr().out
    assert "AgentLoop doctor" in output
    assert "Python:" in output
    assert "uv:" in output
    assert "langchain:" in output
    assert "langgraph:" in output
    assert "OPENAI_API_KEY:" in output
    assert "MODEL_PROVIDER:" in output
    assert "MODEL_BASE_URL:" in output
    assert "MODEL_API_KEY:" in output
    assert "session_store:" in output
    assert "memory_store:" in output
    assert "trace_store:" in output
    assert "safety_policy:" in output
    assert "test-key" not in output
    assert "tools: calculator, echo" in output


def test_doctor_reports_active_native_provider_dependency_and_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("MODEL_NAME", "claude-sonnet-4-6")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    main(["doctor"])

    output = capsys.readouterr().out
    assert "langchain-anthropic:" in output
    assert "ANTHROPIC_API_KEY: configured" in output
    assert "anthropic-key" not in output


def test_run_command_reports_configuration_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_STEPS", "abc")

    with pytest.raises(SystemExit, match="MAX_STEPS must be an integer"):
        main(["run", "hello"])


def test_doctor_reports_configuration_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("MAX_STEPS", "abc")

    main(["doctor"])

    output = capsys.readouterr().out
    assert "configuration: invalid" in output
    assert "MAX_STEPS must be an integer" in output


def test_parser_includes_product_commands() -> None:
    parser = build_parser()

    help_text = parser.format_help()

    assert "run" in help_text
    assert "chat" in help_text
    assert "tools" in help_text
    assert "doctor" in help_text
    assert "sessions" in help_text
    assert "memory" in help_text
    assert "demo" not in help_text


def test_sessions_command_creates_lists_resumes_archives_and_deletes_session(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "sessions.sqlite3"

    main(["sessions", "create", "--db", str(db_path), "--title", "MVP work"])
    created_output = capsys.readouterr().out
    session_id = created_output.split()[1]
    store = SQLiteSessionStore(db_path)
    run = store.create_run(session_id, user_input="hello")
    store.append_message(session_id, HumanMessage(content="hello"), run_id=run.run_id)

    main(["sessions", "list", "--db", str(db_path)])
    list_output = capsys.readouterr().out
    main(["sessions", "resume", session_id, "--db", str(db_path)])
    resume_output = capsys.readouterr().out
    main(["sessions", "archive", session_id, "--db", str(db_path)])
    archive_output = capsys.readouterr().out
    main(["sessions", "delete", session_id, "--db", str(db_path)])
    delete_output = capsys.readouterr().out

    assert f"session {session_id} created" in created_output
    assert f"{session_id} title='MVP work' status=active messages=1" in list_output
    assert (
        f"session {session_id} title='MVP work' status=active messages=1"
        in resume_output
    )
    assert "user: hello" in resume_output
    assert f"session {session_id} archived" in archive_output
    assert f"session {session_id} deleted" in delete_output
    with pytest.raises(KeyError):
        store.get_session(session_id)


def test_memory_command_adds_lists_searches_disables_and_deletes_memory(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "memory.sqlite3"

    main(
        [
            "memory",
            "add",
            "--db",
            str(db_path),
            "--namespace",
            "project",
            "--content",
            "Project uses SQLite for MVP persistence.",
            "--source-session-id",
            "session_1",
            "--source-run-id",
            "run_1",
            "--confirm",
        ]
    )
    add_output = capsys.readouterr().out
    memory_id = add_output.split()[1]

    main(["memory", "list", "--db", str(db_path)])
    list_output = capsys.readouterr().out
    main(["memory", "search", "SQLite", "--db", str(db_path)])
    search_output = capsys.readouterr().out
    main(["memory", "disable", memory_id, "--db", str(db_path)])
    disable_output = capsys.readouterr().out
    main(["memory", "delete", memory_id, "--db", str(db_path)])
    delete_output = capsys.readouterr().out

    store = SQLiteMemoryStore(db_path)
    deleted = store.get_memory(memory_id)
    assert f"memory {memory_id} created namespace=project status=active" in add_output
    assert f"{memory_id} namespace=project status=active" in list_output
    assert "Project uses SQLite" in search_output
    assert f"memory {memory_id} disabled" in disable_output
    assert f"memory {memory_id} deleted" in delete_output
    assert deleted.status is MemoryStatus.DELETED


def test_memory_add_requires_explicit_confirmation(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite3"

    with pytest.raises(SystemExit, match="memory add requires --confirm"):
        main(
            [
                "memory",
                "add",
                "--db",
                str(db_path),
                "--namespace",
                "user",
                "--content",
                "User prefers concise answers.",
                "--source-session-id",
                "session_1",
                "--source-run-id",
                "run_1",
            ]
        )
