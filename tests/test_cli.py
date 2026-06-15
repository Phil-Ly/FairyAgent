import pytest

from agentloop.cli import build_parser, main, normalize_argv
from agentloop.models import ModelConfigurationError


def test_normalize_argv_preserves_subcommands() -> None:
    assert normalize_argv(["run", "What is 2 + 2?"]) == ["run", "What is 2 + 2?"]
    assert normalize_argv(["chat"]) == ["chat"]


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
    assert "calculator" in output
    assert "echo" in output


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

    with pytest.raises(SystemExit, match="MODEL_API_KEY is required"):
        main(["run", "hello"])


def test_chat_command_reports_missing_model_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="MODEL_API_KEY is required"):
        main(["chat"])


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
    assert "test-key" not in output
    assert "tools: calculator, echo" in output


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
    assert "demo" not in help_text
