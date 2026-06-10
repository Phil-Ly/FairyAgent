import pytest

from mini_agent.cli import build_parser, main, normalize_argv
from mini_agent.models import ModelConfigurationError


def test_normalize_argv_preserves_subcommands() -> None:
    assert normalize_argv(["demo", "What is 2 + 2?"]) == ["demo", "What is 2 + 2?"]


def test_normalize_argv_treats_plain_prompt_as_run_command() -> None:
    assert normalize_argv(["What is 2 + 2?"]) == ["run", "What is 2 + 2?"]


def test_tools_command_lists_default_tools(capsys: pytest.CaptureFixture[str]) -> None:
    main(["tools"])

    output = capsys.readouterr().out
    assert "calculator" in output
    assert "echo" in output


def test_demo_command_runs_without_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    main(["demo", "What is 2 + 3 * 4?"])

    output = capsys.readouterr().out
    assert "The answer is 14." in output


def test_run_command_reports_missing_model_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def raise_missing_dependency(config):
        raise ModelConfigurationError(
            "langchain-openai is required for the run command."
        )

    monkeypatch.setattr("mini_agent.cli.build_model", raise_missing_dependency)

    with pytest.raises(SystemExit, match="langchain-openai is required"):
        main(["run", "hello"])


def test_doctor_command_prints_runtime_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["doctor"])

    output = capsys.readouterr().out
    assert "Mini Agent doctor" in output
    assert "OPENAI_API_KEY:" in output
    assert "tools: calculator, echo" in output


def test_parser_includes_product_commands() -> None:
    parser = build_parser()

    help_text = parser.format_help()

    assert "run" in help_text
    assert "demo" in help_text
    assert "tools" in help_text
    assert "doctor" in help_text
