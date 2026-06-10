import pytest

from mini_agent.tools import calculator, echo, get_default_tools


def test_calculator_evaluates_simple_expression() -> None:
    result = calculator.invoke({"expression": "2 + 3 * 4"})

    assert result == "14"


def test_calculator_rejects_dangerous_expression() -> None:
    with pytest.raises(ValueError, match="Unsupported expression"):
        calculator.invoke({"expression": "__import__('os').system('rm -rf /')"})


def test_echo_returns_text() -> None:
    result = echo.invoke({"text": "same text"})

    assert result == "same text"


def test_get_default_tools_returns_calculator_and_echo() -> None:
    tools = get_default_tools()

    assert [tool.name for tool in tools] == ["calculator", "echo"]
