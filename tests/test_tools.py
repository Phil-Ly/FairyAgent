import pytest

from mini_agent.tools import Tool, ToolRegistry, build_default_registry


def test_calculator_evaluates_simple_expression() -> None:
    registry = build_default_registry()

    result = registry.call_tool("calculator", {"expression": "2 + 3 * 4"})

    assert result == "14"


def test_calculator_rejects_dangerous_expression() -> None:
    registry = build_default_registry()

    with pytest.raises(ValueError, match="Unsupported expression"):
        registry.call_tool(
            "calculator", {"expression": "__import__('os').system('rm -rf /')"}
        )


def test_echo_returns_text() -> None:
    registry = build_default_registry()

    result = registry.call_tool("echo", {"text": "same text"})

    assert result == "same text"


def test_tool_registry_registers_and_calls_tool() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="upper",
            description="Uppercase text.",
            parameters_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            function=lambda arguments: arguments["text"].upper(),
        )
    )

    result = registry.call_tool("upper", {"text": "hello"})
    schemas = registry.get_tool_schemas()

    assert result == "HELLO"
    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "upper",
                "description": "Uppercase text.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        }
    ]
