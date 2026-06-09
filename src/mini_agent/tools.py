"""Minimal local tool system."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any, Callable


ToolFunction = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class Tool:
    """A callable tool exposed to the model."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    function: ToolFunction


class ToolRegistry:
    """Registry for tools available to the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by name."""

        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get_tool_schemas(self) -> list[dict]:
        """Return OpenAI-compatible function tool schemas."""

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                },
            }
            for tool in self._tools.values()
        ]

    def call_tool(self, name: str, arguments: dict) -> str:
        """Call a registered tool and return its string result."""

        try:
            tool = self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc
        return tool.function(arguments)


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculator(arguments: dict[str, Any]) -> str:
    """Safely evaluate a simple arithmetic expression."""

    expression = arguments.get("expression")
    if not isinstance(expression, str):
        raise ValueError("calculator requires an expression string.")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate_expression(tree.body)
    except (SyntaxError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("Unsupported expression.") from exc

    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def _evaluate_expression(node: ast.AST) -> int | float:
    """Evaluate only whitelisted arithmetic AST nodes."""

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Unsupported expression.")
        return node.value

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)
        if operator_type not in _BINARY_OPERATORS:
            raise ValueError("Unsupported expression.")
        left = _evaluate_expression(node.left)
        right = _evaluate_expression(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Unsupported expression.")
        return _BINARY_OPERATORS[operator_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)
        if operator_type not in _UNARY_OPERATORS:
            raise ValueError("Unsupported expression.")
        return _UNARY_OPERATORS[operator_type](_evaluate_expression(node.operand))

    raise ValueError("Unsupported expression.")


def echo(arguments: dict[str, Any]) -> str:
    """Return the provided text unchanged."""

    text = arguments.get("text")
    if not isinstance(text, str):
        raise ValueError("echo requires a text string.")
    return text


def build_default_registry() -> ToolRegistry:
    """Create a registry with the built-in calculator and echo tools."""

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="Safely evaluate a simple arithmetic expression.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A simple arithmetic expression, e.g. '2 + 3 * 4'."
                        ),
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            function=calculator,
        )
    )
    registry.register(
        Tool(
            name="echo",
            description="Return the provided text unchanged.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to return unchanged.",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            function=echo,
        )
    )
    return registry
