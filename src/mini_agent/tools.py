"""LangChain tools exposed to the agent."""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable

from langchain_core.tools import BaseTool, tool


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


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression."""

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


@tool
def echo(text: str) -> str:
    """Return the provided text unchanged."""

    return text


def get_default_tools() -> list[BaseTool]:
    """Return the built-in LangChain tools."""

    return [calculator, echo]
