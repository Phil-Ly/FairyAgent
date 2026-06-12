"""Test doubles for deterministic agent tests."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool


class DeterministicCalculatorModel:
    """Fake chat model that calls calculator for simple arithmetic prompts."""

    def bind_tools(self, tools: list[BaseTool]) -> DeterministicCalculatorModel:
        """Return a model-like object with a LangChain-compatible API."""

        self.tools = tools
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Return deterministic messages for tests."""

        last_message = messages[-1]
        if isinstance(last_message, ToolMessage):
            return AIMessage(content=f"The answer is {last_message.content}.")

        if isinstance(last_message, HumanMessage):
            expression = _extract_arithmetic_expression(str(last_message.content))
            if expression is not None:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "test_calculator_1",
                            "name": "calculator",
                            "args": {"expression": expression},
                        }
                    ],
                )

        return AIMessage(content="No calculation requested.")


def _extract_arithmetic_expression(text: str) -> str | None:
    """Extract a simple arithmetic expression from test input."""

    for match in re.finditer(r"[0-9][0-9\s+\-*/%().]*", text):
        expression = match.group(0).strip()
        if _looks_like_expression(expression):
            return expression
    return None


def _looks_like_expression(expression: str) -> bool:
    """Return True when text looks like arithmetic."""

    has_digit = any(character.isdigit() for character in expression)
    has_operator = any(character in "+-*/%" for character in expression)
    return has_digit and has_operator
