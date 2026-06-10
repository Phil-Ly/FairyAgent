"""Deterministic demo model for local smoke tests."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool


class DemoChatModel:
    """Small fake chat model that demonstrates the LangGraph tool loop."""

    def __init__(self) -> None:
        self.tools: list[BaseTool] = []

    def bind_tools(self, tools: list[BaseTool]) -> "DemoChatModel":
        """Store tools and return a model-like object."""

        self.tools = tools
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Return a deterministic AIMessage for demo use."""

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
                            "id": "demo_calculator_1",
                            "name": "calculator",
                            "args": {"expression": expression},
                        }
                    ],
                )

        return AIMessage(
            content="Demo mode can calculate simple arithmetic expressions."
        )


def _extract_arithmetic_expression(text: str) -> str | None:
    """Extract a simple arithmetic expression from a prompt."""

    for match in re.finditer(r"[0-9][0-9\s+\-*/%().]*", text):
        expression = match.group(0).strip()
        if _looks_like_expression(expression):
            return expression
    return None


def _looks_like_expression(expression: str) -> bool:
    """Return True when text looks like arithmetic rather than a single number."""

    has_digit = any(character.isdigit() for character in expression)
    has_operator = any(character in "+-*/%" for character in expression)
    return has_digit and has_operator
