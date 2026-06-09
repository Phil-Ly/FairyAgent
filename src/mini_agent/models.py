"""Model client abstractions and OpenAI implementation."""

from __future__ import annotations

import json
from typing import Protocol

from openai import OpenAI


class ModelClient(Protocol):
    """Small interface used by the agent loop."""

    def generate(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> dict:
        """Generate either a final answer or tool calls."""


class OpenAIModelClient:
    """OpenAI Chat Completions client with normalized responses."""

    def __init__(self, api_key: str, model_name: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name

    def generate(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> dict:
        """Call OpenAI and return the harness response format."""

        kwargs: dict = {
            "model": self._model_name,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        if message.tool_calls:
            return {
                "type": "tool_call",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": self._parse_arguments(
                            tool_call.function.arguments
                        ),
                    }
                    for tool_call in message.tool_calls
                ],
            }

        return {
            "type": "final",
            "content": message.content or "",
            "tool_calls": [],
        }

    @staticmethod
    def _parse_arguments(arguments: str) -> dict:
        """Parse model-supplied tool arguments as JSON objects."""

        parsed = json.loads(arguments or "{}")
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        return parsed
