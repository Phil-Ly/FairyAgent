"""LangChain model construction."""

from __future__ import annotations

import importlib

from mini_agent.config import AppConfig


class ModelConfigurationError(RuntimeError):
    """Raised when the configured chat model cannot be built."""


def build_model(config: AppConfig):
    """Build the chat model used by the LangGraph agent."""

    if config.model_provider == "openai_compatible":
        return build_openai_compatible_model(config)

    raise ModelConfigurationError(
        f"Unsupported MODEL_PROVIDER: {config.model_provider}."
    )


def build_openai_compatible_model(config: AppConfig):
    """Build a ChatOpenAI model for OpenAI-compatible providers."""

    if not config.resolved_model_api_key:
        raise ModelConfigurationError(
            "MODEL_API_KEY is required for MODEL_PROVIDER=openai_compatible."
        )
    try:
        langchain_openai = importlib.import_module("langchain_openai")
    except ModuleNotFoundError as exc:
        if exc.name != "langchain_openai":
            raise
        raise ModelConfigurationError(
            "langchain-openai is required for the run command. "
            "Install dependencies with `uv sync --extra openai-compatible`."
        ) from exc

    return langchain_openai.ChatOpenAI(
        api_key=config.resolved_model_api_key,
        base_url=config.model_base_url,
        model=config.model_name,
        temperature=config.model_temperature,
        timeout=config.model_timeout_seconds,
    )
