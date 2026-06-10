"""LangChain model construction."""

from __future__ import annotations

from mini_agent.config import AppConfig


class ModelConfigurationError(RuntimeError):
    """Raised when the configured chat model cannot be built."""


def build_model(config: AppConfig):
    """Build the chat model used by the LangGraph agent."""

    if not config.openai_api_key:
        raise ModelConfigurationError("OPENAI_API_KEY is required.")
    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:
        if exc.name != "langchain_openai":
            raise
        raise ModelConfigurationError(
            "langchain-openai is required for the run command. "
            "Install dependencies with `uv sync`."
        ) from exc

    return ChatOpenAI(
        api_key=config.openai_api_key,
        model=config.model_name,
        temperature=0,
    )
