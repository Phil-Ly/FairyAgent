"""LangChain model construction."""

from __future__ import annotations

from mini_agent.config import AppConfig


def build_model(config: AppConfig):
    """Build the chat model used by the LangGraph agent."""

    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required.")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=config.openai_api_key,
        model=config.model_name,
        temperature=0,
    )
