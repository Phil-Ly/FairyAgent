"""LangChain model construction."""

from __future__ import annotations

import importlib

from agentloop.config import AppConfig
from agentloop.provider_registry import (
    DEFAULT_PROVIDER_REGISTRY,
    ProviderPreset,
    UnknownProviderError,
)


class ModelConfigurationError(RuntimeError):
    """Raised when the configured chat model cannot be built."""


def build_model(config: AppConfig):
    """Build the chat model used by the LangGraph agent."""

    try:
        preset = DEFAULT_PROVIDER_REGISTRY.get(config.model_provider)
    except UnknownProviderError as exc:
        raise ModelConfigurationError(str(exc)) from exc
    return build_openai_compatible_model(config, preset)


def build_openai_compatible_model(
    config: AppConfig,
    preset: ProviderPreset | None = None,
):
    """Build a ChatOpenAI model for OpenAI-compatible providers."""

    preset = preset or DEFAULT_PROVIDER_REGISTRY.get("openai_compatible")
    if not config.resolved_model_api_key:
        raise ModelConfigurationError(
            f"MODEL_API_KEY is required for MODEL_PROVIDER={config.model_provider}."
        )
    try:
        langchain_openai = importlib.import_module(preset.dependency_module)
    except ModuleNotFoundError as exc:
        if exc.name != preset.dependency_module:
            raise
        raise ModelConfigurationError(
            f"{preset.dependency_package} is required for MODEL_PROVIDER="
            f"{config.model_provider}. Install dependencies with "
            f"`uv sync --extra {preset.dependency_extra}`."
        ) from exc

    model_factory = getattr(langchain_openai, preset.model_factory)
    return model_factory(
        api_key=config.resolved_model_api_key,
        base_url=preset.resolve_base_url(config.model_base_url),
        model=config.model_name,
        temperature=config.model_temperature,
        timeout=config.model_timeout_seconds,
    )
