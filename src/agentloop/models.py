"""LangChain model construction."""

from __future__ import annotations

import importlib
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agentloop.config import AppConfig
from agentloop.provider_registry import (
    DEFAULT_PROVIDER_REGISTRY,
    ModelProtocol,
    ProviderPreset,
    UnknownProviderError,
)


class ModelConfigurationError(RuntimeError):
    """Raised when the configured chat model cannot be built."""


def build_model(config: AppConfig) -> BaseChatModel:
    """Build the chat model used by the LangGraph agent."""

    try:
        preset = DEFAULT_PROVIDER_REGISTRY.get(config.model_provider)
    except UnknownProviderError as exc:
        raise ModelConfigurationError(str(exc)) from exc

    match preset.protocol:
        case ModelProtocol.OPENAI_COMPATIBLE:
            return build_openai_compatible_model(config, preset)
        case ModelProtocol.ANTHROPIC_MESSAGES:
            return build_anthropic_model(config, preset)
        case ModelProtocol.GOOGLE_GENAI:
            return build_google_genai_model(config, preset)

    raise ModelConfigurationError(
        f"Unsupported model protocol: {preset.protocol}."
    )


def build_openai_compatible_model(
    config: AppConfig,
    preset: ProviderPreset | None = None,
) -> BaseChatModel:
    """Build a ChatOpenAI model for OpenAI-compatible providers."""

    preset = preset or DEFAULT_PROVIDER_REGISTRY.get("openai_compatible")
    return _instantiate_model(config, preset)


def build_anthropic_model(
    config: AppConfig,
    preset: ProviderPreset | None = None,
) -> BaseChatModel:
    """Build a ChatAnthropic model for the native Messages API."""

    preset = preset or DEFAULT_PROVIDER_REGISTRY.get("anthropic")
    return _instantiate_model(config, preset)


def build_google_genai_model(
    config: AppConfig,
    preset: ProviderPreset | None = None,
) -> BaseChatModel:
    """Build a ChatGoogleGenerativeAI model for the native Gemini API."""

    preset = preset or DEFAULT_PROVIDER_REGISTRY.get("gemini")
    return _instantiate_model(config, preset)


def _instantiate_model(
    config: AppConfig,
    preset: ProviderPreset,
) -> BaseChatModel:
    api_key = config.resolve_model_api_key(preset.api_key_env_vars)
    if not api_key:
        accepted_keys = " or ".join(("MODEL_API_KEY", *preset.api_key_env_vars))
        raise ModelConfigurationError(
            f"{accepted_keys} is required for MODEL_PROVIDER="
            f"{config.model_provider}."
        )

    model_factory = _load_model_factory(preset, config.model_provider)
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "model": config.model_name,
        "timeout": config.model_timeout_seconds,
    }
    if config.model_temperature is not None:
        kwargs["temperature"] = config.model_temperature
    base_url = preset.resolve_base_url(config.model_base_url)
    if base_url is not None:
        kwargs["base_url"] = base_url
    return model_factory(**kwargs)


def _load_model_factory(preset: ProviderPreset, configured_provider: str):
    try:
        provider_module = importlib.import_module(preset.dependency_module)
    except ModuleNotFoundError as exc:
        if exc.name != preset.dependency_module:
            raise
        raise ModelConfigurationError(
            f"{preset.dependency_package} is required for MODEL_PROVIDER="
            f"{configured_provider}. Install dependencies with "
            f"`uv sync --extra {preset.dependency_extra}`."
        ) from exc
    return getattr(provider_module, preset.model_factory)
