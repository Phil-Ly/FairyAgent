"""Model provider preset registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class UnknownProviderError(RuntimeError):
    """Raised when MODEL_PROVIDER does not match a registered preset."""


class ModelProtocol(StrEnum):
    """Protocols supported by the built-in model builders."""

    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GOOGLE_GENAI = "google_genai"


@dataclass(frozen=True)
class ProviderPreset:
    """Provider metadata used for protocol-aware model construction."""

    name: str
    protocol: ModelProtocol = ModelProtocol.OPENAI_COMPATIBLE
    aliases: tuple[str, ...] = ()
    base_url: str | None = None
    dependency_module: str = "langchain_openai"
    dependency_package: str = "langchain-openai"
    dependency_extra: str = "openai-compatible"
    model_factory: str = "ChatOpenAI"
    api_key_env_vars: tuple[str, ...] = ("OPENAI_API_KEY",)

    def resolve_base_url(self, override: str | None) -> str | None:
        """Return an explicit override or this preset's default base URL."""

        return override or self.base_url

    @property
    def all_names(self) -> tuple[str, ...]:
        """Return canonical name plus aliases."""

        return (self.name, *self.aliases)


class ProviderRegistry:
    """Registry of provider names and protocol-aware presets."""

    def __init__(self, presets: tuple[ProviderPreset, ...] | None = None) -> None:
        self._presets = presets or DEFAULT_PROVIDER_PRESETS
        self._by_name = {
            _normalize_provider_name(name): preset
            for preset in self._presets
            for name in preset.all_names
        }

    def get(self, provider_name: str) -> ProviderPreset:
        """Return a provider preset by canonical name or alias."""

        normalized = _normalize_provider_name(provider_name)
        try:
            return self._by_name[normalized]
        except KeyError as exc:
            supported = ", ".join(self.supported_provider_names())
            raise UnknownProviderError(
                f"Unsupported MODEL_PROVIDER: {provider_name}. "
                f"Supported providers: {supported}."
            ) from exc

    def supported_provider_names(self) -> tuple[str, ...]:
        """Return canonical provider names in display order."""

        return tuple(preset.name for preset in self._presets)


def _normalize_provider_name(provider_name: str) -> str:
    return provider_name.strip().lower().replace("-", "_")


DEFAULT_PROVIDER_PRESETS = (
    ProviderPreset(
        name="openai_compatible",
        aliases=("openai-compatible", "compatible"),
    ),
    ProviderPreset(name="openai", aliases=("openai_official",)),
    ProviderPreset(name="deepseek", base_url="https://api.deepseek.com"),
    ProviderPreset(
        name="qwen",
        aliases=("dashscope", "aliyun"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    ProviderPreset(
        name="kimi",
        aliases=("moonshot",),
        base_url="https://api.moonshot.cn/v1",
    ),
    ProviderPreset(
        name="anthropic",
        protocol=ModelProtocol.ANTHROPIC_MESSAGES,
        aliases=("claude",),
        dependency_module="langchain_anthropic",
        dependency_package="langchain-anthropic",
        dependency_extra="anthropic",
        model_factory="ChatAnthropic",
        api_key_env_vars=("ANTHROPIC_API_KEY",),
    ),
    ProviderPreset(
        name="gemini",
        protocol=ModelProtocol.GOOGLE_GENAI,
        aliases=("google_genai",),
        dependency_module="langchain_google_genai",
        dependency_package="langchain-google-genai",
        dependency_extra="gemini",
        model_factory="ChatGoogleGenerativeAI",
        api_key_env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    ),
)

DEFAULT_PROVIDER_REGISTRY = ProviderRegistry(DEFAULT_PROVIDER_PRESETS)
