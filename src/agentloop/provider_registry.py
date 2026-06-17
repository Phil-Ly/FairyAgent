"""Model provider preset registry."""

from __future__ import annotations

from dataclasses import dataclass


class UnknownProviderError(RuntimeError):
    """Raised when MODEL_PROVIDER does not match a registered preset."""


@dataclass(frozen=True)
class ProviderPreset:
    """OpenAI-compatible provider metadata used by model construction."""

    name: str
    aliases: tuple[str, ...] = ()
    base_url: str | None = None
    dependency_module: str = "langchain_openai"
    dependency_package: str = "langchain-openai"
    dependency_extra: str = "openai-compatible"
    model_factory: str = "ChatOpenAI"

    def resolve_base_url(self, override: str | None) -> str | None:
        """Return an explicit override or this preset's default base URL."""

        return override or self.base_url

    @property
    def all_names(self) -> tuple[str, ...]:
        """Return canonical name plus aliases."""

        return (self.name, *self.aliases)


class ProviderRegistry:
    """Registry of provider names and OpenAI-compatible presets."""

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
)

DEFAULT_PROVIDER_REGISTRY = ProviderRegistry(DEFAULT_PROVIDER_PRESETS)
