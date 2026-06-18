import pytest

from agentloop.provider_registry import (
    DEFAULT_PROVIDER_REGISTRY,
    ProviderRegistry,
    UnknownProviderError,
)


def test_default_provider_registry_resolves_named_openai_compatible_presets() -> None:
    registry = DEFAULT_PROVIDER_REGISTRY

    assert registry.get("openai_compatible").base_url is None
    assert registry.get("deepseek").base_url == "https://api.deepseek.com"
    assert registry.get("qwen").base_url == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    assert registry.get("kimi").base_url == "https://api.moonshot.cn/v1"
    assert registry.get("moonshot").name == "kimi"


def test_provider_registry_reports_supported_names_for_unknown_provider() -> None:
    registry = ProviderRegistry()

    with pytest.raises(UnknownProviderError) as exc_info:
        registry.get("missing")

    assert "Unsupported MODEL_PROVIDER: missing" in str(exc_info.value)
    assert "openai_compatible" in str(exc_info.value)


def test_provider_preset_prefers_explicit_base_url() -> None:
    preset = DEFAULT_PROVIDER_REGISTRY.get("deepseek")

    assert preset.resolve_base_url("https://proxy.example/v1") == (
        "https://proxy.example/v1"
    )
    assert preset.resolve_base_url(None) == "https://api.deepseek.com"
