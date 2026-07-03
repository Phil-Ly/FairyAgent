import pytest

from agentloop.provider_registry import (
    DEFAULT_PROVIDER_REGISTRY,
    ProviderRegistry,
    UnknownProviderError,
)


def test_default_provider_registry_resolves_named_openai_compatible_presets() -> None:
    registry = DEFAULT_PROVIDER_REGISTRY

    assert registry.get("openai_compatible").protocol.value == "openai_compatible"
    assert registry.get("openai_compatible").base_url is None
    assert registry.get("deepseek").base_url == "https://api.deepseek.com"
    assert registry.get("qwen").base_url == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    assert registry.get("kimi").base_url == "https://api.moonshot.cn/v1"
    assert registry.get("moonshot").name == "kimi"


def test_default_provider_registry_resolves_native_protocol_presets() -> None:
    registry = DEFAULT_PROVIDER_REGISTRY

    anthropic = registry.get("claude")
    assert anthropic.name == "anthropic"
    assert anthropic.protocol.value == "anthropic_messages"
    assert anthropic.dependency_module == "langchain_anthropic"
    assert anthropic.dependency_package == "langchain-anthropic"
    assert anthropic.dependency_extra == "anthropic"
    assert anthropic.model_factory == "ChatAnthropic"
    assert anthropic.api_key_env_vars == ("ANTHROPIC_API_KEY",)

    gemini = registry.get("google-genai")
    assert gemini.name == "gemini"
    assert gemini.protocol.value == "google_genai"
    assert gemini.dependency_module == "langchain_google_genai"
    assert gemini.dependency_package == "langchain-google-genai"
    assert gemini.dependency_extra == "gemini"
    assert gemini.model_factory == "ChatGoogleGenerativeAI"
    assert gemini.api_key_env_vars == ("GOOGLE_API_KEY", "GEMINI_API_KEY")


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
