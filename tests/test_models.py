import sys
import types

import pytest

from agentloop.config import AppConfig
from agentloop.models import ModelConfigurationError, build_model


class FakeChatOpenAI:
    """Capture ChatOpenAI constructor arguments without network calls."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_build_model_uses_openai_compatible_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    config = AppConfig(
        MODEL_PROVIDER="openai_compatible",
        MODEL_NAME="deepseek-chat",
        MODEL_BASE_URL="https://api.deepseek.com",
        MODEL_API_KEY="test-key",
        MODEL_TEMPERATURE=0.2,
        MODEL_TIMEOUT_SECONDS=45,
        MAX_STEPS=8,
    )

    model = build_model(config)

    assert isinstance(model, FakeChatOpenAI)
    assert model.kwargs == {
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "temperature": 0.2,
        "timeout": 45,
    }


def test_build_model_falls_back_to_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    config = AppConfig(
        MODEL_PROVIDER="openai_compatible",
        MODEL_NAME="gpt-4.1-mini",
        OPENAI_API_KEY="legacy-key",
        MAX_STEPS=8,
    )

    model = build_model(config)

    assert model.kwargs["api_key"] == "legacy-key"


def test_build_model_uses_provider_preset_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    config = AppConfig(
        MODEL_PROVIDER="deepseek",
        MODEL_NAME="deepseek-chat",
        MODEL_API_KEY="test-key",
        MAX_STEPS=8,
    )

    model = build_model(config)

    assert model.kwargs["base_url"] == "https://api.deepseek.com"


def test_build_model_allows_base_url_override_for_provider_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    config = AppConfig(
        MODEL_PROVIDER="qwen",
        MODEL_NAME="qwen-plus",
        MODEL_API_KEY="test-key",
        MODEL_BASE_URL="https://proxy.example/v1",
        MAX_STEPS=8,
    )

    model = build_model(config)

    assert model.kwargs["base_url"] == "https://proxy.example/v1"


def test_build_model_rejects_missing_compatible_key() -> None:
    config = AppConfig(
        MODEL_PROVIDER="openai_compatible",
        MODEL_NAME="deepseek-chat",
        MODEL_BASE_URL="https://api.deepseek.com",
        MAX_STEPS=8,
    )

    with pytest.raises(ModelConfigurationError, match="MODEL_API_KEY is required"):
        build_model(config)


def test_build_model_rejects_unknown_provider() -> None:
    config = AppConfig(
        MODEL_PROVIDER="unknown",
        MODEL_NAME="deepseek-chat",
        MODEL_API_KEY="test-key",
        MAX_STEPS=8,
    )

    with pytest.raises(ModelConfigurationError) as exc_info:
        build_model(config)

    assert "Unsupported MODEL_PROVIDER: unknown" in str(exc_info.value)
    assert "deepseek" in str(exc_info.value)
