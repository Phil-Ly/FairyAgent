import sys
import types
from typing import Any, cast

import pytest

from agentloop.config import AppConfig
from agentloop.models import ModelConfigurationError, build_model


class FakeChatOpenAI:
    """Capture ChatOpenAI constructor arguments without network calls."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeChatAnthropic:
    """Capture ChatAnthropic constructor arguments without network calls."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeChatGoogleGenerativeAI:
    """Capture Gemini constructor arguments without network calls."""

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

    model = cast(Any, build_model(config))

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

    model = cast(Any, build_model(config))

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

    model = cast(Any, build_model(config))

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

    model = cast(Any, build_model(config))

    assert model.kwargs["base_url"] == "https://proxy.example/v1"


def test_build_model_dispatches_anthropic_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace(ChatAnthropic=FakeChatAnthropic)
    monkeypatch.setitem(sys.modules, "langchain_anthropic", fake_module)
    config = AppConfig(
        MODEL_PROVIDER="anthropic",
        MODEL_NAME="claude-sonnet-4-6",
        ANTHROPIC_API_KEY="anthropic-key",
        MODEL_TEMPERATURE=0.4,
        MODEL_TIMEOUT_SECONDS=30,
        MAX_STEPS=8,
    )

    model = cast(Any, build_model(config))

    assert isinstance(model, FakeChatAnthropic)
    assert model.kwargs == {
        "api_key": "anthropic-key",
        "model": "claude-sonnet-4-6",
        "temperature": 0.4,
        "timeout": 30,
    }


def test_build_model_dispatches_google_genai_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace(
        ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI
    )
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)
    config = AppConfig(
        MODEL_PROVIDER="gemini",
        MODEL_NAME="gemini-2.5-flash",
        GOOGLE_API_KEY="google-key",
        MODEL_BASE_URL="https://gemini.example",
        MODEL_TIMEOUT_SECONDS=25,
        MAX_STEPS=8,
    )

    model = cast(Any, build_model(config))

    assert isinstance(model, FakeChatGoogleGenerativeAI)
    assert model.kwargs == {
        "api_key": "google-key",
        "base_url": "https://gemini.example",
        "model": "gemini-2.5-flash",
        "timeout": 25,
    }


def test_build_model_prefers_model_api_key_for_native_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace(ChatAnthropic=FakeChatAnthropic)
    monkeypatch.setitem(sys.modules, "langchain_anthropic", fake_module)
    config = AppConfig(
        MODEL_PROVIDER="anthropic",
        MODEL_NAME="claude-sonnet-4-6",
        MODEL_API_KEY="explicit-key",
        ANTHROPIC_API_KEY="provider-key",
        MAX_STEPS=8,
    )

    model = cast(Any, build_model(config))

    assert model.kwargs["api_key"] == "explicit-key"


def test_build_model_falls_back_to_gemini_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace(
        ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI
    )
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)
    config = AppConfig(
        MODEL_PROVIDER="gemini",
        MODEL_NAME="gemini-2.5-flash",
        GEMINI_API_KEY="gemini-key",
        MAX_STEPS=8,
    )

    model = cast(Any, build_model(config))

    assert model.kwargs["api_key"] == "gemini-key"


def test_build_model_prefers_google_api_key_over_gemini_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace(
        ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI
    )
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)
    config = AppConfig(
        MODEL_PROVIDER="gemini",
        MODEL_NAME="gemini-2.5-flash",
        GOOGLE_API_KEY="google-key",
        GEMINI_API_KEY="gemini-key",
        MAX_STEPS=8,
    )

    model = cast(Any, build_model(config))

    assert model.kwargs["api_key"] == "google-key"


def test_build_model_rejects_missing_anthropic_key() -> None:
    config = AppConfig(
        MODEL_PROVIDER="anthropic",
        MODEL_NAME="claude-sonnet-4-6",
        MAX_STEPS=8,
    )

    with pytest.raises(
        ModelConfigurationError,
        match="MODEL_API_KEY or ANTHROPIC_API_KEY is required",
    ):
        build_model(config)


def test_build_model_reports_missing_anthropic_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing_dependency(module_name: str):
        raise ModuleNotFoundError(
            f"No module named '{module_name}'",
            name=module_name,
        )

    monkeypatch.setattr(
        "agentloop.models.importlib.import_module",
        raise_missing_dependency,
    )
    config = AppConfig(
        MODEL_PROVIDER="anthropic",
        MODEL_NAME="claude-sonnet-4-6",
        MODEL_API_KEY="test-key",
        MAX_STEPS=8,
    )

    with pytest.raises(ModelConfigurationError) as exc_info:
        build_model(config)

    message = str(exc_info.value)
    assert "langchain-anthropic is required" in message
    assert "uv sync --extra anthropic" in message


def test_build_model_rejects_missing_compatible_key() -> None:
    config = AppConfig(
        MODEL_PROVIDER="openai_compatible",
        MODEL_NAME="deepseek-chat",
        MODEL_BASE_URL="https://api.deepseek.com",
        MAX_STEPS=8,
    )

    with pytest.raises(
        ModelConfigurationError,
        match="MODEL_API_KEY or OPENAI_API_KEY is required",
    ):
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
