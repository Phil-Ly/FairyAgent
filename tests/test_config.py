import pytest

from agentloop.config import ConfigurationError, load_config


def test_load_config_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("MODEL_TEMPERATURE", raising=False)
    monkeypatch.delenv("MODEL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("MAX_STEPS", raising=False)
    monkeypatch.delenv("CONTEXT_MAX_TOKENS", raising=False)
    monkeypatch.delenv("CONTEXT_RESERVED_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("CONTEXT_SUMMARY_MAX_TOKENS", raising=False)

    config = load_config()

    assert config.openai_api_key is None
    assert config.anthropic_api_key is None
    assert config.google_api_key is None
    assert config.gemini_api_key is None
    assert config.model_provider == "openai_compatible"
    assert config.model_api_key is None
    assert config.model_base_url is None
    assert config.model_name == "gpt-4.1-mini"
    assert config.model_temperature is None
    assert config.model_timeout_seconds == 60
    assert config.max_steps == 8
    assert config.context_max_tokens == 8000
    assert config.context_reserved_output_tokens == 1000
    assert config.context_summary_max_tokens == 800


def test_load_config_reads_openai_compatible_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_BASE_URL", " https://api.deepseek.com ")
    monkeypatch.setenv("MODEL_NAME", " deepseek-chat ")
    monkeypatch.setenv("MODEL_TEMPERATURE", "0.3")
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("CONTEXT_MAX_TOKENS", "12000")
    monkeypatch.setenv("CONTEXT_RESERVED_OUTPUT_TOKENS", "1500")
    monkeypatch.setenv("CONTEXT_SUMMARY_MAX_TOKENS", "900")

    config = load_config()

    assert config.model_provider == "openai_compatible"
    assert config.model_api_key == "test-key"
    assert config.resolved_model_api_key == "test-key"
    assert config.model_base_url == "https://api.deepseek.com"
    assert config.model_name == "deepseek-chat"
    assert config.model_temperature == 0.3
    assert config.model_timeout_seconds == 45
    assert config.context_max_tokens == 12000
    assert config.context_reserved_output_tokens == 1500
    assert config.context_summary_max_tokens == 900


def test_load_config_rejects_non_integer_max_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_STEPS", "abc")

    with pytest.raises(ConfigurationError, match="MAX_STEPS must be an integer"):
        load_config()


def test_load_config_rejects_non_positive_max_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_STEPS", "0")

    with pytest.raises(ConfigurationError, match="MAX_STEPS must be greater than 0"):
        load_config()


def test_load_config_rejects_empty_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_NAME", "   ")

    with pytest.raises(ConfigurationError, match="MODEL_NAME cannot be empty"):
        load_config()


def test_load_config_treats_blank_openai_key_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "   ")

    config = load_config()

    assert config.openai_api_key is None


def test_load_config_falls_back_to_legacy_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")

    config = load_config()

    assert config.model_api_key is None
    assert config.openai_api_key == "legacy-key"
    assert config.resolved_model_api_key == "legacy-key"


def test_load_config_reads_native_provider_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", " anthropic-key ")
    monkeypatch.setenv("GOOGLE_API_KEY", " google-key ")
    monkeypatch.setenv("GEMINI_API_KEY", " gemini-key ")

    config = load_config()

    assert config.anthropic_api_key == "anthropic-key"
    assert config.google_api_key == "google-key"
    assert config.gemini_api_key == "gemini-key"


def test_load_config_rejects_empty_model_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "   ")

    with pytest.raises(ConfigurationError, match="MODEL_PROVIDER cannot be empty"):
        load_config()


def test_load_config_rejects_non_number_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_TEMPERATURE", "hot")

    with pytest.raises(ConfigurationError, match="MODEL_TEMPERATURE must be a number"):
        load_config()


def test_load_config_rejects_negative_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "0")

    with pytest.raises(
        ConfigurationError,
        match="MODEL_TIMEOUT_SECONDS must be greater than 0",
    ):
        load_config()


def test_load_config_rejects_non_integer_context_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXT_MAX_TOKENS", "many")

    with pytest.raises(
        ConfigurationError,
        match="CONTEXT_MAX_TOKENS must be an integer",
    ):
        load_config()


def test_load_config_rejects_context_reserved_tokens_exceeding_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXT_MAX_TOKENS", "100")
    monkeypatch.setenv("CONTEXT_RESERVED_OUTPUT_TOKENS", "100")

    with pytest.raises(
        ConfigurationError,
        match="CONTEXT_RESERVED_OUTPUT_TOKENS must be less than CONTEXT_MAX_TOKENS",
    ):
        load_config()
