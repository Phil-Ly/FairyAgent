"""Configuration loading for the agent harness."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, computed_field, field_validator


class ConfigurationError(RuntimeError):
    """Raised when environment configuration is invalid."""


class AppConfig(BaseModel):
    """Runtime configuration read from environment variables."""

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    model_provider: str = Field(default="openai_compatible", alias="MODEL_PROVIDER")
    model_api_key: str | None = Field(default=None, alias="MODEL_API_KEY")
    model_base_url: str | None = Field(default=None, alias="MODEL_BASE_URL")
    model_name: str = Field(default="gpt-4.1-mini", alias="MODEL_NAME")
    model_temperature: float | None = Field(
        default=None,
        alias="MODEL_TEMPERATURE",
        ge=0,
    )
    model_timeout_seconds: int = Field(
        default=60,
        alias="MODEL_TIMEOUT_SECONDS",
        gt=0,
    )
    max_steps: int = Field(default=8, alias="MAX_STEPS", gt=0)
    context_max_tokens: int = Field(default=8000, alias="CONTEXT_MAX_TOKENS", gt=0)
    context_reserved_output_tokens: int = Field(
        default=1000,
        alias="CONTEXT_RESERVED_OUTPUT_TOKENS",
        ge=0,
    )
    context_summary_max_tokens: int = Field(
        default=800,
        alias="CONTEXT_SUMMARY_MAX_TOKENS",
        gt=0,
    )

    @computed_field
    @property
    def resolved_model_api_key(self) -> str | None:
        """Return the preferred model API key with legacy fallback."""

        return self.resolve_model_api_key(("OPENAI_API_KEY",))

    def resolve_model_api_key(
        self,
        fallback_env_vars: tuple[str, ...],
    ) -> str | None:
        """Resolve an explicit key or the configured provider-specific fallback."""

        if self.model_api_key:
            return self.model_api_key
        for env_var in fallback_env_vars:
            value = getattr(self, env_var.lower(), None)
            if value:
                return value
        return None

    @field_validator(
        "openai_api_key",
        "anthropic_api_key",
        "google_api_key",
        "gemini_api_key",
        "model_api_key",
        "model_base_url",
    )
    @classmethod
    def normalize_optional_string(cls, value: str | None) -> str | None:
        """Trim optional strings and treat blank values as missing."""

        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("model_provider", "model_name")
    @classmethod
    def validate_required_string(cls, value: str, info) -> str:
        """Trim and reject blank required model fields."""

        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{info.field_name} cannot be empty.")
        return stripped


def load_config() -> AppConfig:
    """Load configuration from .env and the process environment."""

    load_dotenv()
    max_steps = os.getenv("MAX_STEPS", "8")
    if not _is_integer(max_steps):
        raise ConfigurationError("MAX_STEPS must be an integer.")
    model_temperature = os.getenv("MODEL_TEMPERATURE")
    if model_temperature is not None and not _is_float(model_temperature):
        raise ConfigurationError("MODEL_TEMPERATURE must be a number.")
    model_timeout = os.getenv("MODEL_TIMEOUT_SECONDS", "60")
    if not _is_integer(model_timeout):
        raise ConfigurationError("MODEL_TIMEOUT_SECONDS must be an integer.")
    context_max_tokens = os.getenv("CONTEXT_MAX_TOKENS", "8000")
    if not _is_integer(context_max_tokens):
        raise ConfigurationError("CONTEXT_MAX_TOKENS must be an integer.")
    context_reserved_output_tokens = os.getenv(
        "CONTEXT_RESERVED_OUTPUT_TOKENS",
        "1000",
    )
    if not _is_integer(context_reserved_output_tokens):
        raise ConfigurationError(
            "CONTEXT_RESERVED_OUTPUT_TOKENS must be an integer."
        )
    context_summary_max_tokens = os.getenv("CONTEXT_SUMMARY_MAX_TOKENS", "800")
    if not _is_integer(context_summary_max_tokens):
        raise ConfigurationError("CONTEXT_SUMMARY_MAX_TOKENS must be an integer.")

    try:
        config = AppConfig(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
            ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY"),
            GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY"),
            GEMINI_API_KEY=os.getenv("GEMINI_API_KEY"),
            MODEL_PROVIDER=os.getenv("MODEL_PROVIDER", "openai_compatible"),
            MODEL_API_KEY=os.getenv("MODEL_API_KEY"),
            MODEL_BASE_URL=os.getenv("MODEL_BASE_URL"),
            MODEL_NAME=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            MODEL_TEMPERATURE=(
                float(model_temperature) if model_temperature is not None else None
            ),
            MODEL_TIMEOUT_SECONDS=int(model_timeout),
            MAX_STEPS=int(max_steps),
            CONTEXT_MAX_TOKENS=int(context_max_tokens),
            CONTEXT_RESERVED_OUTPUT_TOKENS=int(context_reserved_output_tokens),
            CONTEXT_SUMMARY_MAX_TOKENS=int(context_summary_max_tokens),
        )
    except ValidationError as exc:
        raise ConfigurationError(_format_validation_error(exc)) from exc
    if config.context_reserved_output_tokens >= config.context_max_tokens:
        raise ConfigurationError(
            "CONTEXT_RESERVED_OUTPUT_TOKENS must be less than CONTEXT_MAX_TOKENS."
        )
    return config


def _is_integer(value: str) -> bool:
    """Return True when value can be parsed as an integer."""

    try:
        int(value)
    except ValueError:
        return False
    return True


def _is_float(value: str) -> bool:
    """Return True when value can be parsed as a float."""

    try:
        float(value)
    except ValueError:
        return False
    return True


def _format_validation_error(error: ValidationError) -> str:
    """Convert pydantic validation errors into user-facing messages."""

    for item in error.errors():
        field = item.get("loc", ["configuration"])[0]
        if field == "MAX_STEPS":
            return "MAX_STEPS must be greater than 0."
        if field == "MODEL_PROVIDER":
            return "MODEL_PROVIDER cannot be empty."
        if field == "MODEL_NAME":
            return "MODEL_NAME cannot be empty."
        if field == "MODEL_TEMPERATURE":
            return "MODEL_TEMPERATURE must be greater than or equal to 0."
        if field == "MODEL_TIMEOUT_SECONDS":
            return "MODEL_TIMEOUT_SECONDS must be greater than 0."
        if field == "CONTEXT_MAX_TOKENS":
            return "CONTEXT_MAX_TOKENS must be greater than 0."
        if field == "CONTEXT_RESERVED_OUTPUT_TOKENS":
            return "CONTEXT_RESERVED_OUTPUT_TOKENS must be greater than or equal to 0."
        if field == "CONTEXT_SUMMARY_MAX_TOKENS":
            return "CONTEXT_SUMMARY_MAX_TOKENS must be greater than 0."
    return "Invalid configuration."
