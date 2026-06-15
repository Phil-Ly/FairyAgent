"""Configuration loading for the agent harness."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, computed_field, field_validator


class ConfigurationError(RuntimeError):
    """Raised when environment configuration is invalid."""


class AppConfig(BaseModel):
    """Runtime configuration read from environment variables."""

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    model_provider: str = Field(default="openai_compatible", alias="MODEL_PROVIDER")
    model_api_key: str | None = Field(default=None, alias="MODEL_API_KEY")
    model_base_url: str | None = Field(default=None, alias="MODEL_BASE_URL")
    model_name: str = Field(default="gpt-4.1-mini", alias="MODEL_NAME")
    model_temperature: float = Field(default=0, alias="MODEL_TEMPERATURE", ge=0)
    model_timeout_seconds: int = Field(
        default=60,
        alias="MODEL_TIMEOUT_SECONDS",
        gt=0,
    )
    max_steps: int = Field(default=8, alias="MAX_STEPS", gt=0)

    @computed_field
    @property
    def resolved_model_api_key(self) -> str | None:
        """Return the preferred model API key with legacy fallback."""

        return self.model_api_key or self.openai_api_key

    @field_validator("openai_api_key", "model_api_key", "model_base_url")
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
    model_temperature = os.getenv("MODEL_TEMPERATURE", "0")
    if not _is_float(model_temperature):
        raise ConfigurationError("MODEL_TEMPERATURE must be a number.")
    model_timeout = os.getenv("MODEL_TIMEOUT_SECONDS", "60")
    if not _is_integer(model_timeout):
        raise ConfigurationError("MODEL_TIMEOUT_SECONDS must be an integer.")

    try:
        return AppConfig(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
            MODEL_PROVIDER=os.getenv("MODEL_PROVIDER", "openai_compatible"),
            MODEL_API_KEY=os.getenv("MODEL_API_KEY"),
            MODEL_BASE_URL=os.getenv("MODEL_BASE_URL"),
            MODEL_NAME=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            MODEL_TEMPERATURE=float(model_temperature),
            MODEL_TIMEOUT_SECONDS=int(model_timeout),
            MAX_STEPS=int(max_steps),
        )
    except ValidationError as exc:
        raise ConfigurationError(_format_validation_error(exc)) from exc


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
    return "Invalid configuration."
