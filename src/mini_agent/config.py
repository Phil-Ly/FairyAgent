"""Configuration loading for the agent harness."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Runtime configuration read from environment variables."""

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-4.1-mini", alias="MODEL_NAME")
    max_steps: int = Field(default=8, alias="MAX_STEPS", gt=0)


def load_config() -> AppConfig:
    """Load configuration from .env and the process environment."""

    load_dotenv()
    return AppConfig(
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
        MODEL_NAME=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
        MAX_STEPS=int(os.getenv("MAX_STEPS", "8")),
    )
