import os

import pytest

from agentloop.agent import AgentLoop
from agentloop.config import AppConfig
from agentloop.memory import Memory
from agentloop.models import build_model
from agentloop.tools import get_default_tools


@pytest.mark.integration
def test_model_provider_can_answer_without_tool() -> None:
    """Optional provider smoke test, enabled only with real credentials."""

    if os.getenv("RUN_MODEL_INTEGRATION_TESTS") != "1":
        pytest.skip("Set RUN_MODEL_INTEGRATION_TESTS=1 to run provider tests.")

    api_key = os.getenv("MODEL_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("MODEL_API_KEY or OPENAI_API_KEY is required.")

    pytest.importorskip("langchain_openai")

    agent = AgentLoop(
        model=build_model(
            AppConfig(
                MODEL_PROVIDER=os.getenv("MODEL_PROVIDER", "openai_compatible"),
                MODEL_API_KEY=api_key,
                MODEL_BASE_URL=os.getenv("MODEL_BASE_URL"),
                MODEL_NAME=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
                MAX_STEPS=2,
            )
        ),
        tools=get_default_tools(),
        memory=Memory(),
        max_steps=2,
    )

    result = agent.run("Reply with exactly: ok")

    assert "ok" in result.lower()
