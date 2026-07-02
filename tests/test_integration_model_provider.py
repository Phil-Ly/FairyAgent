import os

import pytest

from agentloop.agent import AgentLoop
from agentloop.config import load_config
from agentloop.memory import Memory
from agentloop.models import build_model
from agentloop.provider_registry import DEFAULT_PROVIDER_REGISTRY
from agentloop.tools import get_default_tools


@pytest.mark.integration
def test_model_provider_can_answer_without_tool() -> None:
    """Optional provider smoke test, enabled only with real credentials."""

    if os.getenv("RUN_MODEL_INTEGRATION_TESTS") != "1":
        pytest.skip("Set RUN_MODEL_INTEGRATION_TESTS=1 to run provider tests.")

    config = load_config()
    preset = DEFAULT_PROVIDER_REGISTRY.get(config.model_provider)
    api_key = config.resolve_model_api_key(preset.api_key_env_vars)
    if not api_key:
        accepted_keys = " or ".join(("MODEL_API_KEY", *preset.api_key_env_vars))
        pytest.skip(f"{accepted_keys} is required.")

    pytest.importorskip(preset.dependency_module)

    agent = AgentLoop(
        model=build_model(config.model_copy(update={"max_steps": 2})),
        tools=get_default_tools(),
        memory=Memory(),
        max_steps=2,
    )

    result = agent.run("Reply with exactly: ok")

    assert "ok" in result.lower()
