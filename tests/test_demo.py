from mini_agent.agent import MiniAgent
from mini_agent.demo import DemoChatModel
from mini_agent.memory import Memory
from mini_agent.tools import get_default_tools


def test_demo_model_runs_arithmetic_tool_loop() -> None:
    agent = MiniAgent(
        model=DemoChatModel(),
        tools=get_default_tools(),
        memory=Memory(),
        max_steps=4,
    )

    result = agent.run("What is 2 + 3 * 4?")

    assert result == "The answer is 14."


def test_demo_model_returns_direct_answer_without_expression() -> None:
    agent = MiniAgent(
        model=DemoChatModel(),
        tools=get_default_tools(),
        memory=Memory(),
        max_steps=4,
    )

    result = agent.run("hello")

    assert result == "Demo mode can calculate simple arithmetic expressions."
