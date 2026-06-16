"""Minimal LangGraph agent loop."""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph, add_messages

from agentloop.intervention import InterventionRequest
from agentloop.memory import Memory
from agentloop.tool_runtime import ToolResultStatus, ToolRuntime

MAX_STEPS_MESSAGE = "Agent stopped because it reached the maximum number of steps."


class AgentState(TypedDict):
    """State carried through the LangGraph workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    llm_calls: int
    tool_failure_counts: NotRequired[dict[str, int]]
    intervention_request: NotRequired[InterventionRequest | None]


class AgentLoop:
    """Runs a LangGraph model/tool loop until a final answer is produced."""

    def __init__(
        self,
        model: Any,
        tools: list[BaseTool] | ToolRuntime,
        memory: Memory,
        max_steps: int,
        repeated_failure_threshold: int = 3,
    ) -> None:
        if repeated_failure_threshold < 1:
            raise ValueError("repeated_failure_threshold must be greater than 0.")
        self.model = model
        self.tool_runtime = (
            tools if isinstance(tools, ToolRuntime) else ToolRuntime.from_tools(tools)
        )
        self.tools = self.tool_runtime.get_tools()
        self.model_with_tools = model.bind_tools(self.tools)
        self.memory = memory
        self.max_steps = max_steps
        self.repeated_failure_threshold = repeated_failure_threshold
        self.graph = self._build_graph()

    def run(self, user_input: str) -> str:
        """Run the agent for a single user input."""

        self.memory.add_user_message(user_input)
        state = self.graph.invoke(
            {
                "messages": self.memory.get_messages(),
                "llm_calls": 0,
                "tool_failure_counts": {},
                "intervention_request": None,
            }
        )
        messages = state["messages"]
        self.memory.set_messages(messages)

        last_message = messages[-1]
        return str(last_message.content)

    def _build_graph(self):
        """Build and compile the LangGraph workflow."""

        workflow = StateGraph(AgentState)
        workflow.add_node("llm", self._call_model)
        workflow.add_node("tools", self._call_tools)
        workflow.add_node("intervention", self._intervention)
        workflow.add_node("max_steps", self._max_steps)
        workflow.add_edge(START, "llm")
        workflow.add_conditional_edges(
            "llm",
            self._route_after_model,
            {"tools": "tools", "max_steps": "max_steps", END: END},
        )
        workflow.add_conditional_edges(
            "tools",
            self._route_after_tools,
            {"intervention": "intervention", "llm": "llm"},
        )
        workflow.add_edge("intervention", END)
        workflow.add_edge("max_steps", END)
        return workflow.compile()

    def _call_model(self, state: AgentState) -> dict:
        """Call the chat model with the current messages."""

        response = self.model_with_tools.invoke(state["messages"])
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def _call_tools(self, state: AgentState) -> dict:
        """Execute requested tools and return ToolMessage results."""

        last_message = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        failure_counts = dict(state.get("tool_failure_counts") or {})
        intervention_request = None
        for tool_call in getattr(last_message, "tool_calls", []):
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            result = self.tool_runtime.call_tool(tool_name, tool_args)
            intervention_request = intervention_request or self._record_tool_result(
                tool_name=tool_name,
                result_status=result.status,
                failure_counts=failure_counts,
            )
            tool_messages.append(
                ToolMessage(
                    content=result.to_message_content(),
                    tool_call_id=tool_call["id"],
                    additional_kwargs=result.to_message_kwargs(),
                )
            )
        return {
            "messages": tool_messages,
            "tool_failure_counts": failure_counts,
            "intervention_request": intervention_request,
        }

    def _record_tool_result(
        self,
        tool_name: str,
        result_status: ToolResultStatus,
        failure_counts: dict[str, int],
    ) -> InterventionRequest | None:
        """Update tool failure state and return an intervention when needed."""

        if result_status is ToolResultStatus.SUCCESS:
            failure_counts.pop(tool_name, None)
            return None

        if result_status is ToolResultStatus.REQUIRES_CONFIRMATION:
            return InterventionRequest.for_high_risk_tool(tool_name)

        failure_counts[tool_name] = failure_counts.get(tool_name, 0) + 1
        if failure_counts[tool_name] >= self.repeated_failure_threshold:
            return InterventionRequest.for_repeated_failure(
                tool_name=tool_name,
                failure_count=failure_counts[tool_name],
            )
        return None

    def _intervention(self, state: AgentState) -> dict:
        """Return a final message when the runtime needs user judgment."""

        request = state.get("intervention_request")
        if request is None:
            return {}
        return {
            "messages": [
                AIMessage(
                    content=request.message,
                    additional_kwargs={
                        "stop_reason": request.reason.value,
                        "intervention_request": request.to_message_kwargs(),
                    },
                )
            ]
        }

    def _max_steps(self, state: AgentState) -> dict:
        """Return a final message when the LLM call budget is exhausted."""

        return {
            "messages": [
                AIMessage(
                    content=MAX_STEPS_MESSAGE,
                    additional_kwargs={"stop_reason": "step_limit_risk"},
                )
            ]
        }

    def _route_after_model(self, state: AgentState) -> str:
        """Route to tools, max-step stop, or end after a model call."""

        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls:
            return END
        if state.get("llm_calls", 0) >= self.max_steps:
            return "max_steps"
        return "tools"

    def _route_after_tools(self, state: AgentState) -> str:
        """Route to intervention or continue after tool execution."""

        if state.get("intervention_request") is not None:
            return "intervention"
        return "llm"
