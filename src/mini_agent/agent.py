"""Minimal LangGraph agent loop."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph, add_messages

from mini_agent.memory import Memory
from mini_agent.tool_runtime import ToolRuntime

MAX_STEPS_MESSAGE = "Agent stopped because it reached the maximum number of steps."


class AgentState(TypedDict):
    """State carried through the LangGraph workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    llm_calls: int


class MiniAgent:
    """Runs a LangGraph model/tool loop until a final answer is produced."""

    def __init__(
        self,
        model: Any,
        tools: list[BaseTool] | ToolRuntime,
        memory: Memory,
        max_steps: int,
    ) -> None:
        self.model = model
        self.tool_runtime = (
            tools if isinstance(tools, ToolRuntime) else ToolRuntime.from_tools(tools)
        )
        self.tools = self.tool_runtime.get_tools()
        self.model_with_tools = model.bind_tools(self.tools)
        self.memory = memory
        self.max_steps = max_steps
        self.graph = self._build_graph()

    def run(self, user_input: str) -> str:
        """Run the agent for a single user input."""

        self.memory.add_user_message(user_input)
        state = self.graph.invoke(
            {"messages": self.memory.get_messages(), "llm_calls": 0}
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
        workflow.add_node("max_steps", self._max_steps)
        workflow.add_edge(START, "llm")
        workflow.add_conditional_edges(
            "llm",
            self._route_after_model,
            {"tools": "tools", "max_steps": "max_steps", END: END},
        )
        workflow.add_edge("tools", "llm")
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
        for tool_call in getattr(last_message, "tool_calls", []):
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            result = self.tool_runtime.call_tool(tool_name, tool_args)
            tool_messages.append(
                ToolMessage(
                    content=result.to_message_content(),
                    tool_call_id=tool_call["id"],
                    additional_kwargs=result.to_message_kwargs(),
                )
            )
        return {"messages": tool_messages}

    def _max_steps(self, state: AgentState) -> dict:
        """Return a final message when the LLM call budget is exhausted."""

        return {"messages": [AIMessage(content=MAX_STEPS_MESSAGE)]}

    def _route_after_model(self, state: AgentState) -> str:
        """Route to tools, max-step stop, or end after a model call."""

        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls:
            return END
        if state.get("llm_calls", 0) >= self.max_steps:
            return "max_steps"
        return "tools"
