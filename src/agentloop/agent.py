"""Minimal LangGraph agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph, add_messages

from agentloop.context_manager import (
    ContextComposition,
    ContextManager,
    format_long_term_memory_context,
)
from agentloop.intervention import (
    DecisionRecord,
    InterventionAction,
    InterventionRequest,
)
from agentloop.memory import Memory
from agentloop.memory_store import MemoryNamespace
from agentloop.tool_runtime import ToolResultStatus, ToolRuntime
from agentloop.trace_store import TraceEventType, TraceRunStatus

MAX_STEPS_MESSAGE = "Agent stopped because it reached the maximum number of steps."


class AgentState(TypedDict):
    """State carried through the LangGraph workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    llm_calls: int
    tool_failure_counts: NotRequired[dict[str, int]]
    intervention_request: NotRequired[InterventionRequest | None]


@dataclass(frozen=True)
class PendingIntervention:
    """Runtime context needed to resume a stopped agent run."""

    request: InterventionRequest
    messages: list[BaseMessage]
    llm_calls: int
    tool_failure_counts: dict[str, int]
    tool_call: dict[str, Any] | None = None


class AgentLoop:
    """Runs a LangGraph model/tool loop until a final answer is produced."""

    def __init__(
        self,
        model: Any,
        tools: list[BaseTool] | ToolRuntime,
        memory: Memory,
        max_steps: int,
        repeated_failure_threshold: int = 3,
        trace_store: Any | None = None,
        session_id: str | None = None,
        memory_store: Any | None = None,
        context_manager: ContextManager | None = None,
        memory_namespaces: tuple[MemoryNamespace | str, ...] = (
            MemoryNamespace.USER,
            MemoryNamespace.PROJECT,
            MemoryNamespace.DECISION,
        ),
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
        self.trace_store = trace_store
        self.session_id = session_id
        self.memory_store = memory_store
        self.context_manager = context_manager or ContextManager()
        self.memory_namespaces = memory_namespaces
        self._active_trace_run_id: str | None = None
        self._pending_intervention: PendingIntervention | None = None
        self._last_context_composition: ContextComposition | None = None
        self.graph = self._build_graph()

    def run(self, user_input: str) -> str:
        """Run the agent for a single user input."""

        if self._pending_intervention is not None:
            raise RuntimeError(
                "A pending intervention must be resolved before starting "
                "another run."
            )
        self._start_trace_run(user_input)
        self._inject_long_term_memories_if_needed()
        self.memory.add_user_message(user_input)
        try:
            state = self.graph.invoke(
                {
                    "messages": self.memory.get_messages(),
                    "llm_calls": 0,
                    "tool_failure_counts": {},
                    "intervention_request": None,
                }
            )
        except Exception as exc:
            self._record_trace_event(
                TraceEventType.ERROR,
                {"message": str(exc), "error_type": type(exc).__name__},
                error_source="agent_loop",
            )
            self._finish_trace_run(TraceRunStatus.FAILED, "error")
            self._active_trace_run_id = None
            raise
        messages = state["messages"]
        self.memory.set_messages(messages)

        last_message = messages[-1]
        stop_reason = last_message.additional_kwargs.get("stop_reason", "final_answer")
        if self._pending_intervention is not None:
            self._finish_trace_run(TraceRunStatus.INTERRUPTED, stop_reason)
        else:
            self._finish_trace_run(TraceRunStatus.COMPLETED, stop_reason)
            self._active_trace_run_id = None
        return str(last_message.text)

    def get_pending_intervention(self) -> PendingIntervention | None:
        """Return the pending intervention context, if the run is paused."""

        return self._pending_intervention

    def get_context_report(self) -> ContextComposition | None:
        """Return the most recent prepared model context report."""

        return self._last_context_composition

    def resolve_intervention(self, decision: DecisionRecord) -> str:
        """Apply a user decision and resume or stop the pending run."""

        pending = self._require_pending_intervention(decision)
        self._pending_intervention = None

        if decision.action in (InterventionAction.REJECT, InterventionAction.STOP):
            self._record_trace_event(
                TraceEventType.DECISION,
                decision.to_message_kwargs(),
            )
            result = self._stop_from_decision(pending, decision)
            self._finish_trace_run(
                TraceRunStatus.INTERRUPTED,
                pending.request.reason.value,
            )
            self._active_trace_run_id = None
            return result

        self._record_trace_event(TraceEventType.DECISION, decision.to_message_kwargs())
        if decision.action is InterventionAction.CONTINUE:
            result = self._continue_from_messages(
                pending.messages,
                pending.llm_calls,
                {},
            )
            self._finish_trace_after_resume()
            return result

        if decision.action is InterventionAction.SKIP_TOOL:
            result = self._resume_with_skipped_tool(pending, decision)
            self._finish_trace_after_resume()
            return result

        if decision.action in (InterventionAction.APPROVE, InterventionAction.EDIT):
            result = self._resume_with_tool_execution(pending, decision)
            self._finish_trace_after_resume()
            return result

        raise ValueError(f"Unsupported intervention action: {decision.action.value}.")

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

        managed_context = self.context_manager.prepare(state["messages"])
        self._last_context_composition = managed_context.composition
        if any(
            component.kind.value == "summary"
            for component in managed_context.composition.components
        ):
            self._record_trace_event(
                TraceEventType.CONTEXT_COMPRESSION,
                {
                    "estimated_tokens": managed_context.composition.estimated_tokens,
                    "input_token_budget": (
                        managed_context.composition.input_token_budget
                    ),
                    "components": [
                        {
                            "kind": component.kind.value,
                            "token_count": component.token_count,
                            "description": component.description,
                            "source_message_indexes": list(
                                component.source_message_indexes
                            ),
                            "memory_ids": list(component.memory_ids),
                        }
                        for component in managed_context.composition.components
                    ],
                },
            )
        response = self.model_with_tools.invoke(managed_context.messages)
        self._record_trace_event(
            TraceEventType.MODEL_CALL,
            {
                "message_count": len(managed_context.messages),
                "raw_message_count": len(state["messages"]),
                "context_estimated_tokens": (
                    managed_context.composition.estimated_tokens
                ),
                "context_input_budget": managed_context.composition.input_token_budget,
                "llm_call_index": state.get("llm_calls", 0) + 1,
                "tool_call_count": len(getattr(response, "tool_calls", [])),
            },
        )
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
        messages_before_tools = list(state["messages"])
        for tool_call in getattr(last_message, "tool_calls", []):
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            self._record_trace_event(
                TraceEventType.TOOL_CALL,
                {
                    "tool_call_id": tool_call["id"],
                    "tool_name": tool_name,
                    "arguments": tool_args,
                },
            )
            result = self.tool_runtime.call_tool(tool_name, tool_args)
            tool_message = ToolMessage(
                content=result.to_message_content(),
                tool_call_id=tool_call["id"],
                additional_kwargs=result.to_message_kwargs(),
            )
            self._record_tool_result_event(tool_call["id"], result)
            new_intervention = self._record_tool_result(
                tool_name=tool_name,
                result_status=result.status,
                failure_counts=failure_counts,
            )
            if new_intervention is not None and intervention_request is None:
                intervention_request = new_intervention
                self._record_trace_event(
                    TraceEventType.INTERVENTION_REQUEST,
                    new_intervention.to_message_kwargs(),
                )
            tool_messages.append(tool_message)
            if intervention_request is not None and self._pending_intervention is None:
                resume_messages = messages_before_tools
                if result.status is not ToolResultStatus.REQUIRES_CONFIRMATION:
                    resume_messages = [*messages_before_tools, *tool_messages]
                self._pending_intervention = PendingIntervention(
                    request=intervention_request,
                    messages=resume_messages,
                    llm_calls=state.get("llm_calls", 0),
                    tool_failure_counts=dict(failure_counts),
                    tool_call=dict(tool_call),
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

    def _require_pending_intervention(
        self,
        decision: DecisionRecord,
    ) -> PendingIntervention:
        pending = self._pending_intervention
        if pending is None:
            raise RuntimeError("There is no pending intervention to resolve.")
        if decision.resume_token != pending.request.resume_token:
            raise ValueError("Decision resume_token does not match pending request.")
        if decision.action.value not in pending.request.options:
            raise ValueError(
                f"Action '{decision.action.value}' is not allowed for "
                f"intervention reason '{pending.request.reason.value}'."
            )
        return pending

    def _continue_from_messages(
        self,
        messages: list[BaseMessage],
        llm_calls: int,
        failure_counts: dict[str, int],
    ) -> str:
        state = self.graph.invoke(
            {
                "messages": messages,
                "llm_calls": llm_calls,
                "tool_failure_counts": failure_counts,
                "intervention_request": None,
            }
        )
        self.memory.set_messages(state["messages"])
        return str(state["messages"][-1].content)

    def _resume_with_tool_execution(
        self,
        pending: PendingIntervention,
        decision: DecisionRecord,
    ) -> str:
        if pending.tool_call is None:
            raise ValueError("No tool call is available for this intervention.")
        tool_args = pending.tool_call.get("args", {})
        if decision.action is InterventionAction.EDIT:
            if decision.edited_tool_args is None:
                raise ValueError("edited_tool_args is required for edit decisions.")
            tool_args = decision.edited_tool_args
        tool_name = pending.tool_call["name"]
        result = self.tool_runtime.call_tool(tool_name, tool_args, approved=True)
        self._record_trace_event(
            TraceEventType.TOOL_CALL,
            {
                "tool_call_id": pending.tool_call["id"],
                "tool_name": tool_name,
                "arguments": tool_args,
                "approved": True,
                "decision_id": decision.decision_id,
            },
        )
        self._record_tool_result_event(pending.tool_call["id"], result)
        tool_message = ToolMessage(
            content=result.to_message_content(),
            tool_call_id=pending.tool_call["id"],
            additional_kwargs={
                **result.to_message_kwargs(),
                "decision_record": decision.to_message_kwargs(),
            },
        )
        return self._continue_from_messages(
            [*pending.messages, tool_message],
            pending.llm_calls,
            pending.tool_failure_counts,
        )

    def _resume_with_skipped_tool(
        self,
        pending: PendingIntervention,
        decision: DecisionRecord,
    ) -> str:
        if pending.tool_call is None:
            raise ValueError("No tool call is available for this intervention.")
        tool_name = pending.tool_call["name"]
        self._record_trace_event(
            TraceEventType.TOOL_RESULT,
            {
                "tool_call_id": pending.tool_call["id"],
                "tool_name": tool_name,
                "status": ToolResultStatus.REJECTED.value,
                "error_code": "tool_skipped",
                "decision_id": decision.decision_id,
            },
        )
        tool_message = ToolMessage(
            content=f"Tool '{tool_name}' was skipped by user decision.",
            tool_call_id=pending.tool_call["id"],
            additional_kwargs={
                "status": ToolResultStatus.REJECTED.value,
                "tool_name": tool_name,
                "error_code": "tool_skipped",
                "decision_record": decision.to_message_kwargs(),
            },
        )
        return self._continue_from_messages(
            [*pending.messages, tool_message],
            pending.llm_calls,
            {},
        )

    def _stop_from_decision(
        self,
        pending: PendingIntervention,
        decision: DecisionRecord,
    ) -> str:
        message = AIMessage(
            content=(
                "Intervention resolved: "
                f"user selected '{decision.action.value}'."
            ),
            additional_kwargs={
                "stop_reason": pending.request.reason.value,
                "decision_record": decision.to_message_kwargs(),
            },
        )
        self.memory.set_messages([*pending.messages, message])
        return str(message.content)

    def _inject_long_term_memories_if_needed(self) -> None:
        if self.memory_store is None:
            return
        if self.memory.get_messages():
            return
        memories = self.memory_store.search_memories(
            "",
            namespaces=self.memory_namespaces,
        )
        if not memories:
            return
        self.memory.set_messages(
            [
                SystemMessage(
                    content=format_long_term_memory_context(memories),
                    additional_kwargs={
                        "context_component": "long_term_memory",
                        "memory_ids": [memory.memory_id for memory in memories],
                    },
                )
            ]
        )

    def _start_trace_run(self, user_input: str) -> None:
        if self.trace_store is None:
            return
        run = self.trace_store.start_run(
            session_id=self.session_id,
            user_input=user_input,
        )
        self._active_trace_run_id = run.run_id

    def _finish_trace_run(
        self,
        status: TraceRunStatus,
        stop_reason: str | None,
    ) -> None:
        if self.trace_store is None or self._active_trace_run_id is None:
            return
        self.trace_store.finish_run(
            self._active_trace_run_id,
            status=status,
            stop_reason=stop_reason,
        )

    def _finish_trace_after_resume(self) -> None:
        if self._pending_intervention is not None:
            stop_reason = self._pending_intervention.request.reason.value
            self._finish_trace_run(TraceRunStatus.INTERRUPTED, stop_reason)
            return
        self._finish_trace_run(TraceRunStatus.COMPLETED, "final_answer")
        self._active_trace_run_id = None

    def _record_trace_event(
        self,
        event_type: TraceEventType,
        payload: dict[str, Any],
        error_source: str | None = None,
    ) -> None:
        if self.trace_store is None or self._active_trace_run_id is None:
            return
        self.trace_store.record_event(
            self._active_trace_run_id,
            event_type,
            payload=payload,
            error_source=error_source,
        )

    def _record_tool_result_event(self, tool_call_id: str, result) -> None:
        self._record_trace_event(
            TraceEventType.TOOL_RESULT,
            {
                "tool_call_id": tool_call_id,
                "tool_name": result.tool_name,
                "status": result.status.value,
                "content": result.content,
                "error_code": result.error_code,
                "duration_ms": result.duration_ms,
                "risk_level": result.metadata.risk_level.value,
            },
        )
        if result.status is ToolResultStatus.ERROR:
            self._record_trace_event(
                TraceEventType.ERROR,
                {
                    "tool_call_id": tool_call_id,
                    "tool_name": result.tool_name,
                    "error_code": result.error_code,
                    "error_type": result.error_type,
                },
                error_source="tool_runtime",
            )
