"""Context budgeting, trimming, and composition reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from math import ceil
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from agentloop.memory_store import MemoryRecord


class ContextComponentKind(StrEnum):
    """Renderable parts that make up one model context."""

    LONG_TERM_MEMORY = "long_term_memory"
    SUMMARY = "summary"
    ORIGINAL = "original"


@dataclass(frozen=True)
class ContextBudget:
    """Token budget used before each model call."""

    max_tokens: int = 8000
    reserved_output_tokens: int = 1000
    summary_max_tokens: int = 800

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be greater than 0.")
        if self.reserved_output_tokens < 0:
            raise ValueError(
                "reserved_output_tokens must be greater than or equal to 0."
            )
        if self.reserved_output_tokens >= self.max_tokens:
            raise ValueError("reserved_output_tokens must be less than max_tokens.")
        if self.summary_max_tokens < 1:
            raise ValueError("summary_max_tokens must be greater than 0.")

    @property
    def input_token_budget(self) -> int:
        """Return the budget available for input context."""

        return self.max_tokens - self.reserved_output_tokens


@dataclass(frozen=True)
class ContextComponent:
    """One visible contribution to a prepared model context."""

    kind: ContextComponentKind
    token_count: int
    description: str
    source_message_indexes: tuple[int, ...] = ()
    memory_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextComposition:
    """Explains what a prepared context contains."""

    input_token_budget: int
    estimated_tokens: int
    original_message_count: int
    prepared_message_count: int
    components: tuple[ContextComponent, ...]

    def to_display_lines(self) -> list[str]:
        """Return compact user-facing context composition lines."""

        lines = [
            "context: "
            f"estimated_tokens={self.estimated_tokens} "
            f"input_budget={self.input_token_budget} "
            f"messages={self.prepared_message_count}/{self.original_message_count}"
        ]
        for component in self.components:
            detail = (
                f"context: {component.kind.value} "
                f"tokens={component.token_count} "
                f"{component.description}"
            )
            if component.source_message_indexes:
                detail += (
                    " source_messages="
                    f"{_format_index_span(component.source_message_indexes)}"
                )
            if component.memory_ids:
                detail += f" memory_ids={','.join(component.memory_ids)}"
            lines.append(detail)
        return lines


@dataclass(frozen=True)
class ManagedContext:
    """Prepared messages plus their composition report."""

    messages: list[BaseMessage]
    composition: ContextComposition


@dataclass(frozen=True)
class _MessageBlock:
    messages: tuple[BaseMessage, ...]
    source_message_indexes: tuple[int, ...]
    valid_original: bool = True


class ContextManager:
    """Prepare bounded model contexts while preserving message structure."""

    def __init__(self, budget: ContextBudget | None = None) -> None:
        self.budget = budget or ContextBudget()

    def prepare(
        self,
        messages: list[BaseMessage],
        long_term_memories: list[MemoryRecord] | None = None,
    ) -> ManagedContext:
        """Return a budgeted model context and a report of its components."""

        source_messages = list(messages)
        prefix_messages, body_messages, body_offset = _split_system_prefix(
            source_messages
        )
        memory_message = _build_long_term_memory_message(long_term_memories or [])
        if memory_message is not None:
            prefix_messages.append(memory_message)

        blocks = _build_message_blocks(body_messages, body_offset)
        prefix_tokens = estimate_messages_tokens(prefix_messages)
        retained_blocks = _select_recent_blocks(
            blocks,
            prefix_tokens=prefix_tokens,
            input_token_budget=self.budget.input_token_budget,
        )
        retained_ids = {id(block) for block in retained_blocks}
        dropped_blocks = [
            block
            for block in blocks
            if id(block) not in retained_ids or not block.valid_original
        ]
        retained_messages = [
            message
            for block in retained_blocks
            if block.valid_original
            for message in block.messages
        ]

        summary_message = _build_summary_message(
            dropped_blocks,
            max_tokens=min(
                self.budget.summary_max_tokens,
                max(
                    0,
                    self.budget.input_token_budget
                    - prefix_tokens
                    - estimate_messages_tokens(retained_messages),
                ),
            ),
        )
        prepared_messages = [*prefix_messages]
        if summary_message is not None:
            prepared_messages.append(summary_message)
        prepared_messages.extend(retained_messages)

        prepared_messages = _drop_orphan_tool_messages(prepared_messages)
        components = _build_components(
            prepared_messages=prepared_messages,
            original_message_count=len(source_messages),
            retained_blocks=retained_blocks,
            dropped_blocks=dropped_blocks,
        )
        composition = ContextComposition(
            input_token_budget=self.budget.input_token_budget,
            estimated_tokens=estimate_messages_tokens(prepared_messages),
            original_message_count=len(source_messages),
            prepared_message_count=len(prepared_messages),
            components=tuple(components),
        )
        return ManagedContext(messages=prepared_messages, composition=composition)


def format_long_term_memory_context(memories: list[MemoryRecord]) -> str:
    """Format confirmed long-term memories for model context."""

    lines = ["Long-term memory (user-confirmed):"]
    for memory in memories:
        lines.append(
            "- "
            f"[{memory.namespace.value}] {memory.content} "
            f"(source_session_id={memory.source_session_id}, "
            f"source_run_id={memory.source_run_id})"
        )
    return "\n".join(lines)


def estimate_messages_tokens(messages: list[BaseMessage]) -> int:
    """Estimate tokens without adding tokenizer dependencies."""

    return sum(_estimate_message_tokens(message) for message in messages)


def _split_system_prefix(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], list[BaseMessage], int]:
    prefix: list[BaseMessage] = []
    index = 0
    while index < len(messages) and isinstance(messages[index], SystemMessage):
        prefix.append(messages[index])
        index += 1
    return prefix, messages[index:], index


def _build_message_blocks(
    messages: list[BaseMessage],
    offset: int,
) -> list[_MessageBlock]:
    blocks: list[_MessageBlock] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, ToolMessage):
            blocks.append(
                _MessageBlock(
                    messages=(message,),
                    source_message_indexes=(offset + index,),
                    valid_original=False,
                )
            )
            index += 1
            continue

        tool_calls = getattr(message, "tool_calls", [])
        if isinstance(message, AIMessage) and tool_calls:
            block_messages: list[BaseMessage] = [message]
            block_indexes = [offset + index]
            expected_ids = {str(tool_call["id"]) for tool_call in tool_calls}
            seen_ids: set[str] = set()
            index += 1
            while index < len(messages):
                possible_tool_message = messages[index]
                if not isinstance(possible_tool_message, ToolMessage):
                    break
                tool_message = possible_tool_message
                tool_call_id = str(tool_message.tool_call_id)
                if tool_call_id not in expected_ids:
                    break
                block_messages.append(tool_message)
                block_indexes.append(offset + index)
                seen_ids.add(tool_call_id)
                index += 1
                if seen_ids == expected_ids:
                    break
            blocks.append(
                _MessageBlock(
                    messages=tuple(block_messages),
                    source_message_indexes=tuple(block_indexes),
                    valid_original=seen_ids == expected_ids,
                )
            )
            continue

        blocks.append(
            _MessageBlock(
                messages=(message,),
                source_message_indexes=(offset + index,),
            )
        )
        index += 1
    return blocks


def _select_recent_blocks(
    blocks: list[_MessageBlock],
    prefix_tokens: int,
    input_token_budget: int,
) -> list[_MessageBlock]:
    retained_reversed: list[_MessageBlock] = []
    retained_tokens = 0
    for block in reversed(blocks):
        if not block.valid_original:
            continue
        block_tokens = estimate_messages_tokens(list(block.messages))
        if prefix_tokens + retained_tokens + block_tokens <= input_token_budget:
            retained_reversed.append(block)
            retained_tokens += block_tokens
    return list(reversed(retained_reversed))


def _build_long_term_memory_message(
    memories: list[MemoryRecord],
) -> SystemMessage | None:
    if not memories:
        return None
    return SystemMessage(
        content=format_long_term_memory_context(memories),
        additional_kwargs={
            "context_component": ContextComponentKind.LONG_TERM_MEMORY.value,
            "memory_ids": [memory.memory_id for memory in memories],
        },
    )


def _build_summary_message(
    dropped_blocks: list[_MessageBlock],
    max_tokens: int,
) -> SystemMessage | None:
    if not dropped_blocks or max_tokens < 1:
        return None

    lines = ["Running summary of earlier context:"]
    source_indexes: list[int] = []
    for block in dropped_blocks:
        source_indexes.extend(block.source_message_indexes)
        role = _message_role(block.messages[0])
        joined = " / ".join(_message_excerpt(message) for message in block.messages)
        lines.append(f"- messages {_format_index_span(block.source_message_indexes)} ")
        lines[-1] += f"{role}: {joined}"

    content = _fit_summary_content("\n".join(lines), max_tokens)
    return SystemMessage(
        content=content,
        additional_kwargs={
            "context_component": ContextComponentKind.SUMMARY.value,
            "source_message_indexes": source_indexes,
        },
    )


def _fit_summary_content(content: str, max_tokens: int) -> str:
    token_budget_chars = max_tokens * 4
    if len(content) <= token_budget_chars:
        return content
    suffix = "\n- earlier context summary truncated to fit budget"
    return content[: max(0, token_budget_chars - len(suffix))].rstrip() + suffix


def _drop_orphan_tool_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    valid_tool_ids: set[str] = set()
    filtered: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, AIMessage):
            valid_tool_ids.update(
                str(tool_call["id"]) for tool_call in message.tool_calls
            )
            filtered.append(message)
            continue
        if isinstance(message, ToolMessage):
            tool_call_id = str(message.tool_call_id)
            if tool_call_id in valid_tool_ids:
                filtered.append(message)
            continue
        filtered.append(message)
    return filtered


def _build_components(
    prepared_messages: list[BaseMessage],
    original_message_count: int,
    retained_blocks: list[_MessageBlock],
    dropped_blocks: list[_MessageBlock],
) -> list[ContextComponent]:
    components: list[ContextComponent] = []
    memory_messages = [
        message for message in prepared_messages if _component_kind(message)
        is ContextComponentKind.LONG_TERM_MEMORY
    ]
    for message in memory_messages:
        components.append(
            ContextComponent(
                kind=ContextComponentKind.LONG_TERM_MEMORY,
                token_count=estimate_messages_tokens([message]),
                description="user-confirmed long-term memories",
                memory_ids=tuple(message.additional_kwargs.get("memory_ids", [])),
            )
        )

    summary_messages = [
        message
        for message in prepared_messages
        if _component_kind(message) is ContextComponentKind.SUMMARY
    ]
    for message in summary_messages:
        source_indexes = tuple(
            int(index)
            for index in message.additional_kwargs.get("source_message_indexes", [])
        )
        components.append(
            ContextComponent(
                kind=ContextComponentKind.SUMMARY,
                token_count=estimate_messages_tokens([message]),
                description="running summary of trimmed earlier context",
                source_message_indexes=source_indexes,
            )
        )

    retained_indexes = tuple(
        index
        for block in retained_blocks
        if block.valid_original
        for index in block.source_message_indexes
    )
    if retained_indexes or original_message_count == len(prepared_messages):
        components.append(
            ContextComponent(
                kind=ContextComponentKind.ORIGINAL,
                token_count=estimate_messages_tokens(
                    [
                        message
                        for message in prepared_messages
                        if _component_kind(message) is None
                    ]
                ),
                description="retained original messages",
                source_message_indexes=retained_indexes,
            )
        )
    elif dropped_blocks:
        components.append(
            ContextComponent(
                kind=ContextComponentKind.ORIGINAL,
                token_count=0,
                description="no original body messages retained",
            )
        )
    return components


def _component_kind(message: BaseMessage) -> ContextComponentKind | None:
    raw_kind = message.additional_kwargs.get("context_component")
    if raw_kind is not None:
        return ContextComponentKind(raw_kind)
    content = str(message.content)
    if content.startswith("Long-term memory (user-confirmed):"):
        return ContextComponentKind.LONG_TERM_MEMORY
    if content.startswith("Running summary of earlier context:"):
        return ContextComponentKind.SUMMARY
    return None


def _message_role(message: BaseMessage) -> str:
    return str(getattr(message, "type", "message"))


def _message_excerpt(message: BaseMessage, max_chars: int = 120) -> str:
    details = str(message.content).replace("\n", " ")
    tool_calls = getattr(message, "tool_calls", [])
    if tool_calls:
        details = f"tool_calls={_json_dumps(tool_calls)} {details}".strip()
    if isinstance(message, ToolMessage):
        details = f"tool_call_id={message.tool_call_id} {details}".strip()
    if len(details) <= max_chars:
        return details
    return details[: max_chars - 3].rstrip() + "..."


def _estimate_message_tokens(message: BaseMessage) -> int:
    payload: dict[str, Any] = {
        "type": getattr(message, "type", "message"),
        "content": message.content,
        "additional_kwargs": message.additional_kwargs,
    }
    tool_calls = getattr(message, "tool_calls", [])
    if tool_calls:
        payload["tool_calls"] = tool_calls
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
    return 4 + _estimate_text_tokens(_json_dumps(payload))


def _estimate_text_tokens(text: str) -> int:
    return max(1, ceil(len(text) / 4))


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _format_index_span(indexes: tuple[int, ...]) -> str:
    if not indexes:
        return "none"
    if len(indexes) == 1:
        return str(indexes[0])
    sorted_indexes = sorted(indexes)
    return f"{sorted_indexes[0]}-{sorted_indexes[-1]}"
