from pathlib import Path

from agentloop.trace_store import (
    SQLiteTraceStore,
    TraceEventType,
    TraceRunStatus,
)


def test_sqlite_trace_store_persists_run_lifecycle(tmp_path: Path) -> None:
    db_path = tmp_path / "trace.sqlite3"
    store = SQLiteTraceStore(db_path)

    run = store.start_run(
        session_id="session_1",
        user_input="inspect project",
        metadata={"provider": "deepseek"},
    )
    store.finish_run(
        run.run_id,
        status=TraceRunStatus.COMPLETED,
        stop_reason="final_answer",
    )
    store.close()

    reopened = SQLiteTraceStore(db_path)
    persisted = reopened.get_run(run.run_id)

    assert persisted.run_id == run.run_id
    assert persisted.session_id == "session_1"
    assert persisted.status is TraceRunStatus.COMPLETED
    assert persisted.user_input == "inspect project"
    assert persisted.stop_reason == "final_answer"
    assert persisted.started_at is not None
    assert persisted.ended_at is not None
    assert persisted.metadata == {"provider": "deepseek"}


def test_sqlite_trace_store_records_redacted_ordered_events(
    tmp_path: Path,
) -> None:
    store = SQLiteTraceStore(tmp_path / "trace.sqlite3")
    run = store.start_run(session_id="session_1", user_input="hello")

    store.record_event(
        run.run_id,
        TraceEventType.MODEL_CALL,
        payload={"prompt": "Authorization: Bearer live-token"},
    )
    store.record_event(
        run.run_id,
        TraceEventType.TOOL_RESULT,
        payload={"content": "MODEL_API_KEY=sk-1234567890abcdef"},
    )
    store.record_event(
        run.run_id,
        TraceEventType.ERROR,
        payload={"message": "tool failed"},
        error_source="tool_runtime",
    )

    events = store.list_events(run.run_id)

    assert [event.event_type for event in events] == [
        TraceEventType.MODEL_CALL,
        TraceEventType.TOOL_RESULT,
        TraceEventType.ERROR,
    ]
    assert events[0].payload == {"prompt": "Authorization: Bearer [REDACTED_SECRET]"}
    assert events[1].payload == {"content": "MODEL_API_KEY=[REDACTED_SECRET]"}
    assert events[2].error_source == "tool_runtime"
