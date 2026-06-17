from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentloop.intervention import InterventionRequest
from agentloop.session_store import (
    InterventionStatus,
    RunStatus,
    SQLiteSessionStore,
)


def test_sqlite_session_store_persists_messages_across_instances(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.sqlite3"
    store = SQLiteSessionStore(db_path)
    session = store.create_session(title="workspace triage")
    run = store.create_run(session.session_id, user_input="calculate")
    store.append_message(
        session.session_id,
        HumanMessage(content="What is 6 * 7?"),
        run_id=run.run_id,
    )
    store.append_message(
        session.session_id,
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "calculator",
                    "args": {"expression": "6 * 7"},
                }
            ],
        ),
        run_id=run.run_id,
    )
    store.append_message(
        session.session_id,
        ToolMessage(
            content="42",
            tool_call_id="call_1",
            additional_kwargs={"status": "success", "risk_level": "low"},
        ),
        run_id=run.run_id,
    )
    store.close()

    reopened = SQLiteSessionStore(db_path)
    messages = reopened.get_messages(session.session_id)

    assert messages[0] == HumanMessage(content="What is 6 * 7?")
    assert isinstance(messages[1], AIMessage)
    assert messages[1].tool_calls == [
        {
            "id": "call_1",
            "name": "calculator",
            "args": {"expression": "6 * 7"},
            "type": "tool_call",
        }
    ]
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].content == "42"
    assert messages[2].tool_call_id == "call_1"
    assert messages[2].additional_kwargs["status"] == "success"
    assert reopened.get_session(session.session_id).message_count == 3


def test_sqlite_session_store_records_run_lifecycle(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3")
    session = store.create_session()

    run = store.create_run(session.session_id, user_input="inspect project")
    store.finish_run(
        run.run_id,
        status=RunStatus.INTERRUPTED,
        stop_reason="high_risk_action",
    )

    runs = store.list_runs(session.session_id)

    assert len(runs) == 1
    assert runs[0].run_id == run.run_id
    assert runs[0].session_id == session.session_id
    assert runs[0].status is RunStatus.INTERRUPTED
    assert runs[0].user_input == "inspect project"
    assert runs[0].stop_reason == "high_risk_action"
    assert runs[0].ended_at is not None


def test_sqlite_session_store_records_tool_call_and_result(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3")
    session = store.create_session()
    run = store.create_run(session.session_id)

    store.record_tool_call(
        run.run_id,
        tool_call_id="call_search",
        tool_name="search_text",
        arguments={"query": "SessionStore", "path": "."},
        risk_level="low",
        requires_confirmation=False,
        read_only=True,
    )
    store.record_tool_result(
        tool_call_id="call_search",
        status="success",
        content="src/agentloop/session_store.py",
        duration_ms=12.5,
        metadata={"result_count": 1},
    )

    tool_calls = store.list_tool_calls(run.run_id)
    tool_result = store.get_tool_result("call_search")

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_call_id == "call_search"
    assert tool_calls[0].tool_name == "search_text"
    assert tool_calls[0].arguments == {"query": "SessionStore", "path": "."}
    assert tool_calls[0].risk_level == "low"
    assert tool_calls[0].requires_confirmation is False
    assert tool_calls[0].read_only is True
    assert tool_result is not None
    assert tool_result.status == "success"
    assert tool_result.content == "src/agentloop/session_store.py"
    assert tool_result.duration_ms == 12.5
    assert tool_result.metadata == {"result_count": 1}


def test_sqlite_session_store_rejects_run_links_from_another_session(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3")
    first_session = store.create_session()
    second_session = store.create_session()
    run = store.create_run(first_session.session_id)

    with pytest.raises(ValueError, match="does not belong"):
        store.append_message(
            second_session.session_id,
            HumanMessage(content="wrong session"),
            run_id=run.run_id,
        )

    with pytest.raises(ValueError, match="does not belong"):
        store.record_intervention(
            session_id=second_session.session_id,
            run_id=run.run_id,
            request=InterventionRequest.for_high_risk_tool("risky_write"),
        )


def test_sqlite_session_store_recovers_pending_intervention_and_records_decision(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.sqlite3"
    store = SQLiteSessionStore(db_path)
    session = store.create_session()
    run = store.create_run(session.session_id)
    request = InterventionRequest.for_high_risk_tool("risky_write")
    intervention = store.record_intervention(
        session_id=session.session_id,
        run_id=run.run_id,
        request=request,
    )
    store.close()

    reopened = SQLiteSessionStore(db_path)
    pending = reopened.get_pending_intervention(request.resume_token)

    assert pending is not None
    assert pending.intervention_id == intervention.intervention_id
    assert pending.status is InterventionStatus.PENDING
    assert pending.reason == "high_risk_action"
    assert pending.tool_name == "risky_write"
    assert pending.resume_token == request.resume_token

    decision = reopened.record_decision(
        intervention.intervention_id,
        decision="confirm",
        user_note="Approved for this run.",
    )

    assert decision.intervention_id == intervention.intervention_id
    assert decision.session_id == session.session_id
    assert decision.run_id == run.run_id
    assert decision.decision == "confirm"
    assert reopened.get_pending_intervention(request.resume_token) is None
    resolved = reopened.get_intervention(intervention.intervention_id)
    assert resolved is not None
    assert resolved.status is InterventionStatus.RESOLVED
