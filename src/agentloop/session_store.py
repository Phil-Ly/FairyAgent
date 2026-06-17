"""SQLite-backed session persistence for agent runtime state."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from agentloop.intervention import InterventionRequest


class SessionStatus(StrEnum):
    """Lifecycle states for a persisted session."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class RunStatus(StrEnum):
    """Lifecycle states for one agent run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class InterventionStatus(StrEnum):
    """Lifecycle states for a human intervention request."""

    PENDING = "pending"
    RESOLVED = "resolved"


@dataclass(frozen=True)
class SessionRecord:
    """A persisted conversation session."""

    session_id: str
    title: str | None
    status: SessionStatus
    created_at: str
    updated_at: str
    message_count: int


@dataclass(frozen=True)
class RunRecord:
    """A persisted agent run within a session."""

    run_id: str
    session_id: str
    status: RunStatus
    user_input: str | None
    stop_reason: str | None
    started_at: str
    ended_at: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ToolCallRecord:
    """A persisted tool call request emitted by the model."""

    tool_call_id: str
    run_id: str
    tool_name: str
    arguments: dict[str, Any]
    risk_level: str | None
    requires_confirmation: bool
    read_only: bool
    created_at: str


@dataclass(frozen=True)
class ToolResultRecord:
    """A persisted tool result produced by ToolRuntime."""

    result_id: int
    tool_call_id: str
    status: str
    content: str
    duration_ms: float | None
    error_code: str | None
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class InterventionRecord:
    """A persisted human-in-the-loop request."""

    intervention_id: str
    session_id: str
    run_id: str | None
    reason: str
    message: str
    tool_name: str | None
    failure_count: int | None
    recommended_option: str
    options: tuple[str, ...]
    resume_token: str
    status: InterventionStatus
    created_at: str
    resolved_at: str | None


@dataclass(frozen=True)
class DecisionRecord:
    """A persisted user decision resolving an intervention."""

    decision_id: str
    intervention_id: str
    session_id: str
    run_id: str | None
    decision: str
    user_note: str | None
    metadata: dict[str, Any]
    created_at: str


class SQLiteSessionStore:
    """Persist sessions, runs, messages, tools, interventions, and decisions."""

    def __init__(self, db_path: str | Path, timeout_seconds: float = 5.0) -> None:
        self.db_path = Path(db_path)
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(db_path),
            timeout=timeout_seconds,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._enable_wal_if_supported()
        self._migrate()

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self._conn.close()

    def create_session(self, title: str | None = None) -> SessionRecord:
        """Create and return a new active session."""

        session_id = uuid4().hex
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO sessions (
                    session_id, title, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, title, SessionStatus.ACTIVE.value, now, now),
            )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> SessionRecord:
        """Return one session, raising KeyError when it does not exist."""

        row = self._conn.execute(
            """
            SELECT
                sessions.session_id,
                sessions.title,
                sessions.status,
                sessions.created_at,
                sessions.updated_at,
                COUNT(messages.message_id) AS message_count
            FROM sessions
            LEFT JOIN messages ON messages.session_id = sessions.session_id
            WHERE sessions.session_id = ?
            GROUP BY sessions.session_id
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Session '{session_id}' does not exist.")
        return _session_from_row(row)

    def list_sessions(self) -> list[SessionRecord]:
        """Return sessions ordered by latest update."""

        rows = self._conn.execute(
            """
            SELECT
                sessions.session_id,
                sessions.title,
                sessions.status,
                sessions.created_at,
                sessions.updated_at,
                COUNT(messages.message_id) AS message_count
            FROM sessions
            LEFT JOIN messages ON messages.session_id = sessions.session_id
            GROUP BY sessions.session_id
            ORDER BY sessions.updated_at DESC, sessions.created_at DESC
            """
        ).fetchall()
        return [_session_from_row(row) for row in rows]

    def archive_session(self, session_id: str) -> SessionRecord:
        """Mark a session archived without deleting its history."""

        self.get_session(session_id)
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                UPDATE sessions
                SET status = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (SessionStatus.ARCHIVED.value, now, session_id),
            )
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> None:
        """Delete a session and its dependent runtime records."""

        self.get_session(session_id)
        with self._conn:
            self._conn.execute(
                """
                DELETE FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            )

    def create_run(
        self,
        session_id: str,
        user_input: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunRecord:
        """Create a running agent run for a session."""

        self.get_session(session_id)
        run_id = uuid4().hex
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO runs (
                    run_id, session_id, status, user_input, started_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    session_id,
                    RunStatus.RUNNING.value,
                    user_input,
                    now,
                    _json_dumps(metadata or {}),
                ),
            )
            self._touch_session(session_id, now)
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        """Return one run, raising KeyError when it does not exist."""

        row = self._conn.execute(
            """
            SELECT
                run_id, session_id, status, user_input, stop_reason,
                started_at, ended_at, metadata_json
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Run '{run_id}' does not exist.")
        return _run_from_row(row)

    def list_runs(self, session_id: str) -> list[RunRecord]:
        """Return runs for one session in creation order."""

        rows = self._conn.execute(
            """
            SELECT
                run_id, session_id, status, user_input, stop_reason,
                started_at, ended_at, metadata_json
            FROM runs
            WHERE session_id = ?
            ORDER BY started_at ASC, run_id ASC
            """,
            (session_id,),
        ).fetchall()
        return [_run_from_row(row) for row in rows]

    def finish_run(
        self,
        run_id: str,
        status: RunStatus,
        stop_reason: str | None = None,
    ) -> RunRecord:
        """Mark a run complete, failed, or interrupted."""

        if status is RunStatus.RUNNING:
            raise ValueError("finish_run cannot set status back to running.")
        run = self.get_run(run_id)
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                UPDATE runs
                SET status = ?, stop_reason = ?, ended_at = ?
                WHERE run_id = ?
                """,
                (status.value, stop_reason, now, run_id),
            )
            self._touch_session(run.session_id, now)
        return self.get_run(run_id)

    def append_message(
        self,
        session_id: str,
        message: BaseMessage,
        run_id: str | None = None,
    ) -> int:
        """Append one LangChain message to a session and return its row id."""

        self.get_session(session_id)
        if run_id is not None:
            self._require_run_in_session(session_id, run_id)
        payload = message_to_dict(message)
        now = _now()
        with self._conn:
            sequence = self._next_message_sequence(session_id)
            cursor = self._conn.execute(
                """
                INSERT INTO messages (
                    session_id, run_id, sequence, role, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    run_id,
                    sequence,
                    payload["type"],
                    _json_dumps(payload),
                    now,
                ),
            )
            self._touch_session(session_id, now)
        return _lastrowid(cursor)

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        """Return persisted LangChain messages for a session in order."""

        self.get_session(session_id)
        rows = self._conn.execute(
            """
            SELECT payload_json
            FROM messages
            WHERE session_id = ?
            ORDER BY sequence ASC
            """,
            (session_id,),
        ).fetchall()
        payloads = [_json_loads(row["payload_json"]) for row in rows]
        return messages_from_dict(payloads)

    def record_tool_call(
        self,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        risk_level: str | None = None,
        requires_confirmation: bool = False,
        read_only: bool = True,
    ) -> ToolCallRecord:
        """Persist a model-requested tool call for audit and replay."""

        self.get_run(run_id)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO tool_calls (
                    tool_call_id, run_id, tool_name, arguments_json,
                    risk_level, requires_confirmation, read_only, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tool_call_id,
                    run_id,
                    tool_name,
                    _json_dumps(arguments),
                    risk_level,
                    _bool_to_int(requires_confirmation),
                    _bool_to_int(read_only),
                    _now(),
                ),
            )
        return self.get_tool_call(tool_call_id)

    def get_tool_call(self, tool_call_id: str) -> ToolCallRecord:
        """Return one tool call, raising KeyError when it does not exist."""

        row = self._conn.execute(
            """
            SELECT
                tool_call_id, run_id, tool_name, arguments_json, risk_level,
                requires_confirmation, read_only, created_at
            FROM tool_calls
            WHERE tool_call_id = ?
            """,
            (tool_call_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Tool call '{tool_call_id}' does not exist.")
        return _tool_call_from_row(row)

    def list_tool_calls(self, run_id: str) -> list[ToolCallRecord]:
        """Return tool calls for one run in creation order."""

        rows = self._conn.execute(
            """
            SELECT
                tool_call_id, run_id, tool_name, arguments_json, risk_level,
                requires_confirmation, read_only, created_at
            FROM tool_calls
            WHERE run_id = ?
            ORDER BY created_at ASC, tool_call_id ASC
            """,
            (run_id,),
        ).fetchall()
        return [_tool_call_from_row(row) for row in rows]

    def record_tool_result(
        self,
        tool_call_id: str,
        status: str,
        content: str,
        duration_ms: float | None = None,
        error_code: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResultRecord:
        """Persist the result for one tool call."""

        self.get_tool_call(tool_call_id)
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO tool_results (
                    tool_call_id, status, content, duration_ms, error_code,
                    metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tool_call_id,
                    status,
                    content,
                    duration_ms,
                    error_code,
                    _json_dumps(metadata or {}),
                    _now(),
                ),
            )
        return self._get_tool_result_by_id(_lastrowid(cursor))

    def get_tool_result(self, tool_call_id: str) -> ToolResultRecord | None:
        """Return the result for a tool call, if one exists."""

        row = self._conn.execute(
            """
            SELECT
                result_id, tool_call_id, status, content, duration_ms,
                error_code, metadata_json, created_at
            FROM tool_results
            WHERE tool_call_id = ?
            """,
            (tool_call_id,),
        ).fetchone()
        if row is None:
            return None
        return _tool_result_from_row(row)

    def record_intervention(
        self,
        session_id: str,
        run_id: str | None,
        request: InterventionRequest,
    ) -> InterventionRecord:
        """Persist a pending human intervention request."""

        self.get_session(session_id)
        if run_id is not None:
            self._require_run_in_session(session_id, run_id)
        intervention_id = uuid4().hex
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO interventions (
                    intervention_id, session_id, run_id, reason, message,
                    tool_name, failure_count, recommended_option, options_json,
                    resume_token, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    intervention_id,
                    session_id,
                    run_id,
                    request.reason.value,
                    request.message,
                    request.tool_name,
                    request.failure_count,
                    request.recommended_option,
                    _json_dumps(list(request.options)),
                    request.resume_token,
                    InterventionStatus.PENDING.value,
                    now,
                ),
            )
            self._touch_session(session_id, now)
        intervention = self.get_intervention(intervention_id)
        if intervention is None:
            raise KeyError(f"Intervention '{intervention_id}' does not exist.")
        return intervention

    def get_intervention(self, intervention_id: str) -> InterventionRecord | None:
        """Return one intervention, if it exists."""

        row = self._conn.execute(
            _INTERVENTION_SELECT + " WHERE intervention_id = ?",
            (intervention_id,),
        ).fetchone()
        if row is None:
            return None
        return _intervention_from_row(row)

    def get_pending_intervention(
        self,
        resume_token: str,
    ) -> InterventionRecord | None:
        """Return a pending intervention by resume token, if it exists."""

        row = self._conn.execute(
            _INTERVENTION_SELECT
            + " WHERE resume_token = ? AND status = ? ORDER BY created_at DESC",
            (resume_token, InterventionStatus.PENDING.value),
        ).fetchone()
        if row is None:
            return None
        return _intervention_from_row(row)

    def record_decision(
        self,
        intervention_id: str,
        decision: str,
        user_note: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DecisionRecord:
        """Persist a user decision and mark the intervention resolved."""

        intervention = self.get_intervention(intervention_id)
        if intervention is None:
            raise KeyError(f"Intervention '{intervention_id}' does not exist.")
        decision_id = uuid4().hex
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO decisions (
                    decision_id, intervention_id, session_id, run_id, decision,
                    user_note, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    intervention_id,
                    intervention.session_id,
                    intervention.run_id,
                    decision,
                    user_note,
                    _json_dumps(metadata or {}),
                    now,
                ),
            )
            self._conn.execute(
                """
                UPDATE interventions
                SET status = ?, resolved_at = ?
                WHERE intervention_id = ?
                """,
                (
                    InterventionStatus.RESOLVED.value,
                    now,
                    intervention_id,
                ),
            )
            self._touch_session(intervention.session_id, now)
        return self.get_decision(decision_id)

    def get_decision(self, decision_id: str) -> DecisionRecord:
        """Return one decision, raising KeyError when it does not exist."""

        row = self._conn.execute(
            """
            SELECT
                decision_id, intervention_id, session_id, run_id, decision,
                user_note, metadata_json, created_at
            FROM decisions
            WHERE decision_id = ?
            """,
            (decision_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Decision '{decision_id}' does not exist.")
        return _decision_from_row(row)

    def _migrate(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id)
                        ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    user_input TEXT,
                    stop_reason TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_runs_session_id
                    ON runs(session_id);

                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id)
                        ON DELETE CASCADE,
                    run_id TEXT REFERENCES runs(run_id) ON DELETE SET NULL,
                    sequence INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session_sequence
                    ON messages(session_id, sequence);

                CREATE TABLE IF NOT EXISTS tool_calls (
                    tool_call_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    risk_level TEXT,
                    requires_confirmation INTEGER NOT NULL,
                    read_only INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tool_calls_run_id
                    ON tool_calls(run_id);

                CREATE TABLE IF NOT EXISTS tool_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_call_id TEXT NOT NULL UNIQUE
                        REFERENCES tool_calls(tool_call_id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    content TEXT NOT NULL,
                    duration_ms REAL,
                    error_code TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS interventions (
                    intervention_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id)
                        ON DELETE CASCADE,
                    run_id TEXT REFERENCES runs(run_id) ON DELETE SET NULL,
                    reason TEXT NOT NULL,
                    message TEXT NOT NULL,
                    tool_name TEXT,
                    failure_count INTEGER,
                    recommended_option TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    resume_token TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_interventions_resume_status
                    ON interventions(resume_token, status);

                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    intervention_id TEXT NOT NULL
                        REFERENCES interventions(intervention_id) ON DELETE CASCADE,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id)
                        ON DELETE CASCADE,
                    run_id TEXT REFERENCES runs(run_id) ON DELETE SET NULL,
                    decision TEXT NOT NULL,
                    user_note TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_decisions_intervention_id
                    ON decisions(intervention_id);

                PRAGMA user_version = 1;
                """
            )

    def _enable_wal_if_supported(self) -> None:
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass

    def _touch_session(self, session_id: str, updated_at: str) -> None:
        self._conn.execute(
            """
            UPDATE sessions
            SET updated_at = ?
            WHERE session_id = ?
            """,
            (updated_at, session_id),
        )

    def _next_message_sequence(self, session_id: str) -> int:
        row = self._conn.execute(
            """
            SELECT COALESCE(MAX(sequence), -1) + 1 AS next_sequence
            FROM messages
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        return int(row["next_sequence"])

    def _require_run_in_session(self, session_id: str, run_id: str) -> RunRecord:
        run = self.get_run(run_id)
        if run.session_id != session_id:
            raise ValueError(
                f"Run '{run_id}' does not belong to session '{session_id}'."
            )
        return run

    def _get_tool_result_by_id(self, result_id: int) -> ToolResultRecord:
        row = self._conn.execute(
            """
            SELECT
                result_id, tool_call_id, status, content, duration_ms,
                error_code, metadata_json, created_at
            FROM tool_results
            WHERE result_id = ?
            """,
            (result_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Tool result '{result_id}' does not exist.")
        return _tool_result_from_row(row)


_INTERVENTION_SELECT = """
    SELECT
        intervention_id, session_id, run_id, reason, message, tool_name,
        failure_count, recommended_option, options_json, resume_token, status,
        created_at, resolved_at
    FROM interventions
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str) -> Any:
    return json.loads(value)


def _bool_to_int(value: bool) -> int:
    return 1 if value else 0


def _lastrowid(cursor: sqlite3.Cursor) -> int:
    if cursor.lastrowid is None:
        raise RuntimeError("SQLite did not return a row id for the insert.")
    return int(cursor.lastrowid)


def _session_from_row(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        session_id=row["session_id"],
        title=row["title"],
        status=SessionStatus(row["status"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        message_count=int(row["message_count"]),
    )


def _run_from_row(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        session_id=row["session_id"],
        status=RunStatus(row["status"]),
        user_input=row["user_input"],
        stop_reason=row["stop_reason"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        metadata=_json_loads(row["metadata_json"]),
    )


def _tool_call_from_row(row: sqlite3.Row) -> ToolCallRecord:
    return ToolCallRecord(
        tool_call_id=row["tool_call_id"],
        run_id=row["run_id"],
        tool_name=row["tool_name"],
        arguments=_json_loads(row["arguments_json"]),
        risk_level=row["risk_level"],
        requires_confirmation=bool(row["requires_confirmation"]),
        read_only=bool(row["read_only"]),
        created_at=row["created_at"],
    )


def _tool_result_from_row(row: sqlite3.Row) -> ToolResultRecord:
    return ToolResultRecord(
        result_id=int(row["result_id"]),
        tool_call_id=row["tool_call_id"],
        status=row["status"],
        content=row["content"],
        duration_ms=row["duration_ms"],
        error_code=row["error_code"],
        metadata=_json_loads(row["metadata_json"]),
        created_at=row["created_at"],
    )


def _intervention_from_row(row: sqlite3.Row) -> InterventionRecord:
    return InterventionRecord(
        intervention_id=row["intervention_id"],
        session_id=row["session_id"],
        run_id=row["run_id"],
        reason=row["reason"],
        message=row["message"],
        tool_name=row["tool_name"],
        failure_count=row["failure_count"],
        recommended_option=row["recommended_option"],
        options=tuple(_json_loads(row["options_json"])),
        resume_token=row["resume_token"],
        status=InterventionStatus(row["status"]),
        created_at=row["created_at"],
        resolved_at=row["resolved_at"],
    )


def _decision_from_row(row: sqlite3.Row) -> DecisionRecord:
    return DecisionRecord(
        decision_id=row["decision_id"],
        intervention_id=row["intervention_id"],
        session_id=row["session_id"],
        run_id=row["run_id"],
        decision=row["decision"],
        user_note=row["user_note"],
        metadata=_json_loads(row["metadata_json"]),
        created_at=row["created_at"],
    )
