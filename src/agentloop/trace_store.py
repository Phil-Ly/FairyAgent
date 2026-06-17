"""SQLite-backed trace and audit log persistence."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentloop.safety import SafetyPolicy, redact_secrets


class TraceRunStatus(StrEnum):
    """Lifecycle states for an audited agent run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class TraceEventType(StrEnum):
    """Audit event categories emitted by the runtime."""

    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    INTERVENTION_REQUEST = "intervention_request"
    DECISION = "decision"
    CONTEXT_COMPRESSION = "context_compression"
    ERROR = "error"


@dataclass(frozen=True)
class TraceRunRecord:
    """A persisted audited run."""

    run_id: str
    session_id: str | None
    status: TraceRunStatus
    user_input: str | None
    stop_reason: str | None
    started_at: str
    ended_at: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TraceEventRecord:
    """A persisted event within an audited run."""

    event_id: int
    run_id: str
    event_type: TraceEventType
    payload: dict[str, Any]
    created_at: str
    error_source: str | None


class SQLiteTraceStore:
    """Persist run lifecycle and ordered audit events to SQLite."""

    def __init__(
        self,
        db_path: str | Path,
        safety_policy: SafetyPolicy | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.db_path = Path(db_path)
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._safety_policy = safety_policy or SafetyPolicy()
        self._conn = sqlite3.connect(str(db_path), timeout=timeout_seconds)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._enable_wal_if_supported()
        self._migrate()

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self._conn.close()

    def start_run(
        self,
        session_id: str | None = None,
        user_input: str | None = None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> TraceRunRecord:
        """Create a running trace run."""

        trace_run_id = run_id or uuid4().hex
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO trace_runs (
                    run_id, session_id, status, user_input, started_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_run_id,
                    session_id,
                    TraceRunStatus.RUNNING.value,
                    _sanitize_value(user_input),
                    now,
                    _json_dumps(_sanitize_value(metadata or {})),
                ),
            )
        return self.get_run(trace_run_id)

    def finish_run(
        self,
        run_id: str,
        status: TraceRunStatus,
        stop_reason: str | None = None,
    ) -> TraceRunRecord:
        """Mark an audited run as complete, failed, or interrupted."""

        if status is TraceRunStatus.RUNNING:
            raise ValueError("finish_run cannot set status back to running.")
        self.get_run(run_id)
        with self._conn:
            self._conn.execute(
                """
                UPDATE trace_runs
                SET status = ?, stop_reason = ?, ended_at = ?
                WHERE run_id = ?
                """,
                (status.value, stop_reason, _now(), run_id),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> TraceRunRecord:
        """Return one trace run."""

        row = self._conn.execute(
            """
            SELECT
                run_id, session_id, status, user_input, stop_reason,
                started_at, ended_at, metadata_json
            FROM trace_runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Trace run '{run_id}' does not exist.")
        return _run_from_row(row)

    def list_runs(self, session_id: str | None = None) -> list[TraceRunRecord]:
        """Return trace runs, optionally filtered by session."""

        if session_id is None:
            rows = self._conn.execute(
                """
                SELECT
                    run_id, session_id, status, user_input, stop_reason,
                    started_at, ended_at, metadata_json
                FROM trace_runs
                ORDER BY started_at ASC, run_id ASC
                """
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT
                    run_id, session_id, status, user_input, stop_reason,
                    started_at, ended_at, metadata_json
                FROM trace_runs
                WHERE session_id = ?
                ORDER BY started_at ASC, run_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [_run_from_row(row) for row in rows]

    def record_event(
        self,
        run_id: str,
        event_type: TraceEventType,
        payload: dict[str, Any] | None = None,
        error_source: str | None = None,
    ) -> TraceEventRecord:
        """Append an audit event to a trace run."""

        self.get_run(run_id)
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO trace_events (
                    run_id, event_type, payload_json, created_at, error_source
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    event_type.value,
                    _json_dumps(_sanitize_value(payload or {})),
                    _now(),
                    error_source,
                ),
            )
        return self._get_event(_lastrowid(cursor))

    def list_events(self, run_id: str) -> list[TraceEventRecord]:
        """Return ordered audit events for one run."""

        rows = self._conn.execute(
            """
            SELECT
                event_id, run_id, event_type, payload_json, created_at,
                error_source
            FROM trace_events
            WHERE run_id = ?
            ORDER BY event_id ASC
            """,
            (run_id,),
        ).fetchall()
        return [_event_from_row(row) for row in rows]

    def _migrate(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS trace_runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    status TEXT NOT NULL,
                    user_input TEXT,
                    stop_reason TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_trace_runs_session_id
                    ON trace_runs(session_id);

                CREATE TABLE IF NOT EXISTS trace_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES trace_runs(run_id)
                        ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    error_source TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_trace_events_run_id
                    ON trace_events(run_id, event_id);

                PRAGMA user_version = 1;
                """
            )

    def _enable_wal_if_supported(self) -> None:
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass

    def _get_event(self, event_id: int) -> TraceEventRecord:
        row = self._conn.execute(
            """
            SELECT
                event_id, run_id, event_type, payload_json, created_at,
                error_source
            FROM trace_events
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Trace event '{event_id}' does not exist.")
        return _event_from_row(row)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {key: _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_value(item) for item in value]
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str) -> Any:
    return json.loads(value)


def _lastrowid(cursor: sqlite3.Cursor) -> int:
    if cursor.lastrowid is None:
        raise RuntimeError("SQLite did not return a row id for the insert.")
    return int(cursor.lastrowid)


def _run_from_row(row: sqlite3.Row) -> TraceRunRecord:
    return TraceRunRecord(
        run_id=row["run_id"],
        session_id=row["session_id"],
        status=TraceRunStatus(row["status"]),
        user_input=row["user_input"],
        stop_reason=row["stop_reason"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        metadata=_json_loads(row["metadata_json"]),
    )


def _event_from_row(row: sqlite3.Row) -> TraceEventRecord:
    return TraceEventRecord(
        event_id=int(row["event_id"]),
        run_id=row["run_id"],
        event_type=TraceEventType(row["event_type"]),
        payload=_json_loads(row["payload_json"]),
        created_at=row["created_at"],
        error_source=row["error_source"],
    )
