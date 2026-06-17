"""SQLite-backed long-term memory store."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentloop.safety import redact_secrets


class MemorySafetyError(RuntimeError):
    """Raised when a memory write would persist unsafe content."""


class MemoryNamespace(StrEnum):
    """Supported long-term memory namespaces."""

    USER = "user"
    PROJECT = "project"
    DECISION = "decision"


class MemoryStatus(StrEnum):
    """Lifecycle state for long-term memory records."""

    ACTIVE = "active"
    DISABLED = "disabled"
    DELETED = "deleted"


@dataclass(frozen=True)
class MemoryRecord:
    """A user-confirmed long-term memory."""

    memory_id: str
    namespace: MemoryNamespace
    status: MemoryStatus
    content: str
    source_session_id: str
    source_run_id: str
    confirmed_by_user: bool
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


class SQLiteMemoryStore:
    """Persist user-confirmed long-term memories to SQLite."""

    def __init__(self, db_path: str | Path, timeout_seconds: float = 5.0) -> None:
        self.db_path = Path(db_path)
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=timeout_seconds)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._enable_wal_if_supported()
        self._migrate()

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self._conn.close()

    def create_memory(
        self,
        namespace: MemoryNamespace | str,
        content: str,
        source_session_id: str,
        source_run_id: str,
        confirmed_by_user: bool,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Create a new active memory after confirmation and safety checks."""

        if not confirmed_by_user:
            raise ValueError("Memory writes must be confirmed by the user.")
        normalized_namespace = MemoryNamespace(namespace)
        normalized_metadata = metadata or {}
        _ensure_safe_memory_payload(content, normalized_metadata)
        memory_id = uuid4().hex
        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO memories (
                    memory_id, namespace, status, content, source_session_id,
                    source_run_id, confirmed_by_user, metadata_json, created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    normalized_namespace.value,
                    MemoryStatus.ACTIVE.value,
                    content,
                    source_session_id,
                    source_run_id,
                    _bool_to_int(confirmed_by_user),
                    _json_dumps(normalized_metadata),
                    now,
                    now,
                ),
            )
        return self.get_memory(memory_id)

    def get_memory(self, memory_id: str) -> MemoryRecord:
        """Return one memory by id."""

        row = self._conn.execute(
            _MEMORY_SELECT + " WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Memory '{memory_id}' does not exist.")
        return _memory_from_row(row)

    def list_memories(
        self,
        namespace: MemoryNamespace | str | None = None,
        status: MemoryStatus | str | None = None,
    ) -> list[MemoryRecord]:
        """List memories with optional namespace and status filters."""

        clauses: list[str] = []
        params: list[str] = []
        if namespace is not None:
            clauses.append("namespace = ?")
            params.append(MemoryNamespace(namespace).value)
        if status is not None:
            clauses.append("status = ?")
            params.append(MemoryStatus(status).value)

        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            _MEMORY_SELECT + where_sql + " ORDER BY created_at ASC, memory_id ASC",
            params,
        ).fetchall()
        return [_memory_from_row(row) for row in rows]

    def search_memories(
        self,
        query: str,
        namespaces: tuple[MemoryNamespace | str, ...] | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        """Search active confirmed memories by simple case-insensitive content match."""

        query_key = query.strip().casefold()
        namespace_values = (
            {MemoryNamespace(namespace).value for namespace in namespaces}
            if namespaces is not None
            else None
        )
        candidates = self.list_memories(status=MemoryStatus.ACTIVE)
        results: list[MemoryRecord] = []
        for memory in candidates:
            if not memory.confirmed_by_user:
                continue
            if (
                namespace_values is not None
                and memory.namespace.value not in namespace_values
            ):
                continue
            if query_key and query_key not in memory.content.casefold():
                continue
            results.append(memory)
            if len(results) >= limit:
                break
        return results

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Update memory content or metadata after safety checks."""

        current = self.get_memory(memory_id)
        next_content = content if content is not None else current.content
        next_metadata = metadata if metadata is not None else current.metadata
        _ensure_safe_memory_payload(next_content, next_metadata)
        with self._conn:
            self._conn.execute(
                """
                UPDATE memories
                SET content = ?, metadata_json = ?, updated_at = ?
                WHERE memory_id = ?
                """,
                (
                    next_content,
                    _json_dumps(next_metadata),
                    _now(),
                    memory_id,
                ),
            )
        return self.get_memory(memory_id)

    def disable_memory(self, memory_id: str) -> MemoryRecord:
        """Disable a memory without deleting its audit record."""

        return self._set_status(memory_id, MemoryStatus.DISABLED)

    def delete_memory(self, memory_id: str) -> MemoryRecord:
        """Soft-delete a memory so future retrieval no longer uses it."""

        return self._set_status(memory_id, MemoryStatus.DELETED)

    def _set_status(
        self,
        memory_id: str,
        status: MemoryStatus,
    ) -> MemoryRecord:
        self.get_memory(memory_id)
        with self._conn:
            self._conn.execute(
                """
                UPDATE memories
                SET status = ?, updated_at = ?
                WHERE memory_id = ?
                """,
                (status.value, _now(), memory_id),
            )
        return self.get_memory(memory_id)

    def _migrate(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    status TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    source_run_id TEXT NOT NULL,
                    confirmed_by_user INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_namespace_status
                    ON memories(namespace, status);

                PRAGMA user_version = 1;
                """
            )

    def _enable_wal_if_supported(self) -> None:
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass


_MEMORY_SELECT = """
    SELECT
        memory_id, namespace, status, content, source_session_id, source_run_id,
        confirmed_by_user, metadata_json, created_at, updated_at
    FROM memories
"""


def _ensure_safe_memory_payload(content: str, metadata: dict[str, Any]) -> None:
    serialized_metadata = _json_dumps(metadata)
    if redact_secrets(content) != content or (
        redact_secrets(serialized_metadata) != serialized_metadata
    ):
        raise MemorySafetyError("Memory content appears to contain a secret.")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str) -> Any:
    return json.loads(value)


def _bool_to_int(value: bool) -> int:
    return 1 if value else 0


def _memory_from_row(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        memory_id=row["memory_id"],
        namespace=MemoryNamespace(row["namespace"]),
        status=MemoryStatus(row["status"]),
        content=row["content"],
        source_session_id=row["source_session_id"],
        source_run_id=row["source_run_id"],
        confirmed_by_user=bool(row["confirmed_by_user"]),
        metadata=_json_loads(row["metadata_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
