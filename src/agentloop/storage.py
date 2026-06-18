"""Default local storage locations for the CLI/TUI runtime."""

from __future__ import annotations

import os
from pathlib import Path


def get_data_dir() -> Path:
    """Return the local runtime data directory."""

    return Path(os.getenv("AGENTLOOP_DATA_DIR", ".agentloop")).expanduser()


def get_session_db_path() -> Path:
    """Return the default SQLite session database path."""

    return _path_from_env("AGENTLOOP_SESSION_DB", get_data_dir() / "sessions.sqlite3")


def get_memory_db_path() -> Path:
    """Return the default SQLite memory database path."""

    return _path_from_env("AGENTLOOP_MEMORY_DB", get_data_dir() / "memory.sqlite3")


def get_trace_db_path() -> Path:
    """Return the default SQLite trace database path."""

    return _path_from_env("AGENTLOOP_TRACE_DB", get_data_dir() / "trace.sqlite3")


def _path_from_env(env_name: str, default: Path) -> Path:
    raw_value = os.getenv(env_name)
    if raw_value is None or not raw_value.strip():
        return default
    return Path(raw_value).expanduser()
