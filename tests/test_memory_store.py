from pathlib import Path

import pytest

from agentloop.memory_store import (
    MemoryNamespace,
    MemorySafetyError,
    MemoryStatus,
    SQLiteMemoryStore,
)


def test_sqlite_memory_store_creates_and_persists_confirmed_memory(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "memory.sqlite3"
    store = SQLiteMemoryStore(db_path)

    memory = store.create_memory(
        namespace=MemoryNamespace.PROJECT,
        content="The project uses SQLite for MVP persistence.",
        source_session_id="session_1",
        source_run_id="run_1",
        confirmed_by_user=True,
        metadata={"topic": "storage"},
    )
    store.close()

    reopened = SQLiteMemoryStore(db_path)
    persisted = reopened.get_memory(memory.memory_id)

    assert persisted.namespace is MemoryNamespace.PROJECT
    assert persisted.status is MemoryStatus.ACTIVE
    assert persisted.content == "The project uses SQLite for MVP persistence."
    assert persisted.source_session_id == "session_1"
    assert persisted.source_run_id == "run_1"
    assert persisted.confirmed_by_user is True
    assert persisted.metadata == {"topic": "storage"}


def test_sqlite_memory_store_requires_user_confirmation(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")

    with pytest.raises(ValueError, match="confirmed"):
        store.create_memory(
            namespace=MemoryNamespace.USER,
            content="User prefers concise answers.",
            source_session_id="session_1",
            source_run_id="run_1",
            confirmed_by_user=False,
        )


def test_sqlite_memory_store_rejects_secret_content(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")

    with pytest.raises(MemorySafetyError, match="secret"):
        store.create_memory(
            namespace=MemoryNamespace.USER,
            content="MODEL_API_KEY=sk-1234567890abcdef",
            source_session_id="session_1",
            source_run_id="run_1",
            confirmed_by_user=True,
        )


def test_sqlite_memory_store_updates_disables_and_deletes_memory(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    memory = store.create_memory(
        namespace=MemoryNamespace.DECISION,
        content="Use SQLite first.",
        source_session_id="session_1",
        source_run_id="run_1",
        confirmed_by_user=True,
    )

    updated = store.update_memory(memory.memory_id, content="Use SQLite for MVP.")
    disabled = store.disable_memory(memory.memory_id)
    deleted = store.delete_memory(memory.memory_id)

    assert updated.content == "Use SQLite for MVP."
    assert disabled.status is MemoryStatus.DISABLED
    assert deleted.status is MemoryStatus.DELETED
    assert store.search_memories("SQLite") == []


def test_sqlite_memory_store_searches_active_confirmed_memories_by_namespace(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    user_memory = store.create_memory(
        namespace=MemoryNamespace.USER,
        content="User prefers Chinese responses.",
        source_session_id="session_1",
        source_run_id="run_1",
        confirmed_by_user=True,
    )
    project_memory = store.create_memory(
        namespace=MemoryNamespace.PROJECT,
        content="Project uses LangGraph.",
        source_session_id="session_1",
        source_run_id="run_2",
        confirmed_by_user=True,
    )

    user_results = store.search_memories(
        "prefers",
        namespaces=(MemoryNamespace.USER,),
    )
    all_results = store.search_memories("Project")

    assert [memory.memory_id for memory in user_results] == [user_memory.memory_id]
    assert [memory.memory_id for memory in all_results] == [project_memory.memory_id]
