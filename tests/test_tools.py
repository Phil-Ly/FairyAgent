import pytest

import agentloop.tools as tools
from agentloop.tools import calculator, echo, get_default_tools, list_files


def test_calculator_evaluates_simple_expression() -> None:
    result = calculator.invoke({"expression": "2 + 3 * 4"})

    assert result == "14"


def test_calculator_rejects_dangerous_expression() -> None:
    with pytest.raises(ValueError, match="Unsupported expression"):
        calculator.invoke({"expression": "__import__('os').system('rm -rf /')"})


def test_echo_returns_text() -> None:
    result = echo.invoke({"text": "same text"})

    assert result == "same text"


def test_get_default_tools_returns_low_risk_read_only_tools() -> None:
    default_tools = get_default_tools()

    assert [tool.name for tool in default_tools] == [
        "calculator",
        "echo",
        "list_files",
        "read_file",
        "search_text",
        "project_tree",
    ]


def test_list_files_rejects_path_traversal(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="outside the workspace"):
        list_files.invoke({"path": ".."})


def test_read_file_reads_workspace_text_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.md").write_text("hello\nworld\n", encoding="utf-8")

    result = tools.read_file.invoke({"path": "notes.md"})

    assert result == "hello\nworld\n"


def test_read_file_rejects_path_traversal(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="outside the workspace"):
        tools.read_file.invoke({"path": "../secret.txt"})


def test_read_file_rejects_binary_files(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "image.bin").write_bytes(b"abc\x00def")

    with pytest.raises(ValueError, match="text file"):
        tools.read_file.invoke({"path": "image.bin"})


def test_search_text_finds_matches_in_workspace(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("hello world\nno match\n", encoding="utf-8")
    (docs / "notes.txt").write_text("HELLO again\n", encoding="utf-8")

    result = tools.search_text.invoke({"query": "hello", "path": "docs"})

    assert result.splitlines() == [
        "docs/guide.md:1:hello world",
        "docs/notes.txt:1:HELLO again",
    ]


def test_search_text_rejects_empty_query(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="query cannot be empty"):
        tools.search_text.invoke({"query": "   ", "path": "."})


def test_search_text_rejects_path_traversal(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="outside the workspace"):
        tools.search_text.invoke({"query": "secret", "path": ".."})


def test_search_text_skips_symlinks_outside_workspace(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    outside_file = tmp_path.parent / f"{tmp_path.name}_outside.txt"
    outside_file.write_text("external secret\n", encoding="utf-8")
    (tmp_path / "linked.txt").symlink_to(outside_file)

    result = tools.search_text.invoke({"query": "external", "path": "."})

    assert result == "(no matches)"


def test_project_tree_outputs_limited_workspace_tree(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src"
    package = src / "agentloop"
    package.mkdir(parents=True)
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")

    result = tools.project_tree.invoke({"path": ".", "max_depth": 1})

    assert result.splitlines() == [
        ".",
        "  README.md",
        "  src/",
    ]


def test_project_tree_rejects_path_traversal(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="outside the workspace"):
        tools.project_tree.invoke({"path": ".."})
