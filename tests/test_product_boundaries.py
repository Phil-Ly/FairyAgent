from pathlib import Path


def test_product_source_has_no_demo_logic() -> None:
    source_root = Path("src/agentloop")

    assert source_root.is_dir()
    assert not any(path.name == "mini" + "_agent" for path in Path("src").iterdir())
    assert not (source_root / "demo.py").exists()
    for source_file in source_root.glob("*.py"):
        assert "demo" not in source_file.read_text().lower(), source_file
