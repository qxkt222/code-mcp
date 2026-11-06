"""Incremental indexing: skip rules, replacements, deletions and exclusions."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

from code_mcp.config import DEFAULT_EXCLUDES, Config
from code_mcp.db import connect, init_schema
from code_mcp.index.builder import build_index


def _conn(tmp_path: Path) -> sqlite3.Connection:
    conn = connect(tmp_path / "index.db")
    init_schema(conn)
    return conn


def _config(root: Path, languages: list[str] | None = None) -> Config:
    return Config(
        root=root,
        excludes=list(DEFAULT_EXCLUDES),
        languages=list(languages or []),
    )


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def _paths(conn: sqlite3.Connection) -> set[str]:
    return {row["path"] for row in conn.execute("SELECT path FROM files")}


def _names(conn: sqlite3.Connection) -> set[str]:
    return {row["name"] for row in conn.execute("SELECT name FROM symbols")}


def test_build_indexes_supported_files_and_skips_others(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _write(root, "a.py", "def alpha():\n    pass\n")
    _write(root, "b.ts", "export function beta(): void {}\n")
    _write(root, "c.rs", "pub fn gamma() {}\n")
    _write(root, "notes.txt", "not source\n")
    _write(root, "node_modules/dep.py", "def dep(): pass\n")
    _write(root, ".git/hooks.py", "def hook(): pass\n")

    conn = _conn(tmp_path)
    stats = build_index(root, conn, _config(root))

    assert stats.files_indexed == 3
    assert stats.files_skipped == 0
    assert stats.files_removed == 0
    assert stats.symbols == 3  # the per-file module anchors are not counted
    assert _paths(conn) == {"a.py", "b.ts", "c.rs"}
    assert _names(conn) == {"alpha", "beta", "gamma"}

    row = conn.execute("SELECT * FROM files WHERE path = 'a.py'").fetchone()
    source = (root / "a.py").read_bytes()
    assert row["hash"] == hashlib.sha256(source).hexdigest()
    assert row["lang"] == "python"
    assert row["loc"] == 2


def test_incremental_skips_unchanged_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _write(root, "a.py", "def alpha():\n    pass\n")
    _write(root, "b.py", "def beta():\n    pass\n")

    conn = _conn(tmp_path)
    config = _config(root)
    first = build_index(root, conn, config)
    second = build_index(root, conn, config)

    assert first.files_indexed == 2
    assert second.files_indexed == 0
    assert second.files_skipped == 2
    assert second.symbols == 0
    assert second.refs == 0


def test_changed_file_replaces_old_symbols(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    path = _write(root, "a.py", "def alpha():\n    pass\n")
    _write(root, "b.py", "def beta():\n    pass\n")

    conn = _conn(tmp_path)
    config = _config(root)
    build_index(root, conn, config)
    path.write_bytes(b"def renamed():\n    pass\n")
    stats = build_index(root, conn, config)

    assert stats.files_indexed == 1
    assert stats.files_skipped == 1
    names = _names(conn)
    assert "renamed" in names
    assert "alpha" not in names
    assert "beta" in names


def test_mtime_change_triggers_reindex(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    path = _write(root, "a.py", "def alpha():\n    pass\n")

    conn = _conn(tmp_path)
    config = _config(root)
    build_index(root, conn, config)

    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    stats = build_index(root, conn, config)
    assert stats.files_indexed == 1
    assert stats.files_skipped == 0


def test_size_change_with_identical_mtime_triggers_reindex(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    path = _write(root, "a.py", "def one():\n    pass\n")

    conn = _conn(tmp_path)
    config = _config(root)
    build_index(root, conn, config)

    stat = path.stat()
    path.write_bytes(b"def one_much_longer_name():\n    pass\n")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_mtime == stat.st_mtime
    assert path.stat().st_size != stat.st_size

    stats = build_index(root, conn, config)
    assert stats.files_indexed == 1
    assert "one_much_longer_name" in _names(conn)
    assert "one" not in _names(conn)


def test_force_reindexes_everything_without_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _write(root, "a.py", "def alpha():\n    pass\n")
    _write(root, "b.ts", "export function beta(): void {}\n")

    conn = _conn(tmp_path)
    config = _config(root)
    build_index(root, conn, config)
    before = conn.execute("SELECT COUNT(*) AS n FROM symbols").fetchone()["n"]

    stats = build_index(root, conn, config, force=True)
    after = conn.execute("SELECT COUNT(*) AS n FROM symbols").fetchone()["n"]

    assert stats.files_indexed == 2
    assert stats.files_skipped == 0
    assert after == before


def test_deleted_file_is_removed_with_children(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    path = _write(
        root,
        "a.py",
        "import os\n\n# gone soon\ndef alpha():\n    return call()\n",
    )
    _write(root, "b.py", "def beta():\n    pass\n")

    conn = _conn(tmp_path)
    config = _config(root)
    build_index(root, conn, config)
    path.unlink()

    stats = build_index(root, conn, config)
    assert stats.files_removed == 1
    assert _paths(conn) == {"b.py"}
    assert "alpha" not in _names(conn)
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM comments WHERE text = '# gone soon'"
    ).fetchone()["n"] == 0


def test_module_level_refs_get_a_file_scope_anchor(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = b"import os\n\n\ndef first():\n    return call()\n"
    _write(root, "mod.py", source.decode())

    conn = _conn(tmp_path)
    build_index(root, conn, _config(root))

    anchor = conn.execute(
        "SELECT * FROM symbols WHERE kind = 'module'"
    ).fetchone()
    assert anchor is not None
    assert anchor["qualname"] == ""
    assert int(anchor["start_byte"]) == len(source)
    assert int(anchor["end_byte"]) == len(source)

    # The anchor must not shadow the real first top-level symbol.
    first = conn.execute(
        "SELECT name, kind FROM symbols ORDER BY start_line, id LIMIT 1"
    ).fetchone()
    assert first["name"] == "first"

    import_ref = conn.execute(
        "SELECT s.kind AS kind FROM refs r JOIN symbols s ON s.id = r.from_symbol_id "
        "WHERE r.kind = 'import'"
    ).fetchone()
    assert import_ref["kind"] == "module"


def test_config_language_filter(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _write(root, "a.py", "def alpha():\n    pass\n")
    _write(root, "b.rs", "pub fn beta() {}\n")

    conn = _conn(tmp_path)
    build_index(root, conn, _config(root, languages=["rust"]))
    assert _paths(conn) == {"b.rs"}


def test_empty_repository(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    conn = _conn(tmp_path)
    stats = build_index(root, conn, _config(root))
    assert stats.files_indexed == 0
    assert stats.files_skipped == 0
    assert stats.files_removed == 0
