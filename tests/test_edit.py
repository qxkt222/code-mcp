"""Behaviour of the guarded single-symbol rewrite path."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from code_mcp import db
from code_mcp.config import DEFAULT_EXCLUDES, Config
from code_mcp.edit import replace_symbol
from code_mcp.graph.impact import impact
from code_mcp.graph.refs import resolve_refs
from code_mcp.index.builder import build_index
from code_mcp.index.modules import resolve_imports
from code_mcp.paths import resolve_within

PY_SOURCE = (
    b"import os\n"
    b"\n"
    b"def target(x):\n"
    b"    return x + 1\n"
    b"\n"
    b"\n"
    b"def keep():\n"
    b"    return 42\n"
)

TS_SOURCE = (
    b"export function target(x: number): number {\n"
    b"  return x + 1;\n"
    b"}\n"
    b"\n"
    b"export function keep(): number {\n"
    b"  return 2;\n"
    b"}\n"
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(db.DB_HOME_ENV, str(tmp_path / "_home"))


def _lang_for(rel: str) -> str:
    return {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
        ".rs": "rust",
    }[Path(rel).suffix.lower()]


def _index(
    root: Path,
    rel: str,
    data: bytes,
    symbols: list[tuple[str, str, str, str, str]],
    *,
    hash_override: str | None = None,
) -> str:
    """Create the file and an index entry. Symbol tuples are (name, qualname, kind, start_marker, end_marker)."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

    digest = hashlib.sha256(data).hexdigest()
    conn = db.open_index(root)
    try:
        cursor = conn.execute(
            "INSERT INTO files(path, lang, mtime, hash, loc) VALUES (?, ?, ?, ?, ?)",
            (rel, _lang_for(rel), path.stat().st_mtime, hash_override or digest, data.count(b"\n")),
        )
        file_id = cursor.lastrowid
        for name, qualname, kind, start_marker, end_marker in symbols:
            start = data.index(start_marker.encode("utf-8"))
            end = data.index(end_marker.encode("utf-8"), start) + len(end_marker.encode("utf-8"))
            conn.execute(
                "INSERT INTO symbols(file_id, name, qualname, kind, start_line, end_line, "
                "start_byte, end_byte, signature, docstring) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    file_id,
                    name,
                    qualname,
                    kind,
                    data[:start].count(b"\n") + 1,
                    data[:end].count(b"\n") + 1,
                    start,
                    end,
                    "",
                    "",
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return digest


def _indexed_hash(root: Path, rel: str) -> str:
    conn = db.connect(db.database_path(root))
    try:
        row = conn.execute("SELECT hash FROM files WHERE path = ?", (rel,)).fetchone()
        return row["hash"]
    finally:
        conn.close()


def test_replace_applies_and_preserves_rest(tmp_path: Path) -> None:
    _index(
        tmp_path,
        "pkg/mod.py",
        PY_SOURCE,
        [("target", "target", "function", "def target", "return x + 1")],
    )

    result = replace_symbol(
        tmp_path,
        "pkg/mod.py",
        "target",
        "def target(x):\n    return x * 2\n",
        expect_hash=hashlib.sha256(PY_SOURCE).hexdigest(),
    )

    assert result.applied is True
    assert result.reason == ""
    assert result.old_hash == hashlib.sha256(PY_SOURCE).hexdigest()
    assert result.new_hash == hashlib.sha256((tmp_path / "pkg/mod.py").read_bytes()).hexdigest()
    assert result.lines_added >= 1
    assert result.lines_removed >= 1
    assert "return x * 2" in result.diff

    written = (tmp_path / "pkg/mod.py").read_bytes()
    assert written == PY_SOURCE.replace(b"return x + 1", b"return x * 2")
    assert b"def keep():" in written and b"return 42" in written


def test_lookup_by_qualname(tmp_path: Path) -> None:
    source = b"class Box:\n    def put(self, x):\n        return x\n"
    _index(
        tmp_path,
        "pkg/box.py",
        source,
        [("put", "Box.put", "method", "def put", "return x")],
    )

    result = replace_symbol(
        tmp_path,
        "pkg/box.py",
        "Box.put",
        "def put(self, x):\n        return x + 1",
    )

    assert result.applied is True
    assert (tmp_path / "pkg/box.py").read_bytes() == source.replace(b"return x", b"return x + 1")


def test_typescript_edit_applies(tmp_path: Path) -> None:
    _index(
        tmp_path,
        "src/main.ts",
        TS_SOURCE,
        [("target", "target", "function", "export function target", "return x + 1;\n}")],
    )

    result = replace_symbol(
        tmp_path,
        "src/main.ts",
        "target",
        "export function target(x: number): number {\n  return x * 2;\n}",
    )

    assert result.applied is True
    assert b"return x * 2;" in (tmp_path / "src/main.ts").read_bytes()


def test_rust_edit_applies(tmp_path: Path) -> None:
    source = (
        b"pub fn target(x: i32) -> i32 {\n"
        b"    x + 1\n"
        b"}\n"
        b"\n"
        b"pub fn keep() -> i32 {\n"
        b"    2\n"
        b"}\n"
    )
    _index(
        tmp_path,
        "src/lib.rs",
        source,
        [("target", "target", "function", "pub fn target", "x + 1\n}")],
    )

    result = replace_symbol(
        tmp_path,
        "src/lib.rs",
        "target",
        "pub fn target(x: i32) -> i32 {\n    x + 2\n}",
    )

    assert result.applied is True
    assert b"x + 2" in (tmp_path / "src/lib.rs").read_bytes()


def test_index_hash_mismatch_rejects(tmp_path: Path) -> None:
    _index(
        tmp_path,
        "pkg/mod.py",
        PY_SOURCE,
        [("target", "target", "function", "def target", "return x + 1")],
        hash_override="0" * 64,
    )
    before = (tmp_path / "pkg/mod.py").read_bytes()

    result = replace_symbol(tmp_path, "pkg/mod.py", "target", "def target(x):\n    return 2\n")

    assert result.applied is False
    assert "hash" in result.reason.lower()
    assert (tmp_path / "pkg/mod.py").read_bytes() == before


def test_expect_hash_mismatch_rejects(tmp_path: Path) -> None:
    _index(
        tmp_path,
        "pkg/mod.py",
        PY_SOURCE,
        [("target", "target", "function", "def target", "return x + 1")],
    )
    before = (tmp_path / "pkg/mod.py").read_bytes()

    result = replace_symbol(
        tmp_path,
        "pkg/mod.py",
        "target",
        "def target(x):\n    return 2\n",
        expect_hash="deadbeef",
    )

    assert result.applied is False
    assert "expect_hash" in result.reason
    assert (tmp_path / "pkg/mod.py").read_bytes() == before


def test_crlf_file_stays_crlf(tmp_path: Path) -> None:
    crlf_source = (
        b"def target(x):\r\n"
        b"    return x + 1\r\n"
        b"\r\n"
        b"def keep():\r\n"
        b"    return 2\r\n"
    )
    _index(
        tmp_path,
        "pkg/crlf.py",
        crlf_source,
        [("target", "target", "function", "def target", "    return x + 1")],
    )

    result = replace_symbol(
        tmp_path,
        "pkg/crlf.py",
        "target",
        "def target(x):\n    return x * 2\n",
    )

    assert result.applied is True
    written = (tmp_path / "pkg/crlf.py").read_bytes()
    assert written == crlf_source.replace(b"return x + 1", b"return x * 2")
    assert b"\r\n" in written
    assert written.replace(b"\r\n", b"").count(b"\n") == 0
    assert written.replace(b"\r\n", b"").count(b"\r") == 0


def test_python_syntax_error_rolls_back(tmp_path: Path) -> None:
    _index(
        tmp_path,
        "pkg/mod.py",
        PY_SOURCE,
        [("target", "target", "function", "def target", "return x + 1")],
    )
    before = (tmp_path / "pkg/mod.py").read_bytes()

    result = replace_symbol(
        tmp_path,
        "pkg/mod.py",
        "target",
        "def target(x):\n    return (x +\n",
    )

    assert result.applied is False
    assert "syntax" in result.reason.lower() or "valid python" in result.reason.lower()
    assert (tmp_path / "pkg/mod.py").read_bytes() == before


def test_typescript_syntax_error_rolls_back(tmp_path: Path) -> None:
    _index(
        tmp_path,
        "src/main.ts",
        TS_SOURCE,
        [("target", "target", "function", "export function target", "return x + 1;\n}")],
    )
    before = (tmp_path / "src/main.ts").read_bytes()

    result = replace_symbol(
        tmp_path,
        "src/main.ts",
        "target",
        "export function target(x: number): number {\n  return x +;\n}",
    )

    assert result.applied is False
    assert "syntax" in result.reason.lower()
    assert (tmp_path / "src/main.ts").read_bytes() == before


def test_renamed_symbol_is_rejected(tmp_path: Path) -> None:
    _index(
        tmp_path,
        "pkg/mod.py",
        PY_SOURCE,
        [("target", "target", "function", "def target", "return x + 1")],
    )
    before = (tmp_path / "pkg/mod.py").read_bytes()

    result = replace_symbol(
        tmp_path,
        "pkg/mod.py",
        "target",
        "def renamed(x):\n    return x\n",
    )

    assert result.applied is False
    assert "no longer defined" in result.reason
    assert (tmp_path / "pkg/mod.py").read_bytes() == before


def test_unknown_symbol_is_rejected(tmp_path: Path) -> None:
    _index(
        tmp_path,
        "pkg/mod.py",
        PY_SOURCE,
        [("target", "target", "function", "def target", "return x + 1")],
    )
    before = (tmp_path / "pkg/mod.py").read_bytes()

    result = replace_symbol(tmp_path, "pkg/mod.py", "missing", "def missing():\n    pass\n")

    assert result.applied is False
    assert "not found" in result.reason
    assert (tmp_path / "pkg/mod.py").read_bytes() == before


def test_index_hash_updated_after_apply(tmp_path: Path) -> None:
    _index(
        tmp_path,
        "pkg/mod.py",
        PY_SOURCE,
        [("target", "target", "function", "def target", "return x + 1")],
    )

    result = replace_symbol(
        tmp_path,
        "pkg/mod.py",
        "target",
        "def target(x):\n    return x - 1\n",
    )

    assert result.applied is True
    written = (tmp_path / "pkg/mod.py").read_bytes()
    assert _indexed_hash(tmp_path, "pkg/mod.py") == hashlib.sha256(written).hexdigest()


def _symbol_id(conn, rel: str, name: str) -> int | None:
    row = conn.execute(
        """SELECT s.id FROM symbols AS s JOIN files AS f ON f.id = s.file_id
           WHERE f.path = ? AND s.name = ?
           ORDER BY s.start_line, s.id LIMIT 1""",
        (rel, name),
    ).fetchone()
    return int(row["id"]) if row is not None else None


def test_edit_reconnects_refs_and_keeps_impact_non_empty(tmp_path: Path) -> None:
    """A1: replacing a file row cascade-nulls incoming refs; the edit must
    re-resolve them or impact() silently returns nothing."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text(
        "def helper(x):\n    return x + 1\n", encoding="utf-8"
    )
    (root / "b.py").write_text(
        "from a import helper\n\n\ndef caller():\n    return helper(1)\n",
        encoding="utf-8",
    )

    conn = db.open_index(root)
    try:
        build_index(root, conn, Config(root=root, excludes=list(DEFAULT_EXCLUDES)))
        resolve_imports(conn, root)
        resolve_refs(conn)
        before_id = _symbol_id(conn, "a.py", "helper")
        assert before_id is not None
        assert any(node.symbol.name == "caller" for node in impact(conn, before_id))
    finally:
        conn.close()

    result = replace_symbol(
        root, "a.py", "helper", "def helper(x):\n    return x + 2\n"
    )
    assert result.applied is True, result.reason

    conn = db.open_index(root)
    try:
        helper_id = _symbol_id(conn, "a.py", "helper")
        assert helper_id is not None
        row = conn.execute(
            """SELECT r.to_symbol_id FROM refs AS r
               JOIN symbols AS s ON s.id = r.from_symbol_id
               JOIN files AS f ON f.id = s.file_id
               WHERE f.path = 'b.py' AND r.to_name = 'helper'"""
        ).fetchone()
        assert row is not None and row["to_symbol_id"] == helper_id

        nodes = impact(conn, helper_id)
        assert any(node.symbol.name == "caller" for node in nodes)
    finally:
        conn.close()


def test_replace_symbol_rejects_relative_and_absolute_escapes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def target():\n    pass\n", encoding="utf-8")

    for candidate in ("../outside.py", str(outside)):
        result = replace_symbol(
            root, candidate, "target", "def target():\n    return 1\n"
        )
        assert result.applied is False
        assert "outside the repository root" in result.reason
    assert outside.read_text(encoding="utf-8") == "def target():\n    pass\n"


def test_replace_symbol_rejects_symlink_escape(tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret = outside_dir / "secret.py"
    secret.write_text("def target():\n    pass\n", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    link = root / "link.py"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")

    result = replace_symbol(root, "link.py", "target", "def target():\n    return 1\n")
    assert result.applied is False
    assert "outside the repository root" in result.reason


def test_resolve_within_covers_escape_cases(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "inside.py").write_text("", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    assert resolve_within(root, "inside.py") == (root / "inside.py").resolve()
    assert resolve_within(root, "../outside.txt") is None
    assert resolve_within(root, str(outside)) is None
    assert resolve_within(root, "") is None

    link = root / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    assert resolve_within(root, "link.txt") is None
