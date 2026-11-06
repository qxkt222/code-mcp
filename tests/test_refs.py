"""Behavioural tests for reference resolution and adjacency queries."""

from __future__ import annotations

import sqlite3

from code_mcp import db
from code_mcp.graph.refs import callers, callees, find_symbol, resolve_refs
from code_mcp.types import CONF_NAME, REF_CALL, REF_IMPORT


def _db(tmp_path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "index.db")
    db.init_schema(conn)
    return conn


def _file(conn: sqlite3.Connection, path: str, lang: str = "python") -> int:
    cursor = conn.execute(
        "INSERT INTO files(path, lang, mtime, hash, loc) VALUES (?, ?, 0, '', 0)",
        (path, lang),
    )
    return int(cursor.lastrowid)


def _symbol(
    conn: sqlite3.Connection,
    file_id: int,
    name: str,
    qualname: str | None = None,
    kind: str = "function",
    start: int = 1,
    end: int = 1,
) -> int:
    cursor = conn.execute(
        """INSERT INTO symbols(file_id, name, qualname, kind, start_line, end_line,
                               start_byte, end_byte, signature, docstring)
           VALUES (?, ?, ?, ?, ?, ?, 0, 0, '', '')""",
        (file_id, name, qualname or name, kind, start, end),
    )
    return int(cursor.lastrowid)


def _ref(
    conn: sqlite3.Connection,
    from_id: int,
    to_name: str,
    to_id: int | None = None,
    kind: str = REF_CALL,
) -> None:
    conn.execute(
        """INSERT INTO refs(from_symbol_id, to_symbol_id, to_name, kind, confidence)
           VALUES (?, ?, ?, ?, ?)""",
        (from_id, to_id, to_name, kind, CONF_NAME),
    )


def _unresolved(conn: sqlite3.Connection, to_name: str) -> list[tuple[int, object]]:
    return [
        (row["id"], row["to_symbol_id"])
        for row in conn.execute(
            "SELECT id, to_symbol_id FROM refs WHERE to_name = ? ORDER BY id", (to_name,)
        )
    ]


def test_resolve_refs_backfills_only_unambiguous_names(tmp_path):
    conn = _db(tmp_path)
    a = _file(conn, "pkg/a.py")
    b = _file(conn, "pkg/b.py")
    c = _file(conn, "pkg/c.py")

    main = _symbol(conn, a, "main", start=1, end=9)
    _symbol(conn, a, "helper", start=11, end=12)
    _symbol(conn, b, "helper", start=1, end=2)
    util = _symbol(conn, c, "util", start=1, end=2)

    _ref(conn, main, "helper")
    _ref(conn, main, "util")
    _ref(conn, main, "ghost")
    conn.commit()

    assert resolve_refs(conn) == 1

    assert _unresolved(conn, "util") == [(2, util)]
    # Duplicate ``helper`` definitions must not be guessed between.
    assert all(target is None for _id, target in _unresolved(conn, "helper"))
    assert all(target is None for _id, target in _unresolved(conn, "ghost"))

    # Nothing left to resolve the second time around.
    assert resolve_refs(conn) == 0


def test_resolve_refs_uses_unique_qualname_for_qualified_refs(tmp_path):
    conn = _db(tmp_path)
    a = _file(conn, "a.py")
    source = _symbol(conn, a, "source", start=1, end=3)
    runner_run = _symbol(conn, a, "run", qualname="Runner.run", kind="method", start=5, end=7)
    _symbol(conn, a, "run", qualname="Worker.run", kind="method", start=9, end=11)

    _ref(conn, source, "Runner.run")
    _ref(conn, source, "run")  # ambiguous bare name
    conn.commit()

    assert resolve_refs(conn) == 1
    assert _unresolved(conn, "Runner.run") == [(1, runner_run)]
    assert _unresolved(conn, "run")[0][1] is None


def test_resolve_refs_supports_rust_style_qualnames(tmp_path):
    conn = _db(tmp_path)
    f = _file(conn, "src/lib.rs", "rust")
    source = _symbol(conn, f, "caller", start=1, end=3)
    runner = _symbol(conn, f, "run", qualname="Runner::run", kind="method", start=5, end=7)
    _symbol(conn, f, "run", qualname="Worker::run", kind="method", start=9, end=11)

    _ref(conn, source, "Runner::run")
    conn.commit()

    assert resolve_refs(conn) == 1
    assert _unresolved(conn, "Runner::run") == [(1, runner)]


def test_resolve_refs_accepts_unique_bare_name_for_qualified_symbol(tmp_path):
    conn = _db(tmp_path)
    a = _file(conn, "a.py")
    source = _symbol(conn, a, "source", start=1, end=3)
    method = _symbol(conn, a, "run", qualname="Runner.run", kind="method", start=5, end=7)
    _ref(conn, source, "run")
    conn.commit()

    assert resolve_refs(conn) == 1
    assert _unresolved(conn, "run") == [(1, method)]


def test_callers_and_callees(tmp_path):
    conn = _db(tmp_path)
    fa = _file(conn, "a.py")
    fb = _file(conn, "b.py")
    fc = _file(conn, "c.py")
    a = _symbol(conn, fa, "alpha", start=1, end=2)
    b = _symbol(conn, fb, "beta", start=1, end=2)
    c = _symbol(conn, fc, "gamma", start=1, end=2)

    _ref(conn, a, "beta", to_id=b)
    _ref(conn, c, "beta", to_id=b)
    _ref(conn, b, "alpha", to_id=a, kind=REF_IMPORT)
    _ref(conn, a, "alpha", to_id=a)  # self reference is ignored
    conn.commit()

    assert [s.name for s in callers(conn, b)] == ["alpha", "gamma"]
    assert [s.name for s in callees(conn, b)] == ["alpha"]
    assert [s.name for s in callees(conn, a)] == ["beta"]

    limited = callers(conn, b, limit=1)
    assert [s.name for s in limited] == ["alpha"]
    assert callers(conn, b, limit=0) == []


def test_callers_populates_file_and_lines(tmp_path):
    conn = _db(tmp_path)
    fa = _file(conn, "src/a.py")
    fb = _file(conn, "src/b.py")
    a = _symbol(conn, fa, "alpha", start=3, end=6, kind="function")
    b = _symbol(conn, fb, "beta", start=10, end=12)
    _ref(conn, a, "beta", to_id=b)
    conn.commit()

    caller = callers(conn, b)[0]
    assert caller.file == "src/a.py"
    assert caller.start_line == 3
    assert caller.end_line == 6
    assert caller.kind == "function"


def test_find_symbol_prefers_qualname_then_name(tmp_path):
    conn = _db(tmp_path)
    f = _file(conn, "pkg/mod.py")
    _symbol(conn, f, "Runner", kind="class", start=1, end=10)
    method = _symbol(conn, f, "run", qualname="Runner.run", kind="method", start=3, end=5)
    top = _symbol(conn, f, "run", qualname="run", start=12, end=14)

    found = find_symbol(conn, "pkg/mod.py", "Runner.run")
    assert found is not None and found.id == method

    found = find_symbol(conn, "pkg/mod.py", "run")
    assert found is not None and found.id == top

    # Windows separators and a ``./`` prefix are normalised.
    found = find_symbol(conn, ".\\pkg\\mod.py", "Runner")
    assert found is not None and found.kind == "class"

    assert find_symbol(conn, "pkg/mod.py", "missing") is None
    assert find_symbol(conn, "nope/mod.py", "run") is None
