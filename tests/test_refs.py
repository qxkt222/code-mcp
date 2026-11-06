"""Behavioural tests for reference resolution and adjacency queries."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from code_mcp import db
from code_mcp.graph.refs import callers, callees, find_symbol, resolve_refs
from code_mcp.types import (
    CONF_BRIDGE,
    CONF_EXACT,
    CONF_NAME,
    REF_CALL,
    REF_IMPORT,
)


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
    confidence: str = CONF_NAME,
) -> int:
    cursor = conn.execute(
        """INSERT INTO refs(from_symbol_id, to_symbol_id, to_name, kind, confidence)
           VALUES (?, ?, ?, ?, ?)""",
        (from_id, to_id, to_name, kind, confidence),
    )
    return int(cursor.lastrowid)


def _import(
    conn: sqlite3.Connection,
    file_id: int,
    module: str,
    alias: str = "",
    resolved_file_id: int | None = None,
) -> None:
    conn.execute(
        """INSERT INTO imports(file_id, module, alias, resolved_file_id)
           VALUES (?, ?, ?, ?)""",
        (file_id, module, alias, resolved_file_id),
    )


def _refs_named(conn: sqlite3.Connection, to_name: str) -> list[tuple[int, object, str]]:
    return [
        (row["id"], row["to_symbol_id"], row["confidence"])
        for row in conn.execute(
            "SELECT id, to_symbol_id, confidence FROM refs WHERE to_name = ? ORDER BY id",
            (to_name,),
        )
    ]


def test_same_file_unique_name_resolves_as_exact(tmp_path):
    conn = _db(tmp_path)
    a = _file(conn, "pkg/a.py")
    b = _file(conn, "pkg/b.py")
    c = _file(conn, "pkg/c.py")

    main = _symbol(conn, a, "main", start=1, end=9)
    helper = _symbol(conn, a, "helper", start=11, end=12)
    _symbol(conn, b, "helper", start=1, end=2)  # duplicate elsewhere
    ref_id = _ref(conn, main, "helper")
    conn.commit()

    assert resolve_refs(conn) == 1
    assert _refs_named(conn, "helper") == [(ref_id, helper, CONF_EXACT)]
    # Nothing left to resolve the second time around.
    assert resolve_refs(conn) == 0

    # c.py neither defines the name nor imports a defining file, so the
    # duplicate makes the ref ambiguous and it stays unresolved.
    other = _symbol(conn, c, "caller", start=5, end=6)
    _ref(conn, other, "helper")
    conn.commit()
    assert resolve_refs(conn) == 0
    assert _refs_named(conn, "helper")[-1][1] is None


def test_globally_unique_name_without_import_is_not_resolved(tmp_path):
    conn = _db(tmp_path)
    a = _file(conn, "pkg/a.py")
    c = _file(conn, "pkg/c.py")

    main = _symbol(conn, a, "main", start=1, end=9)
    _symbol(conn, c, "util", start=1, end=2)
    ref_id = _ref(conn, main, "util")
    conn.commit()

    assert resolve_refs(conn) == 0
    assert _refs_named(conn, "util") == [(ref_id, None, CONF_NAME)]


def test_missing_name_is_left_unresolved(tmp_path):
    conn = _db(tmp_path)
    a = _file(conn, "pkg/a.py")
    main = _symbol(conn, a, "main", start=1, end=9)
    ref_id = _ref(conn, main, "ghost")
    conn.commit()

    assert resolve_refs(conn) == 0
    assert _refs_named(conn, "ghost") == [(ref_id, None, CONF_NAME)]


def test_cross_file_resolves_when_import_points_at_target(tmp_path):
    conn = _db(tmp_path)
    a = _file(conn, "pkg/a.py")
    b = _file(conn, "pkg/b.py")

    helper = _symbol(conn, a, "helper", start=1, end=2)
    caller = _symbol(conn, b, "caller", start=5, end=7)
    _import(conn, b, "pkg.a", "helper", resolved_file_id=a)
    ref_id = _ref(conn, caller, "helper")
    conn.commit()

    assert resolve_refs(conn) == 1
    assert _refs_named(conn, "helper") == [(ref_id, helper, CONF_EXACT)]


def test_cross_file_resolves_from_raw_module_without_persisted_target(tmp_path):
    """After an edit the stored import target is cascade-nulled; the raw
    module specifier still has to provide the evidence."""
    conn = _db(tmp_path)
    db.set_meta(conn, "root", str(tmp_path))
    a = _file(conn, "pkg/a.py")
    b = _file(conn, "pkg/b.py")

    helper = _symbol(conn, a, "helper", start=1, end=2)
    caller = _symbol(conn, b, "caller", start=5, end=7)
    _import(conn, b, "pkg.a", "helper", resolved_file_id=None)
    ref_id = _ref(conn, caller, "helper")
    conn.commit()

    assert resolve_refs(conn) == 1
    assert _refs_named(conn, "helper") == [(ref_id, helper, CONF_EXACT)]


def test_local_definition_wins_over_imported_duplicate(tmp_path):
    conn = _db(tmp_path)
    a = _file(conn, "pkg/a.py")
    b = _file(conn, "pkg/b.py")

    local = _symbol(conn, a, "helper", start=1, end=2)
    caller = _symbol(conn, a, "caller", start=5, end=7)
    _symbol(conn, b, "helper", start=1, end=2)
    _import(conn, a, "pkg.b", "helper", resolved_file_id=b)
    ref_id = _ref(conn, caller, "helper")
    conn.commit()

    assert resolve_refs(conn) == 1
    assert _refs_named(conn, "helper") == [(ref_id, local, CONF_EXACT)]


def test_ambiguous_imported_candidates_stay_unresolved(tmp_path):
    conn = _db(tmp_path)
    a = _file(conn, "pkg/a.py")
    b = _file(conn, "pkg/b.py")
    c = _file(conn, "pkg/c.py")

    caller = _symbol(conn, a, "caller", start=1, end=3)
    _symbol(conn, b, "helper", start=1, end=2)
    _symbol(conn, c, "helper", start=1, end=2)
    _import(conn, a, "pkg.b", "helper", resolved_file_id=b)
    _import(conn, a, "pkg.c", "helper", resolved_file_id=c)
    ref_id = _ref(conn, caller, "helper")
    conn.commit()

    assert resolve_refs(conn) == 0
    assert _refs_named(conn, "helper") == [(ref_id, None, CONF_NAME)]


def test_cross_language_names_never_resolve(tmp_path):
    conn = _db(tmp_path)
    rust_defs = _file(conn, "src/lib.rs", "rust")
    web_defs = _file(conn, "src/store.ts", "typescript")
    rust_caller_file = _file(conn, "src/other.rs", "rust")
    web_caller_file = _file(conn, "src/other.ts", "typescript")

    rust_read = _symbol(conn, rust_defs, "read", start=5, end=8)
    _symbol(conn, web_defs, "write", start=5, end=8)
    rust_caller = _symbol(conn, rust_caller_file, "read_config", start=1, end=3)
    web_caller = _symbol(conn, web_caller_file, "loadStore", start=1, end=3)

    # Bogus import evidence in both directions must still be rejected: only
    # same-family symbols may match. ``write`` exists only in TypeScript and
    # ``read`` only in Rust, so global uniqueness alone would have matched.
    _import(conn, rust_caller_file, "store", "", resolved_file_id=web_defs)
    _import(conn, web_caller_file, "lib", "", resolved_file_id=rust_defs)
    _ref(conn, rust_caller, "write", kind=REF_CALL)
    _ref(conn, web_caller, "read", kind=REF_CALL)
    conn.commit()

    assert resolve_refs(conn) == 0
    assert [row[1] for row in _refs_named(conn, "write")] == [None]
    assert [row[1] for row in _refs_named(conn, "read")] == [None]

    # Same-file resolution inside one language still works.
    local_caller = _symbol(conn, rust_defs, "local_caller", start=10, end=12)
    local_ref = _ref(conn, local_caller, "read")
    conn.commit()
    assert resolve_refs(conn) == 1
    assert _refs_named(conn, "read")[-1][0] == local_ref
    assert _refs_named(conn, "read")[-1][1] == rust_read


def test_js_dialects_share_one_language_family(tmp_path):
    conn = _db(tmp_path)
    ts = _file(conn, "src/a.ts", "typescript")
    tsx = _file(conn, "src/Button.tsx", "tsx")

    caller = _symbol(conn, ts, "render", start=1, end=3)
    button = _symbol(conn, tsx, "Button", kind="function", start=1, end=9)
    _import(conn, ts, "./Button", "Button", resolved_file_id=tsx)
    ref_id = _ref(conn, caller, "Button")
    conn.commit()

    assert resolve_refs(conn) == 1
    assert _refs_named(conn, "Button") == [(ref_id, button, CONF_EXACT)]


def test_already_resolved_and_bridge_rows_are_untouched(tmp_path):
    conn = _db(tmp_path)
    rust = _file(conn, "src/lib.rs", "rust")
    web = _file(conn, "src/store.ts", "typescript")

    command = _symbol(conn, rust, "greet", start=1, end=4)
    caller = _symbol(conn, web, "call", start=1, end=3)
    bridge_id = _ref(
        conn, caller, "greet", to_id=command, kind=REF_CALL, confidence=CONF_BRIDGE
    )
    conn.commit()

    assert resolve_refs(conn) == 0
    row = conn.execute(
        "SELECT to_symbol_id, confidence FROM refs WHERE id = ?", (bridge_id,)
    ).fetchone()
    assert row["to_symbol_id"] == command
    assert row["confidence"] == CONF_BRIDGE


def test_resolve_refs_uses_unique_qualname_for_qualified_refs(tmp_path):
    conn = _db(tmp_path)
    a = _file(conn, "a.py")
    source = _symbol(conn, a, "source", start=1, end=3)
    runner_run = _symbol(conn, a, "run", qualname="Runner.run", kind="method", start=5, end=7)
    _symbol(conn, a, "run", qualname="Worker.run", kind="method", start=9, end=11)

    qualified = _ref(conn, source, "Runner.run")
    bare = _ref(conn, source, "run")  # ambiguous bare name
    conn.commit()

    assert resolve_refs(conn) == 1
    assert _refs_named(conn, "Runner.run") == [(qualified, runner_run, CONF_EXACT)]
    assert _refs_named(conn, "run") == [(bare, None, CONF_NAME)]


def test_resolve_refs_supports_rust_style_qualnames(tmp_path):
    conn = _db(tmp_path)
    f = _file(conn, "src/lib.rs", "rust")
    source = _symbol(conn, f, "caller", start=1, end=3)
    runner = _symbol(conn, f, "run", qualname="Runner::run", kind="method", start=5, end=7)
    _symbol(conn, f, "run", qualname="Worker::run", kind="method", start=9, end=11)

    ref_id = _ref(conn, source, "Runner::run")
    conn.commit()

    assert resolve_refs(conn) == 1
    assert _refs_named(conn, "Runner::run") == [(ref_id, runner, CONF_EXACT)]


def test_resolve_refs_accepts_unique_bare_name_for_qualified_symbol(tmp_path):
    conn = _db(tmp_path)
    a = _file(conn, "a.py")
    source = _symbol(conn, a, "source", start=1, end=3)
    method = _symbol(conn, a, "run", qualname="Runner.run", kind="method", start=5, end=7)
    ref_id = _ref(conn, source, "run")
    conn.commit()

    assert resolve_refs(conn) == 1
    assert _refs_named(conn, "run") == [(ref_id, method, CONF_EXACT)]


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
