"""Behavioural tests for transitive impact traversal."""

from __future__ import annotations

import sqlite3

from code_mcp import db
from code_mcp.graph.impact import impact
from code_mcp.types import CONF_BRIDGE, CONF_NAME, REF_CALL, REF_IMPORT


def _db(tmp_path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "index.db")
    db.init_schema(conn)
    return conn


def _file(conn: sqlite3.Connection, path: str) -> int:
    cursor = conn.execute(
        "INSERT INTO files(path, lang, mtime, hash, loc) VALUES (?, 'python', 0, '', 0)",
        (path,),
    )
    return int(cursor.lastrowid)


def _symbol(
    conn: sqlite3.Connection, file_id: int, name: str, start: int = 1
) -> int:
    cursor = conn.execute(
        """INSERT INTO symbols(file_id, name, qualname, kind, start_line, end_line,
                               start_byte, end_byte, signature, docstring)
           VALUES (?, ?, ?, 'function', ?, ?, 0, 0, '', '')""",
        (file_id, name, name, start, start + 1),
    )
    return int(cursor.lastrowid)


def _ref(
    conn: sqlite3.Connection,
    from_id: int,
    to_id: int,
    kind: str = REF_CALL,
    confidence: str = CONF_NAME,
) -> None:
    conn.execute(
        """INSERT INTO refs(from_symbol_id, to_symbol_id, to_name, kind, confidence)
           VALUES (?, ?, '', ?, ?)""",
        (from_id, to_id, kind, confidence),
    )


def _graph(conn: sqlite3.Connection) -> dict[str, int]:
    """a -> b -> c and a -> d, plus an import edge and a bridge edge to c."""
    fa = _file(conn, "a.py")
    fb = _file(conn, "b.py")
    fc = _file(conn, "c.py")
    fd = _file(conn, "d.py")
    ids = {
        "a": _symbol(conn, fa, "alpha", start=1),
        "b": _symbol(conn, fb, "beta", start=1),
        "c": _symbol(conn, fc, "gamma", start=1),
        "d": _symbol(conn, fd, "delta", start=1),
    }
    _ref(conn, ids["a"], ids["b"])
    _ref(conn, ids["b"], ids["c"], kind=REF_IMPORT)
    _ref(conn, ids["a"], ids["d"])
    _ref(conn, ids["d"], ids["c"], confidence=CONF_BRIDGE)
    conn.commit()
    return ids


def test_impact_walks_callers_breadth_first(tmp_path):
    conn = _db(tmp_path)
    ids = _graph(conn)

    nodes = impact(conn, ids["c"], depth=2)
    summary = [(node.symbol.name, node.depth, node.via) for node in nodes]

    # Level 1: beta (import) and delta (bridge), ordered by file path.
    # Level 2: alpha, reached through beta.
    assert summary == [
        ("beta", 1, "import"),
        ("delta", 1, "bridge"),
        ("alpha", 2, "call"),
    ]


def test_impact_respects_depth_and_limit(tmp_path):
    conn = _db(tmp_path)
    ids = _graph(conn)

    assert impact(conn, ids["c"], depth=1) and all(
        node.depth == 1 for node in impact(conn, ids["c"], depth=1)
    )
    assert impact(conn, ids["c"], depth=0) == []
    assert impact(conn, ids["c"], limit=0) == []

    limited = impact(conn, ids["c"], depth=2, limit=2)
    assert [node.symbol.name for node in limited] == ["beta", "delta"]


def test_impact_never_returns_the_origin(tmp_path):
    conn = _db(tmp_path)
    fa = _file(conn, "a.py")
    fb = _file(conn, "b.py")
    a = _symbol(conn, fa, "alpha")
    b = _symbol(conn, fb, "beta")
    _ref(conn, a, b)
    _ref(conn, b, a)  # cycle back to the origin
    conn.commit()

    nodes = impact(conn, a, depth=3)
    assert [node.symbol.name for node in nodes] == ["beta"]
    assert nodes[0].depth == 1


def test_impact_reports_each_node_at_its_shortest_depth(tmp_path):
    conn = _db(tmp_path)
    fa = _file(conn, "a.py")
    fb = _file(conn, "b.py")
    fc = _file(conn, "c.py")
    fd = _file(conn, "d.py")
    a = _symbol(conn, fa, "alpha")
    b = _symbol(conn, fb, "beta")
    c = _symbol(conn, fc, "gamma")
    d = _symbol(conn, fd, "delta")
    _ref(conn, b, a)  # beta and gamma both reach alpha directly...
    _ref(conn, c, a)
    _ref(conn, c, b)  # ...so gamma must stay at depth 1, not 2
    _ref(conn, d, b)
    conn.commit()

    nodes = impact(conn, a, depth=3)
    summary = [(node.symbol.name, node.depth) for node in nodes]
    assert summary == [("beta", 1), ("gamma", 1), ("delta", 2)]
