"""Behavioural tests for the hard-coded Tauri bridge rules."""

from __future__ import annotations

import sqlite3

from code_mcp import db
from code_mcp.graph.tauri import build_bridges, find_bridge_targets
from code_mcp.types import CONF_BRIDGE, REF_CALL

RUST_SOURCE = """use tauri::command;

#[tauri::command]
pub fn greet(name: String) -> String {
    format!("hello {name}")
}

pub fn helper() -> u8 {
    1
}

#[tauri::command]
#[allow(unused_variables)]
pub async fn ping(ready: bool) -> String {
    "pong".into()
}
"""

TS_SOURCE = """export async function loadProfile() {
  const greeting = await invoke<string>("greet", { name: "Ada" });
  return greeting;
}

export function toggle(flag: boolean) {
  return invoke("ping", { ready: flag });
}

export function dynamic(name: string) {
  return invoke(name);
}
"""


def _db(tmp_path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "index.db")
    db.init_schema(conn)
    return conn


def _file(conn: sqlite3.Connection, path: str, lang: str) -> int:
    cursor = conn.execute(
        "INSERT INTO files(path, lang, mtime, hash, loc) VALUES (?, ?, 0, '', 0)",
        (path, lang),
    )
    return int(cursor.lastrowid)


def _symbol(
    conn: sqlite3.Connection,
    file_id: int,
    name: str,
    start: int,
    end: int,
    kind: str = "function",
) -> int:
    cursor = conn.execute(
        """INSERT INTO symbols(file_id, name, qualname, kind, start_line, end_line,
                               start_byte, end_byte, signature, docstring)
           VALUES (?, ?, ?, ?, ?, ?, 0, 0, '', '')""",
        (file_id, name, name, kind, start, end),
    )
    return int(cursor.lastrowid)


def _setup(tmp_path) -> tuple[sqlite3.Connection, dict[str, int]]:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text(RUST_SOURCE, encoding="utf-8")
    (tmp_path / "src" / "app.ts").write_text(TS_SOURCE, encoding="utf-8")

    conn = _db(tmp_path)
    rust = _file(conn, "src/lib.rs", "rust")
    ts = _file(conn, "src/app.ts", "typescript")

    ids = {
        "greet": _symbol(conn, rust, "greet", 4, 6),
        "helper": _symbol(conn, rust, "helper", 8, 10),
        "ping": _symbol(conn, rust, "ping", 14, 16),
        "loadProfile": _symbol(conn, ts, "loadProfile", 1, 4),
        "toggle": _symbol(conn, ts, "toggle", 6, 8),
        "dynamic": _symbol(conn, ts, "dynamic", 10, 12),
    }
    db.set_meta(conn, "root", str(tmp_path))
    conn.commit()
    return conn, ids


def test_build_bridges_links_rust_commands_to_invoke_calls(tmp_path):
    conn, ids = _setup(tmp_path)

    assert build_bridges(conn) == 2

    rows = conn.execute(
        """SELECT from_symbol_id, to_symbol_id, to_name, kind, confidence
           FROM refs ORDER BY from_symbol_id, to_symbol_id"""
    ).fetchall()
    edges = {(row["from_symbol_id"], row["to_symbol_id"], row["to_name"]) for row in rows}

    assert edges == {
        (ids["loadProfile"], ids["greet"], "greet"),
        (ids["toggle"], ids["ping"], "ping"),
    }
    assert all(row["kind"] == REF_CALL for row in rows)
    assert all(row["confidence"] == CONF_BRIDGE for row in rows)


def test_build_bridges_is_idempotent(tmp_path):
    conn, _ids = _setup(tmp_path)
    assert build_bridges(conn) == 2
    assert build_bridges(conn) == 0
    count = conn.execute("SELECT COUNT(*) AS n FROM refs").fetchone()["n"]
    assert count == 2


def test_build_bridges_without_root_does_nothing(tmp_path):
    conn = _db(tmp_path)
    rust = _file(conn, "src/lib.rs", "rust")
    _symbol(conn, rust, "greet", 4, 6)
    assert build_bridges(conn) == 0


def test_build_bridges_ignores_plain_functions_and_dynamic_invokes(tmp_path):
    conn, ids = _setup(tmp_path)
    build_bridges(conn)

    outgoing = {
        row["from_symbol_id"]
        for row in conn.execute("SELECT from_symbol_id FROM refs")
    }
    assert ids["dynamic"] not in outgoing  # invoke(name) is not a literal
    assert ids["helper"] not in outgoing  # no #[tauri::command]
    assert ids["greet"] not in outgoing


def test_find_bridge_targets(tmp_path):
    conn, ids = _setup(tmp_path)
    build_bridges(conn)

    greet = find_bridge_targets(conn, "greet")
    assert [symbol.id for symbol in greet] == [ids["greet"]]
    assert greet[0].file == "src/lib.rs"
    assert greet[0].name == "greet"

    assert [symbol.id for symbol in find_bridge_targets(conn, "ping")] == [ids["ping"]]
    assert find_bridge_targets(conn, "missing") == []
