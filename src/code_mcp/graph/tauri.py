"""Bridges between Rust ``#[tauri::command]`` functions and JS ``invoke`` calls.

Only the two hard-coded Tauri conventions are recognised:

* Rust: a ``#[tauri::command]`` attribute directly above ``fn name(...)``.
* JS/TS: ``invoke("name")`` or ``invoke<Type>("name")`` with a string literal.

A matching name on both sides creates a ``REF_CALL`` edge with
``confidence=CONF_BRIDGE`` from the calling symbol to the Rust command.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from .. import db
from ..types import CONF_BRIDGE, REF_CALL, Symbol
from .refs import SYMBOL_COLUMNS, row_to_symbol

_RUST_COMMAND = re.compile(
    r"#\s*\[\s*tauri\s*::\s*command\s*(?:\([^)]*\))?\s*\]"
    r"(?:\s*#\s*\[[^\]]*\])*"
    r"\s*(?:pub(?:\s*\([^)]*\))?\s+)?"
    r"(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?"
    r"fn\s+([A-Za-z_][A-Za-z0-9_]*)"
)

_INVOKE = re.compile(r"\binvoke\s*(?:<[^()]*>)?\s*\(\s*(['\"])([^'\"]+)\1")

_FRONTEND_LANGS = ("typescript", "tsx", "javascript")


def _read(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _innermost_symbol(
    conn: sqlite3.Connection, file_id: int, line: int
) -> Symbol | None:
    row = conn.execute(
        f"""SELECT {SYMBOL_COLUMNS}
            FROM symbols AS s
            JOIN files AS f ON f.id = s.file_id
            WHERE s.file_id = ? AND s.start_line <= ? AND s.end_line >= ?
            ORDER BY (s.end_line - s.start_line) ASC, s.start_line DESC, s.id
            LIMIT 1""",
        (file_id, line, line),
    ).fetchone()
    return row_to_symbol(row) if row is not None else None


def _rust_command_symbol(
    conn: sqlite3.Connection, file_id: int, name: str, line: int
) -> Symbol | None:
    row = conn.execute(
        f"""SELECT {SYMBOL_COLUMNS}
            FROM symbols AS s
            JOIN files AS f ON f.id = s.file_id
            WHERE s.file_id = ? AND s.name = ?
            ORDER BY ABS(s.start_line - ?), s.start_line, s.id
            LIMIT 1""",
        (file_id, name, line),
    ).fetchone()
    return row_to_symbol(row) if row is not None else None


def _bridge_exists(
    conn: sqlite3.Connection, from_id: int, to_id: int
) -> bool:
    row = conn.execute(
        """SELECT 1 FROM refs
           WHERE from_symbol_id = ? AND to_symbol_id = ?
             AND kind = ? AND confidence = ?
           LIMIT 1""",
        (from_id, to_id, REF_CALL, CONF_BRIDGE),
    ).fetchone()
    return row is not None


def _root(conn: sqlite3.Connection) -> Path | None:
    value = db.get_meta(conn, "root")
    if not value:
        return None
    root = Path(value)
    return root if root.is_dir() else None


def build_bridges(conn: sqlite3.Connection) -> int:
    """Create bridge refs for matching Tauri commands and ``invoke`` calls.

    Source files are read from the repository root recorded in ``meta``. The
    function is idempotent: existing bridge edges are not duplicated. Returns
    the number of newly created edges.
    """
    root = _root(conn)
    if root is None:
        return 0

    rows = conn.execute(
        "SELECT id, path, lang FROM files ORDER BY path"
    ).fetchall()

    # First pass: collect Rust command symbols by name.
    commands: dict[str, list[Symbol]] = {}
    for row in rows:
        if row["lang"] != "rust":
            continue
        text = _read(root / row["path"])
        if "tauri" not in text:
            continue
        for match in _RUST_COMMAND.finditer(text):
            name = match.group(1)
            symbol = _rust_command_symbol(
                conn, int(row["id"]), name, _line_of(text, match.start())
            )
            if symbol is not None:
                commands.setdefault(name, []).append(symbol)

    if not commands:
        return 0

    # Second pass: frontend ``invoke`` literals become edges to the command.
    created = 0
    for row in rows:
        if row["lang"] not in _FRONTEND_LANGS:
            continue
        text = _read(root / row["path"])
        if "invoke" not in text:
            continue
        for match in _INVOKE.finditer(text):
            name = match.group(2)
            targets = commands.get(name)
            if not targets:
                continue
            caller = _innermost_symbol(
                conn, int(row["id"]), _line_of(text, match.start())
            )
            if caller is None:
                # refs.from_symbol_id is NOT NULL: skip module-level calls.
                continue
            for target in targets:
                if _bridge_exists(conn, caller.id, target.id):
                    continue
                conn.execute(
                    """INSERT INTO refs(from_symbol_id, to_symbol_id, to_name, kind, confidence)
                       VALUES (?, ?, ?, ?, ?)""",
                    (caller.id, target.id, name, REF_CALL, CONF_BRIDGE),
                )
                created += 1

    if created:
        conn.commit()
    return created


def find_bridge_targets(conn: sqlite3.Connection, name: str) -> list[Symbol]:
    """Rust commands reachable through stored bridge refs to ``name``."""
    rows = conn.execute(
        f"""SELECT DISTINCT {SYMBOL_COLUMNS}
            FROM refs AS r
            JOIN symbols AS s ON s.id = r.to_symbol_id
            JOIN files AS f ON f.id = s.file_id
            WHERE r.confidence = ? AND r.to_name = ?
            ORDER BY f.path, s.start_line, s.id""",
        (CONF_BRIDGE, name),
    ).fetchall()
    return [row_to_symbol(row) for row in rows]
