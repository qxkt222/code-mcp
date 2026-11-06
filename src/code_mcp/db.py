"""SQLite storage for the code index.

The database lives outside the indexed repository (``~/.code-mcp/indexes``) so
that pointing the tool at a project never adds files to it. A schema version is
stored in ``meta`` so future migrations can be applied safely.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1
DB_HOME_ENV = "CODE_MCP_HOME"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id    INTEGER PRIMARY KEY,
    path  TEXT NOT NULL UNIQUE,
    lang  TEXT NOT NULL,
    mtime REAL NOT NULL,
    hash  TEXT NOT NULL,
    loc   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS symbols (
    id         INTEGER PRIMARY KEY,
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    qualname   TEXT NOT NULL,
    kind       TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    start_byte INTEGER NOT NULL,
    end_byte   INTEGER NOT NULL,
    signature  TEXT NOT NULL DEFAULT '',
    docstring  TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_symbols_name     ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_qualname ON symbols(qualname);
CREATE INDEX IF NOT EXISTS idx_symbols_file     ON symbols(file_id);

CREATE TABLE IF NOT EXISTS refs (
    id             INTEGER PRIMARY KEY,
    from_symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    to_symbol_id   INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    to_name        TEXT NOT NULL,
    kind           TEXT NOT NULL,
    confidence     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_refs_from ON refs(from_symbol_id);
CREATE INDEX IF NOT EXISTS idx_refs_to   ON refs(to_symbol_id);
CREATE INDEX IF NOT EXISTS idx_refs_name ON refs(to_name);

CREATE TABLE IF NOT EXISTS imports (
    id               INTEGER PRIMARY KEY,
    file_id          INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    module           TEXT NOT NULL,
    alias            TEXT NOT NULL DEFAULT '',
    resolved_file_id INTEGER REFERENCES files(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_imports_file ON imports(file_id);

CREATE TABLE IF NOT EXISTS comments (
    id      INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    line    INTEGER NOT NULL,
    text    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_comments_file ON comments(file_id);

CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    name, qualname, signature, docstring,
    content='symbols', content_rowid='id',
    tokenize='unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS comments_fts USING fts5(
    text,
    content='comments', content_rowid='id',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
    INSERT INTO symbols_fts(rowid, name, qualname, signature, docstring)
    VALUES (new.id, new.name, new.qualname, new.signature, new.docstring);
END;

CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, name, qualname, signature, docstring)
    VALUES ('delete', old.id, old.name, old.qualname, old.signature, old.docstring);
END;

CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, name, qualname, signature, docstring)
    VALUES ('delete', old.id, old.name, old.qualname, old.signature, old.docstring);
    INSERT INTO symbols_fts(rowid, name, qualname, signature, docstring)
    VALUES (new.id, new.name, new.qualname, new.signature, new.docstring);
END;

CREATE TRIGGER IF NOT EXISTS comments_ai AFTER INSERT ON comments BEGIN
    INSERT INTO comments_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS comments_ad AFTER DELETE ON comments BEGIN
    INSERT INTO comments_fts(comments_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""


def default_home() -> Path:
    """Directory holding index databases, overridable for tests and packaging."""
    override = os.environ.get(DB_HOME_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".code-mcp"


def database_path(root: Path) -> Path:
    """Stable per-repository database path, outside the repository itself."""
    resolved = root.resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:16]
    return default_home() / "indexes" / f"{digest}.db"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    set_meta(conn, "schema_version", str(SCHEMA_VERSION))
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def open_index(root: Path) -> sqlite3.Connection:
    """Open (creating if needed) the index for ``root`` and record the root path."""
    db_path = database_path(root)
    conn = connect(db_path)
    init_schema(conn)
    set_meta(conn, "root", str(root.resolve()))
    conn.commit()
    return conn
