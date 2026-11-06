"""Reference resolution and adjacency queries over the SQLite index.

Everything here is schema-driven: it works no matter which indexer produced the
rows. Name matching is deliberately conservative -- an ambiguous name is left
unresolved rather than guessed.
"""

from __future__ import annotations

import sqlite3

from ..types import Symbol

# Column list shared by every symbol-returning query. Aliases are fixed so
# ``row_to_symbol`` can be reused from other graph modules.
SYMBOL_COLUMNS = (
    "s.id AS id, s.file_id AS file_id, f.path AS file, s.name AS name, "
    "s.qualname AS qualname, s.kind AS kind, s.start_line AS start_line, "
    "s.end_line AS end_line, s.start_byte AS start_byte, s.end_byte AS end_byte, "
    "s.signature AS signature, s.docstring AS docstring"
)


def normalize_path(path: str) -> str:
    """Normalise a repo-relative path to the form stored in ``files.path``."""
    cleaned = path.replace("\\", "/").strip()
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def is_qualified(name: str) -> bool:
    """True when ``name`` carries a container prefix such as ``Class.method``."""
    return "." in name or "::" in name


def row_to_symbol(row: sqlite3.Row) -> Symbol:
    """Build a :class:`Symbol` from a row selected with ``SYMBOL_COLUMNS``."""
    return Symbol(
        id=row["id"],
        file_id=row["file_id"],
        file=row["file"],
        name=row["name"],
        qualname=row["qualname"],
        kind=row["kind"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        start_byte=row["start_byte"],
        end_byte=row["end_byte"],
        signature=row["signature"],
        docstring=row["docstring"],
    )


def _resolve_id(
    to_name: str,
    by_name: dict[str, list[int]],
    by_qualname: dict[str, list[int]],
) -> int | None:
    """Resolve one name to a single symbol id, or return None when ambiguous."""
    matches = by_name.get(to_name, [])
    if len(matches) == 1:
        return matches[0]
    if matches:
        # Same bare name defined more than once: never guess between them.
        return None
    if not is_qualified(to_name):
        return None
    # The ref already carries a qualifier (``Class.method``): use it exactly.
    qualified = by_qualname.get(to_name, [])
    return qualified[0] if len(qualified) == 1 else None


def resolve_refs(conn: sqlite3.Connection) -> int:
    """Backfill ``refs.to_symbol_id`` where the target name is unambiguous.

    Only unresolved refs are touched. A name is accepted when it matches exactly
    one symbol name, or -- for qualified names such as ``Class.method`` -- when
    it matches exactly one qualname. Returns the number of refs updated.
    """
    rows = conn.execute(
        "SELECT id, to_name FROM refs WHERE to_symbol_id IS NULL ORDER BY id"
    ).fetchall()
    if not rows:
        return 0

    by_name: dict[str, list[int]] = {}
    by_qualname: dict[str, list[int]] = {}
    for symbol in conn.execute("SELECT id, name, qualname FROM symbols"):
        by_name.setdefault(symbol["name"], []).append(int(symbol["id"]))
        by_qualname.setdefault(symbol["qualname"], []).append(int(symbol["id"]))

    updates: list[tuple[int, int]] = []
    for row in rows:
        target = _resolve_id(row["to_name"], by_name, by_qualname)
        if target is not None:
            updates.append((target, int(row["id"])))

    if updates:
        conn.executemany(
            "UPDATE refs SET to_symbol_id = ? WHERE id = ?", updates
        )
        conn.commit()
    return len(updates)


def callers(
    conn: sqlite3.Connection, symbol_id: int, limit: int = 50
) -> list[Symbol]:
    """Symbols that reference ``symbol_id``, ordered by file and line."""
    if limit <= 0:
        return []
    rows = conn.execute(
        f"""SELECT DISTINCT {SYMBOL_COLUMNS}
            FROM refs AS r
            JOIN symbols AS s ON s.id = r.from_symbol_id
            JOIN files AS f ON f.id = s.file_id
            WHERE r.to_symbol_id = ? AND s.id <> ?
            ORDER BY f.path, s.start_line, s.id
            LIMIT ?""",
        (symbol_id, symbol_id, limit),
    ).fetchall()
    return [row_to_symbol(row) for row in rows]


def callees(
    conn: sqlite3.Connection, symbol_id: int, limit: int = 50
) -> list[Symbol]:
    """Symbols referenced by ``symbol_id``, ordered by file and line."""
    if limit <= 0:
        return []
    rows = conn.execute(
        f"""SELECT DISTINCT {SYMBOL_COLUMNS}
            FROM refs AS r
            JOIN symbols AS s ON s.id = r.to_symbol_id
            JOIN files AS f ON f.id = s.file_id
            WHERE r.from_symbol_id = ? AND s.id <> ?
            ORDER BY f.path, s.start_line, s.id
            LIMIT ?""",
        (symbol_id, symbol_id, limit),
    ).fetchall()
    return [row_to_symbol(row) for row in rows]


def find_symbol(
    conn: sqlite3.Connection, file: str, name: str
) -> Symbol | None:
    """Find a symbol by repo-relative file and name.

    ``qualname`` is matched first (so ``Class.method`` works), then the bare
    ``name``. The first definition in source order wins when a file repeats a
    name.
    """
    path = normalize_path(file)
    row = conn.execute(
        f"""SELECT {SYMBOL_COLUMNS}
            FROM symbols AS s
            JOIN files AS f ON f.id = s.file_id
            WHERE f.path = ? AND s.qualname = ?
            ORDER BY s.start_line, s.id
            LIMIT 1""",
        (path, name),
    ).fetchone()
    if row is not None:
        return row_to_symbol(row)

    row = conn.execute(
        f"""SELECT {SYMBOL_COLUMNS}
            FROM symbols AS s
            JOIN files AS f ON f.id = s.file_id
            WHERE f.path = ? AND s.name = ?
            ORDER BY s.start_line, s.id
            LIMIT 1""",
        (path, name),
    ).fetchone()
    return row_to_symbol(row) if row is not None else None
