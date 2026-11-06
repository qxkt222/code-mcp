"""Full-text search over indexed symbols and comments.

User input is tokenised before it reaches FTS5, so quotes, ``*``, ``-``,
parentheses and colons in a query can never change the MATCH expression.
"""

from __future__ import annotations

import re
import sqlite3

from ..types import Symbol

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MAX_TOKENS = 32
_MAX_LIMIT = 200

_SYMBOL_COLUMNS = (
    "s.id, s.file_id, f.path AS file, s.name, s.qualname, s.kind, "
    "s.start_line, s.end_line, s.start_byte, s.end_byte, s.signature, s.docstring"
)


def fts_escape(query: str) -> str:
    """Turn arbitrary user input into a safe FTS5 MATCH expression.

    Every word becomes a quoted prefix query; words are OR-ed so a multi-word
    query still finds symbols that match only one of them. Returns ``""`` when
    the input contains nothing searchable.
    """
    if not query:
        return ""
    tokens = [
        token
        for token in _WORD_RE.findall(query)
        if any(char.isalnum() for char in token)
    ]
    if not tokens:
        return ""
    # A single stray letter ("a file") would prefix-match far too much; keep
    # one-letter terms only when the whole query is that short.
    if any(len(token) > 1 for token in tokens):
        tokens = [token for token in tokens if len(token) > 1]
    tokens = tokens[:_MAX_TOKENS]
    return " OR ".join(f'"{token}"*' for token in tokens)


def search_symbols(
    conn: sqlite3.Connection,
    query: str,
    kinds: list[str] | None = None,
    limit: int = 20,
) -> list[Symbol]:
    """Search symbol names, signatures and docstrings, plus body comments.

    Comment hits are mapped to the innermost symbol containing them, then
    merged with direct symbol hits by BM25 rank and de-duplicated by id.
    """
    match = fts_escape(query)
    limit = _clamp(limit)
    if not match or limit == 0:
        return []

    kind_sql, kind_params = _kind_filter(kinds)
    pool = max(limit * 4, 50)

    symbol_rows = conn.execute(
        f"SELECT {_SYMBOL_COLUMNS}, bm25(symbols_fts) AS rank "
        "FROM symbols_fts "
        "JOIN symbols s ON s.id = symbols_fts.rowid "
        "JOIN files f ON f.id = s.file_id "
        "WHERE symbols_fts MATCH ? AND s.kind != 'module' "
        f"{kind_sql} ORDER BY rank LIMIT ?",
        [match, *kind_params, pool],
    ).fetchall()

    best: dict[int, tuple[float, Symbol]] = {}
    for row in symbol_rows:
        symbol = _symbol(row)
        best[symbol.id] = (float(row["rank"]), symbol)

    comment_rows = conn.execute(
        "SELECT c.file_id, c.line, bm25(comments_fts) AS rank "
        "FROM comments_fts "
        "JOIN comments c ON c.id = comments_fts.rowid "
        "WHERE comments_fts MATCH ? ORDER BY rank LIMIT ?",
        (match, pool),
    ).fetchall()

    for row in comment_rows:
        symbol_row = conn.execute(
            f"SELECT {_SYMBOL_COLUMNS} "
            "FROM symbols s "
            "JOIN files f ON f.id = s.file_id "
            "WHERE s.file_id = ? AND s.start_line <= ? AND s.end_line >= ? "
            "AND s.kind != 'module' "
            f"{kind_sql} "
            "ORDER BY (s.end_line - s.start_line) ASC, s.start_line DESC LIMIT 1",
            [row["file_id"], row["line"], row["line"], *kind_params],
        ).fetchone()
        if symbol_row is None:
            continue
        symbol = _symbol(symbol_row)
        rank = float(row["rank"])
        previous = best.get(symbol.id)
        if previous is None or rank < previous[0]:
            best[symbol.id] = (rank, symbol)

    ordered = sorted(best.values(), key=lambda item: (item[0], item[1].qualname, item[1].file))
    return [symbol for _rank, symbol in ordered[:limit]]


def search_comments(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[tuple[str, int, str]]:
    """Search comment text and return ``(file, line, text)``, best match first."""
    match = fts_escape(query)
    limit = _clamp(limit)
    if not match or limit == 0:
        return []
    rows = conn.execute(
        "SELECT f.path AS file, c.line AS line, c.text AS text, bm25(comments_fts) AS rank "
        "FROM comments_fts "
        "JOIN comments c ON c.id = comments_fts.rowid "
        "JOIN files f ON f.id = c.file_id "
        "WHERE comments_fts MATCH ? ORDER BY rank LIMIT ?",
        (match, limit),
    ).fetchall()
    return [(row["file"], int(row["line"]), row["text"]) for row in rows]


def _clamp(limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = 20
    return max(0, min(value, _MAX_LIMIT))


def _kind_filter(kinds: list[str] | None) -> tuple[str, list[str]]:
    if not kinds:
        return "", []
    placeholders = ", ".join("?" for _ in kinds)
    return f" AND s.kind IN ({placeholders})", list(kinds)


def _symbol(row: sqlite3.Row) -> Symbol:
    return Symbol(
        id=int(row["id"]),
        file_id=int(row["file_id"]),
        file=row["file"],
        name=row["name"],
        qualname=row["qualname"],
        kind=row["kind"],
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
        start_byte=int(row["start_byte"]),
        end_byte=int(row["end_byte"]),
        signature=row["signature"],
        docstring=row["docstring"],
    )
