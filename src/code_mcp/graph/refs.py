"""Reference resolution and adjacency queries over the SQLite index.

Everything here is schema-driven: it works no matter which indexer produced the
rows. Resolution is deliberately conservative and evidence-based:

* a ref only crosses files when the referring file imports the target file;
* a bare name that is merely unique somewhere in the repository is not enough;
* symbols in different language families are never matched -- Tauri
  Rust/TypeScript edges are created separately by :mod:`graph.tauri`.

Ambiguous names are left unresolved rather than guessed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..index.modules import ModuleResolver
from ..types import CONF_EXACT, Symbol

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


_WEB_LANGS = frozenset({"typescript", "tsx", "javascript"})


def _family(lang: str) -> str:
    """Collapse the JS dialects into one family; everything else is distinct."""
    return "web" if lang in _WEB_LANGS else (lang or "")


def _compatible(from_lang: str, to_lang: str) -> bool:
    if not from_lang or not to_lang:
        # Unknown languages cannot be proven different, so do not reject.
        return True
    return _family(from_lang) == _family(to_lang)


def _repo_root(conn: sqlite3.Connection) -> Path | None:
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'root'"
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    root = Path(row["value"])
    return root if root.is_dir() else None


def _imported_files(
    conn: sqlite3.Connection,
    files: dict[str, tuple[int, str]],
    by_id: dict[int, str],
) -> dict[int, set[int]]:
    """Files each indexed file imports, from stored or freshly resolved rows."""
    root = _repo_root(conn)
    # ``root`` is only needed for tsconfig aliases; Python, Rust and relative
    # web specifiers resolve from the path snapshot alone.
    resolver = ModuleResolver(files, root) if files else None

    out: dict[int, set[int]] = {}
    for row in conn.execute(
        "SELECT file_id, module, resolved_file_id FROM imports"
    ):
        file_id = int(row["file_id"])
        resolved = row["resolved_file_id"]
        if resolved is not None:
            out.setdefault(file_id, set()).add(int(resolved))
            continue
        if resolver is None:
            continue
        from_path = by_id.get(file_id)
        if from_path is None:
            continue
        target = resolver.resolve(from_path, row["module"])
        if target is not None:
            out.setdefault(file_id, set()).add(files[target][0])
    return out


def _resolve_target(
    to_name: str,
    from_file: int,
    lang_by_file: dict[int, str],
    by_name: dict[str, list[tuple[int, int]]],
    by_qualname: dict[str, list[tuple[int, int]]],
    imported: dict[int, set[int]],
) -> int | None:
    """Pick one symbol id for a ref, or None when the evidence is insufficient.

    Order: language family, then a unique same-file definition, then a unique
    cross-file candidate whose file the referring file imports. Anything else
    (globally unique names included) stays unresolved.
    """
    matches = by_name.get(to_name)
    if not matches and is_qualified(to_name):
        matches = by_qualname.get(to_name)
    if not matches:
        return None

    from_lang = lang_by_file.get(from_file, "")
    candidates = [
        (symbol_id, file_id)
        for symbol_id, file_id in matches
        if _compatible(from_lang, lang_by_file.get(file_id, ""))
    ]
    if not candidates:
        return None

    local = [symbol_id for symbol_id, file_id in candidates if file_id == from_file]
    if len(local) == 1:
        return local[0]
    if local:
        # Several same-file definitions with this name: never guess.
        return None

    targets = imported.get(from_file, set())
    cross_file = [
        symbol_id for symbol_id, file_id in candidates if file_id in targets
    ]
    return cross_file[0] if len(cross_file) == 1 else None


def resolve_refs(conn: sqlite3.Connection) -> int:
    """Backfill ``refs.to_symbol_id`` from local or import-backed evidence.

    Only rows with ``to_symbol_id IS NULL`` are considered, so existing edges
    (including Tauri bridges) are never rewritten. A ref is resolved when the
    target is the only same-file definition of that name, or when it is the
    only candidate in a file the referring file imports. Cross-language names
    never match. Returns the number of refs updated.
    """
    rows = conn.execute(
        """SELECT r.id AS id, r.to_name AS to_name, s.file_id AS from_file_id
           FROM refs AS r
           JOIN symbols AS s ON s.id = r.from_symbol_id
           WHERE r.to_symbol_id IS NULL
           ORDER BY r.id"""
    ).fetchall()
    if not rows:
        return 0

    files: dict[str, tuple[int, str]] = {
        row["path"]: (int(row["id"]), row["lang"] or "")
        for row in conn.execute("SELECT id, path, lang FROM files")
    }
    by_id: dict[int, str] = {}
    lang_by_file: dict[int, str] = {}
    for path, (file_id, lang) in files.items():
        by_id[file_id] = path
        lang_by_file[file_id] = lang

    by_name: dict[str, list[tuple[int, int]]] = {}
    by_qualname: dict[str, list[tuple[int, int]]] = {}
    for symbol in conn.execute("SELECT id, name, qualname, file_id FROM symbols"):
        entry = (int(symbol["id"]), int(symbol["file_id"]))
        by_name.setdefault(symbol["name"], []).append(entry)
        by_qualname.setdefault(symbol["qualname"], []).append(entry)

    imported = _imported_files(conn, files, by_id)

    updates: list[tuple[int, str, int]] = []
    for row in rows:
        target = _resolve_target(
            row["to_name"],
            int(row["from_file_id"]),
            lang_by_file,
            by_name,
            by_qualname,
            imported,
        )
        if target is not None:
            updates.append((target, CONF_EXACT, int(row["id"])))

    if updates:
        conn.executemany(
            "UPDATE refs SET to_symbol_id = ?, confidence = ? WHERE id = ?",
            updates,
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
