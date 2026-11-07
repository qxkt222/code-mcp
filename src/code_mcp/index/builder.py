"""Incremental index builder.

The frozen ``files`` schema has no size column, so the byte size of each file
is mirrored in the ``meta`` table under ``file_sizes``. That keeps the
``(mtime, size)`` skip rule exact without touching the schema.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..db import get_meta, set_meta
from .parsers import ParsedFile, detect_language, parse_source

_SIZES_META_KEY = "file_sizes"

# Generated bundles and minified sources parse into one giant anonymous symbol
# and poison both the index and the token baseline, so they are skipped.
_MINIFIED_SUFFIXES = (".min.js", ".min.mjs", ".min.css", ".bundle.js", "-min.js")
_MAX_FILE_BYTES = 1_500_000
_MINIFIED_SIZE_THRESHOLD = 200_000
_MINIFIED_AVG_LINE = 800


def _lastrowid(cursor: sqlite3.Cursor) -> int:
    """INSERT 之后的 rowid 必然存在；把 ``int | None`` 收窄成 ``int``。"""
    rowid = cursor.lastrowid
    assert rowid is not None, "INSERT 之后 cursor.lastrowid 不应为 None"
    return rowid


@dataclass
class IndexStats:
    files_indexed: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    symbols: int = 0
    refs: int = 0
    duration_s: float = 0.0


@dataclass
class _Candidate:
    rel: str
    path: Path
    lang: str
    mtime: float
    size: int


def build_index(
    root: Path,
    conn: sqlite3.Connection,
    config: Config,
    force: bool = False,
) -> IndexStats:
    """Index every supported file under ``root``, reusing unchanged rows."""
    started = time.perf_counter()
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")

    conn.execute("PRAGMA foreign_keys = ON")

    existing = {
        row["path"]: row
        for row in conn.execute("SELECT id, path, mtime FROM files")
    }
    sizes = _load_sizes(conn)

    stats = IndexStats()
    seen: set[str] = set()
    current_sizes: dict[str, int] = {}

    for candidate in _scan(root, config):
        seen.add(candidate.rel)
        current_sizes[candidate.rel] = candidate.size

        previous = existing.get(candidate.rel)
        unchanged = (
            not force
            and previous is not None
            and float(previous["mtime"]) == candidate.mtime
            and sizes.get(candidate.rel) == candidate.size
        )
        if unchanged:
            stats.files_skipped += 1
            continue

        try:
            source = candidate.path.read_bytes()
        except OSError:
            # Do not record the size, so the next run retries the file.
            current_sizes.pop(candidate.rel, None)
            stats.files_skipped += 1
            continue

        parsed = parse_source(source, candidate.lang)
        _replace_file(conn, candidate, source, parsed)
        stats.files_indexed += 1
        stats.symbols += len(parsed.symbols)
        stats.refs += len(parsed.refs)

    for rel in sorted(set(existing) - seen):
        conn.execute("DELETE FROM files WHERE path = ?", (rel,))
        stats.files_removed += 1

    set_meta(conn, _SIZES_META_KEY, json.dumps(current_sizes, sort_keys=True))
    conn.commit()
    stats.duration_s = time.perf_counter() - started
    return stats


def _scan(root: Path, config: Config) -> list[_Candidate]:
    """Collect indexable files.

    Inside a git repository ``git ls-files`` is the source of truth: it is orders
    of magnitude faster than walking a tree that may contain vendored toolchains
    (one real project had 146k untracked files under a bundled directory), and it
    already respects ignore rules.
    """
    listed = _git_files(root)
    if listed:
        return _build_candidates(root, config, listed)
    # An empty listing also happens when ``root`` merely sits inside some
    # unrelated repository (a home directory that happens to be one), so fall
    # back to walking rather than reporting an empty index.
    return _build_candidates(root, config, _walk_files(root, config))


def _build_candidates(root: Path, config: Config, relatives: list[str]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for rel in relatives:
        if config.is_excluded(rel):
            continue
        path = root / rel
        lang = detect_language(path)
        if lang is None:
            continue
        if not _language_allowed(lang, config.languages):
            continue
        if _is_generated(rel, path):
            continue
        try:
            if not path.is_file():
                continue
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size > _MAX_FILE_BYTES:
            continue
        candidates.append(
            _Candidate(
                rel=rel,
                path=path,
                lang=lang,
                mtime=stat.st_mtime,
                size=stat.st_size,
            )
        )
    return candidates


def _walk_files(root: Path, config: Config) -> list[str]:
    relatives: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        rel_dir = current.relative_to(root).as_posix()
        prefix = "" if rel_dir == "." else f"{rel_dir}/"
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in config.excludes
            and not config.is_excluded(f"{prefix}{name}")
        )
        for filename in sorted(filenames):
            relatives.append(f"{prefix}{filename}")
    return relatives


def _git_files(root: Path) -> list[str] | None:
    """Repository-relative paths git knows about, or None when not a repository."""
    try:
        prefix_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-prefix"],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if prefix_result.returncode != 0:
            return None
        subdir = prefix_result.stdout.decode("utf-8", "replace").strip()

        listing = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )
        if listing.returncode != 0:
            return None
    except (OSError, subprocess.SubprocessError):
        return None

    paths = [part for part in listing.stdout.decode("utf-8", "replace").split("\0") if part]
    if not subdir:
        return paths
    trimmed = []
    for path in paths:
        if path.startswith(subdir):
            trimmed.append(path[len(subdir) :])
    return trimmed


def _is_generated(rel: str, path: Path) -> bool:
    """Skip bundled or minified sources before they reach the parser."""
    lowered = rel.lower()
    if lowered.endswith(_MINIFIED_SUFFIXES):
        return True
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= _MINIFIED_SIZE_THRESHOLD:
        return False
    try:
        data = path.read_bytes()
    except OSError:
        return False
    lines = data.count(b"\n") + 1
    return lines > 0 and size / lines > _MINIFIED_AVG_LINE




def _language_allowed(lang: str, configured: list[str]) -> bool:
    if not configured:
        return True
    if lang in configured:
        return True
    if lang in ("tsx", "javascript") and "typescript" in configured:
        return True
    return lang == "tsx" and "javascript" in configured


def _load_sizes(conn: sqlite3.Connection) -> dict[str, int]:
    raw = get_meta(conn, _SIZES_META_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): int(value) for key, value in data.items() if isinstance(value, int)}


def reindex_file(root: Path, conn: sqlite3.Connection, rel: str) -> bool:
    """Re-parse a single file and replace its rows.

    Called after a surgical edit: the write changes the byte offsets of every
    later symbol in the file, so refreshing only the file hash would leave the
    index pointing at the wrong ranges.
    """
    root = Path(root).resolve()
    path = root / rel
    try:
        source = path.read_bytes()
        stat = path.stat()
    except OSError:
        return False

    lang = detect_language(path)
    if lang is None:
        return False

    candidate = _Candidate(
        rel=rel,
        path=path,
        lang=lang,
        mtime=stat.st_mtime,
        size=stat.st_size,
    )
    parsed = parse_source(source, lang)
    _replace_file(conn, candidate, source, parsed)

    sizes = _load_sizes(conn)
    sizes[rel] = stat.st_size
    set_meta(conn, _SIZES_META_KEY, json.dumps(sizes, sort_keys=True))
    conn.commit()
    return True


def _replace_file(
    conn: sqlite3.Connection,
    candidate: _Candidate,
    source: bytes,
    parsed: ParsedFile,
) -> None:
    # Replace wholesale: deleting the file row cascades to symbols, refs,
    # imports and comments, and the FTS triggers stay consistent.
    conn.execute("DELETE FROM files WHERE path = ?", (candidate.rel,))
    cursor = conn.execute(
        "INSERT INTO files(path, lang, mtime, hash, loc) VALUES (?, ?, ?, ?, ?)",
        (
            candidate.rel,
            candidate.lang,
            candidate.mtime,
            hashlib.sha256(source).hexdigest(),
            parsed.loc,
        ),
    )
    file_id = _lastrowid(cursor)

    symbol_ids: dict[str, int] = {}
    for symbol in parsed.symbols:
        cursor = conn.execute(
            "INSERT INTO symbols("
            "file_id, name, qualname, kind, start_line, end_line, "
            "start_byte, end_byte, signature, docstring"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                file_id,
                symbol.name,
                symbol.qualname,
                symbol.kind,
                symbol.start_line,
                symbol.end_line,
                symbol.start_byte,
                symbol.end_byte,
                symbol.signature,
                symbol.docstring,
            ),
        )
        symbol_ids.setdefault(symbol.qualname, _lastrowid(cursor))

    module_id: int | None = None
    for ref in parsed.refs:
        from_id = symbol_ids.get(ref.from_qualname) if ref.from_qualname else None
        if from_id is None:
            if module_id is None:
                module_id = _insert_module_symbol(conn, file_id, candidate.rel, parsed.loc, len(source))
            from_id = module_id
        conn.execute(
            "INSERT INTO refs(from_symbol_id, to_symbol_id, to_name, kind, confidence) "
            "VALUES (?, NULL, ?, ?, ?)",
            (from_id, ref.to_name, ref.kind, ref.confidence),
        )

    if parsed.imports:
        conn.executemany(
            "INSERT INTO imports(file_id, module, alias, resolved_file_id) VALUES (?, ?, ?, NULL)",
            [(file_id, module, alias) for module, alias in parsed.imports],
        )
    if parsed.comments:
        conn.executemany(
            "INSERT INTO comments(file_id, line, text) VALUES (?, ?, ?)",
            [(file_id, line, text) for line, text in parsed.comments],
        )


def _insert_module_symbol(
    conn: sqlite3.Connection,
    file_id: int,
    rel: str,
    loc: int,
    byte_length: int,
) -> int:
    """Anchor for module-level refs (imports, top-level calls).

    ``refs.from_symbol_id`` is NOT NULL, so a file-scope symbol is required.
    It is zero-width and parked at end-of-file: choosing the first top-level
    symbol would misattribute imports, and spanning the whole file would make
    ``pack``'s "first top-level symbol" default pick the module instead.
    """
    name = Path(rel).stem or Path(rel).name or rel
    line = max(1, loc)
    cursor = conn.execute(
        "INSERT INTO symbols("
        "file_id, name, qualname, kind, start_line, end_line, "
        "start_byte, end_byte, signature, docstring"
        ") VALUES (?, ?, '', 'module', ?, ?, ?, ?, ?, '')",
        (file_id, name, line, line, byte_length, byte_length, rel),
    )
    return _lastrowid(cursor)
