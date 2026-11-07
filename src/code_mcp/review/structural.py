"""Structural review rules computed from the index plus the files it describes.

Only the SQLite index and the source files referenced by it are read. Nesting
depth is approximated from the indentation of control-keyword lines only, with
the indent unit inferred per file (two spaces in TS/JS, four in Python); no
cyclomatic-complexity dependency is used.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
import sqlite3
import textwrap
from pathlib import Path

from ..paths import is_test_path
from ..types import REF_CALL, Finding

# UI components legitimately run long, so the limit is per language. Only a
# function more than twice the limit is treated as P1.
LONG_FUNCTION_LINES: dict[str, int] = {
    "python": 80,
    "rust": 80,
    "typescript": 150,
    "tsx": 250,
    "javascript": 150,
}
DEFAULT_LONG_FUNCTION_LINES = 80
DEEP_NESTING_LEVELS = 5
HIGH_FANOUT = 15
HIGH_FANIN = 20
DUPLICATE_MIN_LINES = 10

_FUNCTION_KINDS = ("function", "method")
_SYMBOL_KINDS = ("function", "method", "class")
# Test discovery for JS/TS has no reliable signal yet, so the rule is skipped
# there rather than reporting false positives.
_NO_TESTS_SKIP_LANGS = {"typescript", "tsx", "javascript"}
_CONTROL = re.compile(
    r"^(?:if|elif|else|for|while|match|loop|switch|case|default|"
    r"try|catch|finally|except|with|do)\b"
)

_SYMBOL_QUERY = (
    "SELECT s.id, s.name, s.qualname, s.kind, s.start_line, s.end_line, "
    "       s.start_byte, s.end_byte, s.docstring, f.path AS file, f.lang AS lang "
    "FROM symbols s JOIN files f ON f.id = s.file_id "
)


def analyze(conn: sqlite3.Connection, files: list[str] | None = None) -> list[Finding]:
    """Return every structural finding for the indexed symbols.

    ``files`` limits the analysis to the given repo-relative paths. Test-file
    detection for the ``no-tests`` rule always looks at the whole index.
    """
    symbols = _load_symbols(conn, files)
    root = _index_root(conn)
    cache: dict[str, bytes | None] = {}

    findings: list[Finding] = []
    findings.extend(_long_functions(symbols))
    findings.extend(_deep_nesting(symbols, root, cache))
    findings.extend(_high_fanout(conn, symbols))
    findings.extend(_high_fanin(conn, symbols))
    findings.extend(_duplicate_bodies(symbols, root, cache))
    findings.extend(_missing_docstrings(symbols))
    findings.extend(_no_tests(conn, symbols))
    return findings


def _load_symbols(conn: sqlite3.Connection, files: list[str] | None) -> list[sqlite3.Row]:
    if not files:
        rows = conn.execute(_SYMBOL_QUERY + "ORDER BY f.path, s.start_byte").fetchall()
        return list(rows)

    wanted = [entry.replace("\\", "/") for entry in files]
    placeholders = ",".join("?" for _ in wanted)
    rows = conn.execute(
        _SYMBOL_QUERY + f"WHERE f.path IN ({placeholders}) ORDER BY f.path, s.start_byte",
        wanted,
    ).fetchall()
    return list(rows)


def _index_root(conn: sqlite3.Connection) -> Path | None:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'root'").fetchone()
    except sqlite3.OperationalError:
        return None
    return Path(row["value"]) if row else None


def _long_functions(symbols: list[sqlite3.Row]) -> list[Finding]:
    findings: list[Finding] = []
    for symbol in symbols:
        if symbol["kind"] not in _FUNCTION_KINDS:
            continue
        limit = LONG_FUNCTION_LINES.get(symbol["lang"] or "", DEFAULT_LONG_FUNCTION_LINES)
        length = int(symbol["end_line"]) - int(symbol["start_line"]) + 1
        if length <= limit:
            continue
        findings.append(
            Finding(
                rule="long-function",
                severity="P1" if length > 2 * limit else "P2",
                file=symbol["file"],
                line=int(symbol["start_line"]),
                message=f"{_label(symbol)} is {length} lines long (limit {limit})",
                symbol=_label(symbol),
            )
        )
    return findings


def _deep_nesting(
    symbols: list[sqlite3.Row], root: Path | None, cache: dict[str, bytes | None]
) -> list[Finding]:
    findings: list[Finding] = []
    for symbol in symbols:
        if symbol["kind"] not in _FUNCTION_KINDS:
            continue
        text = _symbol_text(symbol, root, cache)
        if text is None:
            continue
        depth = _max_control_depth(text)
        if depth <= DEEP_NESTING_LEVELS:
            continue
        findings.append(
            Finding(
                rule="deep-nesting",
                severity="P1",
                file=symbol["file"],
                line=int(symbol["start_line"]),
                message=f"{_label(symbol)} nests to depth {depth} (limit {DEEP_NESTING_LEVELS})",
                symbol=_label(symbol),
            )
        )
    return findings


def _high_fanout(conn: sqlite3.Connection, symbols: list[sqlite3.Row]) -> list[Finding]:
    counts = {
        int(row["from_symbol_id"]): int(row["n"])
        for row in conn.execute(
            "SELECT from_symbol_id, COUNT(DISTINCT to_name) AS n "
            "FROM refs WHERE kind = ? GROUP BY from_symbol_id",
            (REF_CALL,),
        )
    }
    findings: list[Finding] = []
    for symbol in symbols:
        count = counts.get(int(symbol["id"]), 0)
        if count <= HIGH_FANOUT:
            continue
        findings.append(
            Finding(
                rule="high-fanout",
                severity="P2",
                file=symbol["file"],
                line=int(symbol["start_line"]),
                message=f"{_label(symbol)} calls {count} distinct symbols (limit {HIGH_FANOUT})",
                symbol=_label(symbol),
            )
        )
    return findings


def _high_fanin(conn: sqlite3.Connection, symbols: list[sqlite3.Row]) -> list[Finding]:
    counts = {
        int(row["to_symbol_id"]): int(row["n"])
        for row in conn.execute(
            "SELECT to_symbol_id, COUNT(DISTINCT from_symbol_id) AS n "
            "FROM refs WHERE to_symbol_id IS NOT NULL GROUP BY to_symbol_id"
        )
    }
    findings: list[Finding] = []
    for symbol in symbols:
        count = counts.get(int(symbol["id"]), 0)
        if count <= HIGH_FANIN:
            continue
        findings.append(
            Finding(
                rule="high-fanin",
                severity="P2",
                file=symbol["file"],
                line=int(symbol["start_line"]),
                message=f"{_label(symbol)} has {count} callers (limit {HIGH_FANIN})",
                symbol=_label(symbol),
            )
        )
    return findings


def _duplicate_bodies(
    symbols: list[sqlite3.Row], root: Path | None, cache: dict[str, bytes | None]
) -> list[Finding]:
    groups: dict[str, list[tuple[sqlite3.Row, int]]] = {}
    for symbol in symbols:
        if symbol["kind"] not in _FUNCTION_KINDS:
            continue
        text = _symbol_text(symbol, root, cache)
        if text is None:
            continue
        body = _body_text(text)
        if not body:
            continue
        line_count = sum(1 for line in body.splitlines() if line.strip())
        if line_count <= DUPLICATE_MIN_LINES:
            continue
        digest = hashlib.sha256(body.encode("utf-8", errors="replace")).hexdigest()
        groups.setdefault(digest, []).append((symbol, line_count))

    findings: list[Finding] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda item: (item[0]["file"], item[0]["start_byte"]))
        first = ordered[0][0]
        for symbol, _ in ordered[1:]:
            findings.append(
                Finding(
                    rule="duplicate-body",
                    severity="P2",
                    file=symbol["file"],
                    line=int(symbol["start_line"]),
                    message=(
                        f"{_label(symbol)} duplicates {_label(first)} "
                        f"at {first['file']}:{first['start_line']}"
                    ),
                    symbol=_label(symbol),
                )
            )
    return findings


def _missing_docstrings(symbols: list[sqlite3.Row]) -> list[Finding]:
    findings: list[Finding] = []
    for symbol in symbols:
        if symbol["kind"] not in _SYMBOL_KINDS:
            continue
        name = symbol["name"]
        if name.startswith("_") or is_test_path(symbol["file"]):
            continue
        if (symbol["docstring"] or "").strip():
            continue
        findings.append(
            Finding(
                rule="missing-docstring",
                severity="P3",
                file=symbol["file"],
                line=int(symbol["start_line"]),
                message=f"Public {symbol['kind']} '{name}' has no docstring",
                symbol=_label(symbol),
            )
        )
    return findings


def _no_tests(conn: sqlite3.Connection, symbols: list[sqlite3.Row]) -> list[Finding]:
    test_names: set[str] = set()
    rows = conn.execute(
        "SELECT s.name, f.path AS file FROM symbols s JOIN files f ON f.id = s.file_id"
    ).fetchall()
    for row in rows:
        if is_test_path(row["file"]):
            test_names.add(row["name"])

    findings: list[Finding] = []
    for symbol in symbols:
        if symbol["kind"] not in _SYMBOL_KINDS:
            continue
        if symbol["lang"] in _NO_TESTS_SKIP_LANGS:
            continue
        name = symbol["name"]
        if name.startswith("_") or is_test_path(symbol["file"]):
            continue
        candidates = {f"test_{name}", f"{name}_test"}
        if symbol["kind"] == "class":
            candidates.update({f"Test{name}", f"{name}Test"})
        if any(candidate in test_names for candidate in candidates):
            continue
        findings.append(
            Finding(
                rule="no-tests",
                severity="P3",
                file=symbol["file"],
                line=int(symbol["start_line"]),
                message=f"{_label(symbol)} has no matching test",
                symbol=_label(symbol),
            )
        )
    return findings


def _label(symbol: sqlite3.Row) -> str:
    return str(symbol["qualname"] or symbol["name"])


def _symbol_text(
    symbol: sqlite3.Row, root: Path | None, cache: dict[str, bytes | None]
) -> str | None:
    path = symbol["file"]
    data = _file_bytes(root, path, cache)
    if data is None:
        return None
    start = int(symbol["start_byte"])
    end = int(symbol["end_byte"])
    if not 0 <= start <= end <= len(data):
        return None
    return data[start:end].decode("utf-8", errors="replace")


def _file_bytes(
    root: Path | None, path: str, cache: dict[str, bytes | None]
) -> bytes | None:
    if path in cache:
        return cache[path]
    if root is None:
        cache[path] = None
        return None
    try:
        data = (root / path).read_bytes()
    except OSError:
        data = None
    cache[path] = data
    return data


def _indent_unit(lines: list[str]) -> int:
    """Most common positive gap between distinct indent widths; 1..8.

    TS/JS commonly indent by two spaces and Python by four, so a fixed unit
    would under- or over-count one of them by a factor of two. The GCD of the
    gaps between the distinct indentation widths of the file's non-empty lines
    recovers the unit the file actually uses. Tabs already count as four
    columns in ``_indent_width``. With no usable data the default is four.
    """
    widths = sorted({_indent_width(line) for line in lines if line.strip()})
    if len(widths) < 2:
        return 4
    unit = 0
    for left, right in itertools.pairwise(widths):
        unit = math.gcd(unit, right - left)
    if unit <= 0:
        return 4
    return min(8, max(1, unit))


def _max_control_depth(text: str) -> int:
    """Deepest indentation among lines that open a control construct.

    Raw indentation is not nesting: JSX attributes, wrapped call arguments and
    object continuations all sit deep without branching. Requiring a control
    keyword keeps the rule focused on actual control flow. The indent width of
    one level is inferred from the file itself.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines:
        return 0
    base = _indent_width(lines[0])
    unit = _indent_unit(lines)
    deepest = 0
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped or not _CONTROL.match(stripped):
            continue
        deepest = max(deepest, (_indent_width(line) - base) // unit)
    return max(deepest, 0)


def _body_text(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines:
        return ""
    base = _indent_width(lines[0])
    body = [line for line in lines[1:] if line.strip() and _indent_width(line) > base]
    if not body:
        return ""
    return textwrap.dedent("\n".join(body)).strip()


def _indent_width(line: str) -> int:
    width = 0
    for char in line:
        if char == " ":
            width += 1
        elif char == "\t":
            width += 4
        else:
            break
    return width
