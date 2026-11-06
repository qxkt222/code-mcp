"""Budget-aware context packing for one symbol.

The packer answers "what does an agent need to read to work on this symbol?"
with the smallest set of real source slices: the target itself, its callers,
its callees, the definitions it imports, and matching tests. Everything is read
from disk, never reconstructed from index summaries.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import tokens
from .graph.refs import (
    SYMBOL_COLUMNS,
    callees,
    callers,
    find_symbol,
    is_qualified,
    normalize_path,
    row_to_symbol,
)
from .types import FILE_KIND, Pack, Slice, Symbol

# Over-budget sections are emptied in this order (least essential first).
# Tests are the last thing dropped: knowing which tests cover the symbol is
# worth more than a few extra import lines.
_TRIM_ORDER = ("imports", "callees", "callers", "tests")
# Surplus budget expands sections in this order.
_UPGRADE_ORDER = ("callers", "callees", "imports")
_SECTION_LABELS = {
    "tests": "test(s)",
    "callers": "caller(s)",
    "callees": "callee(s)",
    "imports": "import(s)",
}

_FILE_HEADER_LINES = 15

# Ceiling on packed cost as a fraction of the plain-read baseline. Keeps the
# tool from ever being a net loss: below this the caller still saves tokens.
PACK_CAP_RATIO = 0.7

_PY_SUFFIXES = (".py", "/__init__.py")
_WEB_SUFFIXES = ("", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
_RUST_PATTERNS = ("src/{}.rs", "src/{}/mod.rs", "{}.rs", "{}/mod.rs")


# --------------------------------------------------------------------------- #
# Real source text
# --------------------------------------------------------------------------- #


class _Reader:
    """Lazy, cached access to decoded source text and 1-based line lists."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._text: dict[str, str] = {}
        self._lines: dict[str, list[str]] = {}

    def text(self, rel: str) -> str:
        if rel not in self._text:
            try:
                raw = (self.root / rel).read_bytes()
            except OSError:
                self._text[rel] = ""
            else:
                self._text[rel] = raw.decode("utf-8", errors="replace")
        return self._text[rel]

    def lines(self, rel: str) -> list[str]:
        if rel not in self._lines:
            self._lines[rel] = self.text(rel).splitlines()
        return self._lines[rel]

    def slice(self, rel: str, start: int, end: int) -> tuple[int, int, str]:
        """Return the clamped ``(start, end, text)`` for an inclusive range."""
        lines = self.lines(rel)
        if not lines or start < 1 or start > len(lines):
            safe = max(1, start)
            return safe, safe, ""
        start = max(1, start)
        end = min(max(start, end), len(lines))
        return start, end, "\n".join(lines[start - 1 : end])


@dataclass(slots=True)
class _Piece:
    """A candidate slice that can still be expanded or trimmed."""

    symbol: Symbol | None
    file: str
    start_line: int
    end_line: int
    kind: str
    label: str
    full: bool
    text: str = ""
    tokens: int = 0


def _label(symbol: Symbol) -> str:
    return symbol.qualname or symbol.name


def _first_docstring_line(docstring: str) -> str:
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _full_piece(symbol: Symbol, reader: _Reader) -> _Piece:
    start, end, text = reader.slice(symbol.file, symbol.start_line, symbol.end_line)
    return _Piece(
        symbol=symbol,
        file=symbol.file,
        start_line=start,
        end_line=end,
        kind=symbol.kind,
        label=_label(symbol),
        full=True,
        text=text,
        tokens=tokens.estimate(text),
    )


def _declaration_end(symbol: Symbol, lines: list[str]) -> int:
    """Last line of the declaration, located by matching the stored signature.

    Indexed signatures are whitespace-collapsed, so a multi-line ``def`` cannot
    be sliced by line count; the collapsed text is matched against the real
    lines instead.
    """
    start = max(1, symbol.start_line)
    if start > len(lines):
        return start
    ceiling = symbol.end_line if symbol.end_line >= start else len(lines)
    ceiling = min(ceiling, len(lines))

    signature = " ".join(symbol.signature.split())
    if not signature:
        return start

    limit = min(ceiling, start + 39)
    accumulated: list[str] = []
    for line_no in range(start, limit + 1):
        accumulated.append(lines[line_no - 1])
        if signature in " ".join(" ".join(accumulated).split()):
            return line_no

    # Fallback: stop once brackets balance and the declaration terminates.
    depth = 0
    for line_no in range(start, limit + 1):
        line = lines[line_no - 1]
        for char in line:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth = max(0, depth - 1)
        stripped = line.rstrip()
        if depth == 0 and (
            stripped.endswith(":")
            or stripped.endswith(";")
            or stripped.endswith("{")
        ):
            return line_no
    return start


def _signature_piece(symbol: Symbol, reader: _Reader) -> _Piece:
    """Signature line(s) plus the first docstring line, read from disk."""
    lines = reader.lines(symbol.file)
    start = max(1, symbol.start_line)
    if not lines or start > len(lines):
        return _Piece(
            symbol=symbol,
            file=symbol.file,
            start_line=start,
            end_line=start,
            kind=symbol.kind,
            label=_label(symbol),
            full=False,
            text="",
            tokens=0,
        )

    ceiling = symbol.end_line if symbol.end_line >= start else start
    end = min(_declaration_end(symbol, lines), ceiling, len(lines))

    doc_line = _first_docstring_line(symbol.docstring)
    if doc_line:
        limit = min(end + 3, ceiling, len(lines))
        for line_no in range(end + 1, limit + 1):
            if doc_line in lines[line_no - 1]:
                end = line_no
                break

    start, end, text = reader.slice(symbol.file, start, end)
    return _Piece(
        symbol=symbol,
        file=symbol.file,
        start_line=start,
        end_line=end,
        kind=symbol.kind,
        label=_label(symbol),
        full=end >= ceiling,
        text=text,
        tokens=tokens.estimate(text),
    )


def _file_piece(rel: str, reader: _Reader) -> _Piece:
    """Header of a file used when an import names no resolvable symbol."""
    lines = reader.lines(rel)
    start, end, text = reader.slice(rel, 1, min(_FILE_HEADER_LINES, len(lines) or 1))
    return _Piece(
        symbol=None,
        file=rel,
        start_line=start,
        end_line=end,
        kind=FILE_KIND,
        label=rel,
        full=bool(lines) and end >= len(lines),
        text=text,
        tokens=tokens.estimate(text),
    )


def _expand(piece: _Piece, reader: _Reader) -> _Piece:
    if piece.full:
        return piece
    if piece.symbol is not None:
        return _full_piece(piece.symbol, reader)
    lines = reader.lines(piece.file)
    start, end, text = reader.slice(piece.file, 1, max(1, len(lines)))
    return _Piece(
        symbol=None,
        file=piece.file,
        start_line=start,
        end_line=end,
        kind=piece.kind,
        label=piece.label,
        full=True,
        text=text,
        tokens=tokens.estimate(text),
    )


def _to_slice(piece: _Piece) -> Slice:
    return Slice(
        file=piece.file,
        start_line=piece.start_line,
        end_line=piece.end_line,
        text=piece.text,
        symbol=piece.label,
        kind=piece.kind,
        tokens=piece.tokens,
    )


# --------------------------------------------------------------------------- #
# Target and neighbours
# --------------------------------------------------------------------------- #


def _default_target(
    conn: sqlite3.Connection, rel: str
) -> Symbol | None:
    """First top-level symbol in ``rel``, falling back to the first symbol."""
    rows = conn.execute(
        f"""SELECT {SYMBOL_COLUMNS}
            FROM symbols AS s
            JOIN files AS f ON f.id = s.file_id
            WHERE f.path = ?
            ORDER BY s.start_line, s.id""",
        (rel,),
    ).fetchall()
    if not rows:
        return None
    for row in rows:
        symbol = row_to_symbol(row)
        if not is_qualified(symbol.qualname):
            return symbol
    return row_to_symbol(rows[0])


def _posix_join(base: str, rel: str) -> str:
    stack = [part for part in base.replace("\\", "/").split("/") if part]
    for part in rel.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if stack:
                stack.pop()
        else:
            stack.append(part)
    return "/".join(stack)


def _resolve_module(
    files: dict[str, tuple[int, str]], from_path: str, module: str, lang: str
) -> str | None:
    """Best-effort mapping from an import specifier to an indexed file path."""
    module = module.strip().strip("'\"")
    if not module:
        return None

    directory = from_path.rsplit("/", 1)[0] if "/" in from_path else ""
    candidates: list[str] = []

    if lang == "python":
        if module.startswith("."):
            dots = len(module) - len(module.lstrip("."))
            rest = module[dots:]
            base = directory
            for _ in range(max(0, dots - 1)):
                base = base.rsplit("/", 1)[0] if "/" in base else ""
            stem = _posix_join(base, rest.replace(".", "/")) if rest else base
            candidates.extend(stem + suffix for suffix in _PY_SUFFIXES)
        else:
            dotted = module.replace(".", "/")
            candidates.extend(dotted + suffix for suffix in _PY_SUFFIXES)
    elif lang in ("typescript", "tsx", "javascript"):
        if module.startswith("."):
            stem = _posix_join(directory, module)
            for suffix in _WEB_SUFFIXES:
                candidates.append(stem + suffix)
                if suffix:
                    candidates.append(f"{stem}/index{suffix}")
        elif module.startswith("/"):
            stem = module.lstrip("/")
            candidates.extend(stem + suffix for suffix in _WEB_SUFFIXES)
    else:
        if "::" in module or module in ("crate", "self", "super"):
            parts = [
                part
                for part in module.split("::")
                if part and part not in ("crate", "self", "super")
            ]
            if parts:
                joined = "/".join(parts)
                candidates.extend(pattern.format(joined) for pattern in _RUST_PATTERNS)
        elif "/" in module:
            candidates.append(module)

    cleaned = [candidate.replace("\\", "/").lstrip("/") for candidate in candidates]
    for candidate in cleaned:
        if candidate in files:
            return candidate
    for candidate in cleaned:
        if not candidate:
            continue
        suffix = "/" + candidate
        matches = sorted(path for path in files if path.endswith(suffix))
        if matches:
            return matches[0]
    return None


def _last_name(specifier: str) -> str:
    text = specifier.strip().strip("'\"")
    for separator in ("::", "/", "."):
        if separator in text:
            text = text.split(separator)[-1]
    return text.strip()


def _import_symbol(
    conn: sqlite3.Connection, file_id: int, module: str, alias: str
) -> Symbol | None:
    candidates: list[str] = []
    for raw in (module, alias):
        name = _last_name(raw)
        if name and name not in candidates:
            candidates.append(name)
    for name in candidates:
        row = conn.execute(
            f"""SELECT {SYMBOL_COLUMNS}
                FROM symbols AS s
                JOIN files AS f ON f.id = s.file_id
                WHERE s.file_id = ? AND s.name = ?
                ORDER BY s.start_line, s.id
                LIMIT 1""",
            (file_id, name),
        ).fetchone()
        if row is not None:
            return row_to_symbol(row)
    return None


def _import_pieces(
    conn: sqlite3.Connection,
    rel: str,
    reader: _Reader,
    taken: set[int],
) -> list[_Piece]:
    file_row = conn.execute(
        "SELECT id, lang FROM files WHERE path = ?", (rel,)
    ).fetchone()
    if file_row is None:
        return []

    files: dict[str, tuple[int, str]] = {}
    for row in conn.execute("SELECT id, path, lang FROM files"):
        files[row["path"]] = (int(row["id"]), row["lang"])
    by_id = {file_id: path for path, (file_id, _lang) in files.items()}

    import_rows = conn.execute(
        """SELECT module, alias, resolved_file_id
           FROM imports WHERE file_id = ? ORDER BY id""",
        (int(file_row["id"]),),
    ).fetchall()

    pieces: list[_Piece] = []
    seen_files: set[int] = set()
    for imp in import_rows:
        resolved = imp["resolved_file_id"]
        target_path = by_id.get(int(resolved)) if resolved is not None else None
        if target_path is None:
            target_path = _resolve_module(files, rel, imp["module"], file_row["lang"])
        if target_path is None:
            continue

        file_id, _lang = files[target_path]
        symbol = _import_symbol(conn, file_id, imp["module"], imp["alias"])
        if symbol is not None:
            if symbol.id in taken:
                continue
            taken.add(symbol.id)
            pieces.append(_signature_piece(symbol, reader))
        elif file_id not in seen_files:
            seen_files.add(file_id)
            pieces.append(_file_piece(target_path, reader))
    return pieces


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

_TEST_SUFFIXES = (
    ".test.ts",
    ".test.tsx",
    ".test.js",
    ".test.jsx",
    ".spec.ts",
    ".spec.tsx",
    ".spec.js",
    ".spec.jsx",
)


def _is_test_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    if name.startswith("test_") and name.endswith(".py"):
        return True
    if name.endswith("_test.py"):
        return True
    if name.endswith(_TEST_SUFFIXES):
        return True
    return "/tests/" in f"/{path}"


def _test_variants(name: str, qualname: str) -> set[str]:
    variants: set[str] = set()
    for base in (name, qualname):
        if not base:
            continue
        ident = "_".join(part for part in _split_ident(base) if part)
        if not ident:
            continue
        variants.add(f"test_{ident}")
        variants.add(f"{ident}_test")
    return variants


def _split_ident(value: str) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    for char in value:
        if char.isalnum() or char == "_":
            current.append(char)
        else:
            pieces.append("".join(current))
            current = []
    pieces.append("".join(current))
    return [piece for piece in pieces if piece]


def _name_parts(name: str) -> set[str]:
    return {part for part in _split_ident(name) if part}


def _is_test_match(symbol: Symbol, target: Symbol) -> bool:
    if symbol.name in _test_variants(target.name, target.qualname):
        return True
    if not _is_test_path(symbol.file):
        return False
    if not symbol.name.startswith("test"):
        return False
    wanted = _name_parts(target.name)
    present = set(_split_ident(symbol.name))
    return bool(wanted) and wanted.issubset(present)


def _test_rank(symbol: Symbol, target: Symbol) -> tuple[int, str, int, int]:
    exact = symbol.name in _test_variants(target.name, target.qualname)
    return (0 if exact else 1, symbol.file, symbol.start_line, symbol.id)


def _test_pieces(
    conn: sqlite3.Connection, target: Symbol, taken: set[int], reader: _Reader
) -> list[_Piece]:
    rows = conn.execute(
        f"""SELECT {SYMBOL_COLUMNS}
            FROM symbols AS s
            JOIN files AS f ON f.id = s.file_id
            WHERE s.name LIKE '%test%' OR f.path LIKE '%test%'
            ORDER BY f.path, s.start_line, s.id"""
    ).fetchall()

    matches: list[tuple[tuple[int, str, int, int], Symbol]] = []
    for row in rows:
        symbol = row_to_symbol(row)
        if symbol.id in taken or symbol.id == target.id:
            continue
        if _is_test_match(symbol, target):
            matches.append((_test_rank(symbol, target), symbol))
    matches.sort(key=lambda item: item[0])

    pieces: list[_Piece] = []
    for _rank, symbol in matches:
        if symbol.id in taken:
            continue
        taken.add(symbol.id)
        pieces.append(_full_piece(symbol, reader))
    return pieces


# --------------------------------------------------------------------------- #
# Note and entry point
# --------------------------------------------------------------------------- #


def _build_note(
    total: int,
    baseline: int,
    budget: int,
    trimmed: dict[str, int],
    upgraded: int,
    worth: bool,
) -> str:
    savings = tokens.savings_pct(total, baseline)
    summary = f"packed {total} tokens vs {baseline} baseline"
    if savings > 0:
        summary += f" ({savings}% saved)"
    elif savings < 0:
        summary += f" ({-savings}% larger)"
    else:
        summary += " (no saving)"
    parts = [summary]

    if trimmed:
        detail = ", ".join(
            f"{count} {_SECTION_LABELS[key]}" for key, count in trimmed.items()
        )
        parts.append(f"over budget {budget}: trimmed {detail}")
    if total > budget:
        parts.append(f"target alone exceeds budget {budget}")
    if upgraded:
        parts.append(f"expanded {upgraded} slice(s) to full body")
    if not worth:
        if baseline <= 0:
            parts.append("not recommended: no readable file text")
        elif total >= baseline:
            parts.append(
                "not recommended: packed context is not smaller than the file; "
                "read the file directly instead"
            )
        else:
            parts.append(
                f"not recommended: {savings}% saving is below the 25% threshold; "
                "read the file directly instead"
            )
    return "; ".join(parts)


def pack(
    conn: sqlite3.Connection,
    root: Path,
    file: str,
    name: str | None,
    budget_tokens: int = 30000,
    include_tests: bool = True,
    full_body: bool = True,
) -> Pack:
    """Pack the causal neighbourhood of ``file``:``name`` within a token budget.

    Slices are read from disk. The target always keeps its full body; callers,
    callees and imports start as signature slices and are expanded only when
    the budget allows. When over budget, sections are trimmed in the order
    tests, callers, callees, imports, and the counts land in ``note``.
    """
    rel = normalize_path(file)
    target = find_symbol(conn, rel, name) if name else _default_target(conn, rel)
    if target is None:
        label = f"{rel}:{name}" if name else rel
        return Pack(note=f"target not found: {label}", worth_using=False)

    reader = _Reader(Path(root))
    baseline = tokens.estimate(reader.text(rel))

    taken = {target.id}
    sections: dict[str, list[_Piece]] = {
        "tests": [],
        "callers": [],
        "callees": [],
        "imports": [],
    }

    # Tests are classified first so a test that calls the target is reported
    # under ``tests`` instead of being buried among the callers.
    if include_tests:
        sections["tests"] = _test_pieces(conn, target, taken, reader)

    def usable(symbol: Symbol) -> bool:
        if symbol.id in taken:
            return False
        return include_tests or not _is_test_match(symbol, target)

    for symbol in callers(conn, target.id):
        if not usable(symbol):
            continue
        taken.add(symbol.id)
        sections["callers"].append(_signature_piece(symbol, reader))

    for symbol in callees(conn, target.id):
        if not usable(symbol):
            continue
        taken.add(symbol.id)
        sections["callees"].append(_signature_piece(symbol, reader))

    sections["imports"] = _import_pieces(conn, rel, reader, taken)

    target_piece = _full_piece(target, reader)
    budget = max(0, int(budget_tokens))
    total = target_piece.tokens + sum(
        piece.tokens for items in sections.values() for piece in items
    )

    trimmed: dict[str, int] = {}
    if total > budget:
        for key in _TRIM_ORDER:
            items = sections[key]
            while items and total > budget:
                removed = items.pop()
                total -= removed.tokens
                trimmed[key] = trimmed.get(key, 0) + 1

    # Upgrading a signature to a full body is only worth it while the packed
    # result stays well below the cost of a plain read. Without this cap a
    # small target with many callers expands into something several times more
    # expensive than the file it came from.
    cap = min(budget, int(baseline * PACK_CAP_RATIO)) if baseline > 0 else budget

    upgraded = 0
    if full_body and total < cap:
        for key in _UPGRADE_ORDER:
            items = sections[key]
            for index, piece in enumerate(items):
                if piece.full:
                    continue
                expanded = _expand(piece, reader)
                delta = expanded.tokens - piece.tokens
                if delta <= 0:
                    items[index] = expanded
                    continue
                if total + delta <= cap:
                    items[index] = expanded
                    total += delta
                    upgraded += 1

    target_slice = _to_slice(target_piece)
    caller_slices = [_to_slice(piece) for piece in sections["callers"]]
    callee_slices = [_to_slice(piece) for piece in sections["callees"]]
    import_slices = [_to_slice(piece) for piece in sections["imports"]]
    test_slices = [_to_slice(piece) for piece in sections["tests"]]

    token_total = target_slice.tokens + sum(
        item.tokens
        for group in (caller_slices, callee_slices, import_slices, test_slices)
        for item in group
    )
    worth = tokens.worth_using(token_total, baseline)

    return Pack(
        target=target_slice,
        callers=caller_slices,
        callees=callee_slices,
        imports=import_slices,
        tests=test_slices,
        token_total=token_total,
        token_baseline=baseline,
        savings_pct=tokens.savings_pct(token_total, baseline),
        worth_using=worth,
        note=_build_note(token_total, baseline, budget, trimmed, upgraded, worth),
    )
