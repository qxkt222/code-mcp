"""Budget-aware context packing for one symbol.

The packer answers "what does an agent need to read to work on this symbol?"
with the smallest set of real source slices: the target itself, its callers,
its callees, the definitions it imports, and matching tests. Everything is read
from disk, never reconstructed from index summaries.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
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
from .index.chunks import (
    SEGMENT_MAX_TOKENS,
    SEGMENT_THRESHOLD_TOKENS,
    segment_symbol,
)
from .paths import is_test_path
from .types import FILE_KIND, Pack, Segment, Slice, Symbol

# Over-budget sections are emptied in this order (least essential first).
# Tests are the last thing dropped: knowing which tests cover the symbol is
# worth more than a few extra import lines.
_TRIM_ORDER = ("imports", "callees", "callers", "tests")
# Surplus budget expands sections in this order. Imports are deliberately
# absent: expanding one import used to mean pasting an entire unrelated file,
# which dominated the packed cost.
_UPGRADE_ORDER = ("callers", "callees")
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

# Below this baseline the response envelope alone (one fixed JSON wrapper plus
# a header per slice) tends to cost more than reading the file, no matter how
# small the target is. Such files are refused outright instead of pretending
# a pack is useful. Measured on real packs: 0-300 token files lose most often.
MIN_PACK_BASELINE_TOKENS = 300

# Absolute ceiling on expansion. Even when the baseline is huge, never grow the
# packed result past this many tokens by pulling in neighbour bodies. Signature
# slices still fit; whole implementations of unrelated files do not.
PACK_EXPAND_CAP_TOKENS = 4000

# If imports ever become expandable again, a single import may only pull in a
# small definition, never a whole file.
_IMPORT_EXPAND_MAX_LINES = 40
_IMPORT_EXPAND_MAX_TOKENS = 300

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
        self._raw: dict[str, bytes] = {}
        self._text: dict[str, str] = {}
        self._lines: dict[str, list[str]] = {}

    def raw(self, rel: str) -> bytes:
        """Raw bytes, which is what index byte offsets refer to."""
        if rel not in self._raw:
            try:
                self._raw[rel] = (self.root / rel).read_bytes()
            except OSError:
                self._raw[rel] = b""
        return self._raw[rel]

    def text(self, rel: str) -> str:
        if rel not in self._text:
            self._text[rel] = self.raw(rel).decode("utf-8", errors="replace")
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
    # Carried onto the Slice when an oversized symbol is reduced to an outline.
    segments: list[Segment] = field(default_factory=list)
    truncated: bool = False
    total_tokens: int = 0


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
    """Last line of the declaration head, located by matching the signature.

    Indexed signatures are whitespace-collapsed, so a multi-line ``def`` cannot
    be sliced by line count; the collapsed text is matched against the real
    lines instead. A signature clipped to ``...`` is only the prefix of the real
    text, so the exact end still comes from bracket balancing.
    """
    start = max(1, symbol.start_line)
    if start > len(lines):
        return start
    ceiling = symbol.end_line if symbol.end_line >= start else len(lines)
    ceiling = min(ceiling, len(lines))

    signature = " ".join(symbol.signature.split())
    if not signature:
        return start

    clipped = signature.endswith("...")
    needle = signature[:-3].rstrip() if clipped else signature
    if not needle:
        needle = signature

    limit = min(ceiling, start + 39)
    accumulated: list[str] = []
    for line_no in range(start, limit + 1):
        accumulated.append(lines[line_no - 1])
        collapsed = " ".join(" ".join(accumulated).split())
        if clipped:
            # The clipped signature covers only the start of the declaration
            # head; the end of the head is still found by bracket balance.
            if len(collapsed) >= len(needle) and collapsed.startswith(needle):
                break
        elif signature in collapsed:
            return line_no

    # Fallback: stop once brackets balance and the declaration terminates.
    depth = 0
    for line_no in range(start, limit + 1):
        line = lines[line_no - 1]
        stripped = line.rstrip()
        for index, char in enumerate(line):
            # A ``{`` that ends the line opens the body: the head ends here even
            # though the brace raises the depth, so test it before counting.
            if char == "{" and depth == 0 and not line[index + 1 :].strip():
                return line_no
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth = max(0, depth - 1)
        if depth == 0 and (stripped.endswith(":") or stripped.endswith(";")):
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
    # A piece without a symbol is a whole-file header (an import that resolved
    # to no specific definition). Expanding it would paste an unrelated file,
    # so it is never upgradeable.
    if piece.symbol is None:
        return piece
    return _full_piece(piece.symbol, reader)


def _to_slice(piece: _Piece) -> Slice:
    return Slice(
        file=piece.file,
        start_line=piece.start_line,
        end_line=piece.end_line,
        text=piece.text,
        symbol=piece.label,
        kind=piece.kind,
        tokens=piece.tokens,
        segments=list(piece.segments),
        truncated=piece.truncated,
        total_tokens=piece.total_tokens,
    )


def _outline_piece(
    target: Symbol,
    reader: _Reader,
    lang: str,
    budget: int,
    max_target_tokens: int,
) -> tuple[_Piece, int]:
    """Target body, or a signature plus outline when it is oversized.

    The second value is the number of segments in an outline, or 0 when the
    body was small enough to return whole.
    """
    piece = _full_piece(target, reader)
    threshold = min(max_target_tokens, budget) if budget > 0 else max_target_tokens
    if piece.tokens <= threshold:
        return piece, 0
    segments = _segment_target(target, reader, lang)
    if not segments:
        return piece, 0
    outline = _signature_piece(target, reader)
    outline.segments = segments
    outline.truncated = True
    outline.total_tokens = piece.tokens
    return outline, len(segments)


def _target_only_pack(
    target: Symbol,
    reader: _Reader,
    lang: str,
    baseline: int,
    budget: int,
    max_target_tokens: int,
    force: bool,
    reason: str = "",
) -> Pack:
    """Return the target alone; neighbour sections are never read."""
    piece, split_segments = _outline_piece(
        target, reader, lang, budget, max_target_tokens
    )
    target_slice = _to_slice(piece)
    token_total = target_slice.tokens
    response_tokens = token_total + tokens.envelope_tokens(
        [(target_slice.file, target_slice.symbol)]
    )
    worth = tokens.worth_using(response_tokens, baseline)
    note = _build_note(
        token_total,
        baseline,
        budget,
        {},
        0,
        worth,
        split_segments,
        response_tokens=response_tokens,
    )
    note += "; target only: neighbour sections omitted; use include_* to fetch them"
    if reason:
        note += f"; {reason}"
    if force:
        note = f"forced: {note}"
    return Pack(
        target=target_slice,
        token_total=token_total,
        response_tokens=response_tokens,
        token_baseline=baseline,
        savings_pct=tokens.savings_pct(token_total, baseline),
        worth_using=worth,
        note=note,
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


def _import_summary(module: str, alias: str) -> str:
    """One-line import statement for an import row.

    ``imports`` rows store ``(module, imported name)``, so the two common forms
    can be rebuilt without reading source.
    """
    module = (module or "").strip().strip("'\"")
    alias = (alias or "").strip().strip("'\"")
    if module and alias:
        return f"from {module} import {alias}"
    if module:
        return f"module: {module}"
    if alias:
        return f"module: {alias}"
    return ""


def _summary_piece(rel: str, label: str, text: str) -> _Piece:
    """Synthetic one-line import summary.

    ``start_line``/``end_line`` are 0: the text is not a source range, unlike a
    real slice. ``full`` is set so a summary is never expanded.
    """
    return _Piece(
        symbol=None,
        file=rel,
        start_line=0,
        end_line=0,
        kind="module",
        label=label,
        full=True,
        text=text,
        tokens=tokens.estimate(text),
    )


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
    # Module anchors carry no source range, so their import rows are collected
    # per module and rendered as one ``from X import a, b`` summary line.
    summaries: dict[str, _Piece] = {}
    summary_aliases: dict[str, list[str]] = {}
    seen_rows: set[tuple[str, str]] = set()
    for imp in import_rows:
        resolved = imp["resolved_file_id"]
        target_path = by_id.get(int(resolved)) if resolved is not None else None
        if target_path is None:
            target_path = _resolve_module(files, rel, imp["module"], file_row["lang"])
        if target_path is None:
            continue

        file_id, _lang = files[target_path]
        symbol = _import_symbol(conn, file_id, imp["module"], imp["alias"])
        if symbol is not None and symbol.kind == "module":
            # Module anchors are zero-width, parked at end of file, and exist
            # only so module-level refs have a ``from_symbol_id``. Slicing
            # their byte range returns the file's trailing characters, so the
            # import row becomes a one-line summary instead.
            module = (imp["module"] or "").strip().strip("'\"")
            alias = (imp["alias"] or "").strip().strip("'\"")
            key = (module, alias)
            if key in seen_rows:
                continue
            seen_rows.add(key)
            if module and alias:
                if module not in summaries:
                    summaries[module] = _summary_piece(target_path, module, "")
                    summary_aliases[module] = []
                    pieces.append(summaries[module])
                summary_aliases[module].append(alias)
            else:
                summary = _import_summary(module, alias)
                if summary:
                    pieces.append(_summary_piece(target_path, module or alias, summary))
        elif symbol is not None:
            if symbol.id in taken:
                continue
            taken.add(symbol.id)
            pieces.append(_signature_piece(symbol, reader))
        elif file_id not in seen_files:
            seen_files.add(file_id)
            pieces.append(_file_piece(target_path, reader))

    for module, piece in summaries.items():
        piece.text = _import_summary(module, ", ".join(summary_aliases[module]))
        piece.tokens = tokens.estimate(piece.text)
    return pieces


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

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


def _is_test_match(
    symbol: Symbol, target: Symbol, variants: set[str] | None = None
) -> bool:
    if variants is None:
        variants = _test_variants(target.name, target.qualname)
    if symbol.name in variants:
        return True
    if not is_test_path(symbol.file):
        return False
    if not symbol.name.startswith("test"):
        return False
    wanted = _name_parts(target.name)
    present = set(_split_ident(symbol.name))
    return bool(wanted) and wanted.issubset(present)


def _test_rank(
    symbol: Symbol, target: Symbol, variants: set[str] | None = None
) -> tuple[int, str, int, int]:
    if variants is None:
        variants = _test_variants(target.name, target.qualname)
    exact = symbol.name in variants
    return (0 if exact else 1, symbol.file, symbol.start_line, symbol.id)


def _like_escape(value: str) -> str:
    """Escape LIKE wildcards so a symbol name is matched literally."""
    return (
        value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )


def _like_needle(name: str) -> str:
    """Longest identifier part of ``name``, used as a cheap SQL prefilter.

    Every name this module can match against ``target`` contains at least one
    part of the target's bare name (test variants join the parts with
    underscores, and the prefix rule requires each part to appear), so the
    longest part is a safe superset filter.
    """
    parts = _split_ident(name)
    return max(parts, key=len) if parts else name


def _test_pieces(
    conn: sqlite3.Connection, target: Symbol, taken: set[int], reader: _Reader
) -> list[_Piece]:
    variants = _test_variants(target.name, target.qualname)
    pattern = f"%{_like_escape(_like_needle(target.name))}%"
    rows = conn.execute(
        f"""SELECT {SYMBOL_COLUMNS}
            FROM symbols AS s
            JOIN files AS f ON f.id = s.file_id
            WHERE (s.name LIKE '%test%' OR f.path LIKE '%test%')
              AND s.name LIKE ? ESCAPE '\\'
            ORDER BY f.path, s.start_line, s.id""",
        (pattern,),
    ).fetchall()

    matches: list[tuple[tuple[int, str, int, int], Symbol]] = []
    for row in rows:
        symbol = row_to_symbol(row)
        # Module anchors are zero-width bookkeeping rows, not sliceable tests.
        if symbol.kind == "module":
            continue
        if symbol.id in taken or symbol.id == target.id:
            continue
        if _is_test_match(symbol, target, variants):
            matches.append((_test_rank(symbol, target, variants), symbol))
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
    split_segments: int = 0,
    response_tokens: int | None = None,
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
    if split_segments:
        parts.append(
            f"target split into {split_segments} segments; "
            "fetch one with slice(file, start_line, end_line)"
        )
    if upgraded:
        parts.append(f"expanded {upgraded} slice(s) to full body")
    if not worth:
        if baseline <= 0:
            parts.append("not recommended: no readable file text")
        elif response_tokens is not None:
            # The percentage above is body-only; explain why the serialised
            # response can still lose even when the body looks smaller.
            effective = tokens.savings_pct(response_tokens, baseline)
            if response_tokens >= baseline:
                parts.append(
                    "not recommended: including response overhead the packed "
                    "response is not smaller than the file; "
                    "read the file directly instead"
                )
            else:
                parts.append(
                    f"not recommended: including response overhead the saving "
                    f"is {effective}% (body {savings}%), below the 25% "
                    "threshold; read the file directly instead"
                )
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


def _file_lang(conn: sqlite3.Connection, rel: str) -> str:
    row = conn.execute("SELECT lang FROM files WHERE path = ?", (rel,)).fetchone()
    return row["lang"] if row is not None else ""


# --------------------------------------------------------------------------- #
# Index freshness
# --------------------------------------------------------------------------- #


def ensure_fresh(conn: sqlite3.Connection, root: Path, rel: str) -> bool:
    """Re-index ``rel`` when its mtime or size drifted from the index.

    The index stores byte offsets, so a file edited outside ``edit`` shifts
    every later symbol; slicing by those offsets would return a neighbour's
    code. Returns True when the file is safe to slice by offset.
    """
    # Imported here rather than at module load: builder and graph do not import
    # pack today, but keeping the dependency local keeps that cycle impossible.
    from .graph.refs import resolve_refs
    from .graph.tauri import build_bridges
    from .index.builder import _load_sizes, reindex_file

    rel = normalize_path(rel)
    root = Path(root)
    try:
        stat = (root / rel).stat()
    except OSError:
        return False

    row = conn.execute("SELECT mtime FROM files WHERE path = ?", (rel,)).fetchone()
    if row is not None:
        sizes = _load_sizes(conn)
        if float(row["mtime"]) == stat.st_mtime and sizes.get(rel) == stat.st_size:
            return True

    if not reindex_file(root, conn, rel):
        return False
    # Replacing the file row cascaded its symbols away and nulled every ref
    # that pointed at them, so rebuild the graph edges before returning.
    resolve_refs(conn)
    build_bridges(conn)
    conn.commit()
    return True


# --------------------------------------------------------------------------- #
# Relevance ordering
# --------------------------------------------------------------------------- #


def _relevance_key(item: Symbol | _Piece, target_file: str) -> tuple[int, int, str]:
    """Distance from ``target_file``: same file, directory, package, rest.

    Within one tier, earlier source lines come first. ``file`` only breaks ties
    between different files that happen to share a line number.
    """
    file = item.file
    if file == target_file:
        tier = 0
    else:
        target_dir = target_file.rsplit("/", 1)[0] if "/" in target_file else ""
        item_dir = file.rsplit("/", 1)[0] if "/" in file else ""
        if item_dir == target_dir:
            tier = 1
        elif file.split("/", 1)[0] == target_file.split("/", 1)[0]:
            tier = 2
        else:
            tier = 3
    return tier, item.start_line, file


def _line_byte_range(source: bytes, start_line: int, end_line: int) -> tuple[int, int]:
    """Byte offsets for an inclusive 1-based line range, for stale index rows."""
    lines = source.splitlines(keepends=True)
    if not lines:
        return 0, 0
    first = min(max(1, start_line), len(lines))
    last = min(max(first, end_line), len(lines))
    start = sum(len(line) for line in lines[: first - 1])
    end = start + sum(len(line) for line in lines[first - 1 : last])
    return start, min(end, len(source))


def _target_bytes(target: Symbol, reader: _Reader) -> tuple[bytes, int, int]:
    """Raw source plus byte offsets; derive offsets when the index has none."""
    source = reader.raw(target.file)
    start, end = target.start_byte, target.end_byte
    if 0 <= start < end <= len(source):
        return source, start, end
    start, end = _line_byte_range(source, target.start_line, target.end_line)
    return source, start, end


def _segment_target(target: Symbol, reader: _Reader, lang: str) -> list[Segment]:
    source, start, end = _target_bytes(target, reader)
    if start >= end:
        return []
    return segment_symbol(source, lang, start, end, SEGMENT_MAX_TOKENS)


def _pack_segment(
    target: Symbol, reader: _Reader, lang: str, segment: int, baseline: int
) -> Pack:
    """Return exactly one block of a segmented symbol."""
    label = _label(target)
    segments = _segment_target(target, reader, lang)
    count = len(segments)

    if count == 0:
        if segment != 1:
            return Pack(
                note=(
                    f"segment {segment} out of range: "
                    f"{target.file}:{label} has 1 segment"
                ),
                worth_using=False,
            )
        piece = _full_piece(target, reader)
    else:
        if segment < 1 or segment > count:
            unit = "segment" if count == 1 else "segments"
            return Pack(
                note=(
                    f"segment {segment} out of range: "
                    f"{target.file}:{label} has {count} {unit}"
                ),
                worth_using=False,
            )
        chosen = segments[segment - 1]
        start, end, text = reader.slice(
            target.file, chosen.start_line, chosen.end_line
        )
        piece = _Piece(
            symbol=target,
            file=target.file,
            start_line=start,
            end_line=end,
            kind=target.kind,
            label=label,
            full=False,
            text=text,
            tokens=tokens.estimate(text),
        )

    total = piece.tokens
    response_tokens = total + tokens.envelope_tokens([(piece.file, piece.label)])
    worth = tokens.worth_using(response_tokens, baseline)
    note = _build_note(
        total, baseline, total, {}, 0, worth, response_tokens=response_tokens
    )
    count = count or 1
    return Pack(
        target=_to_slice(piece),
        token_total=total,
        response_tokens=response_tokens,
        token_baseline=baseline,
        savings_pct=tokens.savings_pct(total, baseline),
        worth_using=worth,
        note=f"segment {segment}/{count} of {label}; {note}",
    )


def pack(
    conn: sqlite3.Connection,
    root: Path,
    file: str,
    name: str | None,
    budget_tokens: int = 30000,
    include_tests: bool = True,
    full_body: bool = True,
    max_target_tokens: int = SEGMENT_THRESHOLD_TOKENS,
    segment: int | None = None,
    force: bool = False,
    target_only: bool = False,
) -> Pack:
    """Pack the causal neighbourhood of ``file``:``name`` within a token budget.

    Slices are read from disk. A target larger than ``max_target_tokens`` is
    reduced to its signature plus an outline of contiguous ``segments``; the
    caller can then fetch one block with ``segment=N`` or by line range. Small
    targets keep their full body. Callers and callees start as signature slices
    and are expanded only while the packed result stays under both the budget
    ratio and the absolute expansion cap. Imports are never expanded.

    When over budget, sections are trimmed in the order tests, callers, callees,
    imports, and the counts land in ``note``.

    ``target_only=True`` returns just the target (or its outline) and leaves
    every neighbour section empty: no callers, callees, imports or tests are
    collected or expanded. Use it when only the symbol body is wanted.

    ``force=True`` skips the two refusals (tiny-file floor and response costing
    more than a plain read), returning the payload anyway. The usual savings
    fields are still filled in and ``note`` is prefixed with ``forced: `` so the
    caller can judge the cost itself. A forced pack whose full response would
    still cost more than a plain read is degraded to the target alone.
    """
    rel = normalize_path(file)
    # Offsets in the index are only meaningful while the file still matches
    # what was parsed. A drifted file is re-indexed before anything is sliced.
    if not ensure_fresh(conn, root, rel):
        return Pack(
            note="file changed on disk and could not be re-indexed; run index",
            worth_using=False,
        )

    target = find_symbol(conn, rel, name) if name else _default_target(conn, rel)
    label = f"{rel}:{name}" if name else rel
    if target is None:
        return Pack(note=f"target not found: {label}", worth_using=False)
    if target.kind == "module":
        # Module anchors are zero-width bookkeeping rows, not source symbols;
        # slicing one would return the tail of the file.
        return Pack(note=f"target not found: {label}", worth_using=False)

    reader = _Reader(Path(root))
    baseline = tokens.estimate(reader.text(rel))

    # A tiny file can never pay for the response envelope, however small the
    # target is: reading the file directly is always the better move. A forced
    # call still returns the payload, with the cost stated in the note.
    if baseline < MIN_PACK_BASELINE_TOKENS and not force:
        return Pack(
            token_baseline=baseline,
            worth_using=False,
            note=f"file is only {baseline} tokens; read it directly",
        )

    lang = _file_lang(conn, rel)
    budget = max(0, int(budget_tokens))
    max_target = max(0, int(max_target_tokens))

    if segment is not None:
        result = _pack_segment(target, reader, lang, segment, baseline)
        if force and result.target is not None:
            result.note = f"forced: {result.note}"
        return result

    if target_only:
        return _target_only_pack(
            target, reader, lang, baseline, budget, max_target, force
        )

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
        # Module anchors are zero-width bookkeeping rows, not sliceable source.
        if symbol.kind == "module":
            return False
        if symbol.id in taken:
            return False
        return include_tests or not _is_test_match(symbol, target)

    # Rank by distance from the target before any budget trimming: the least
    # relevant neighbours are the ones dropped first.
    for symbol in sorted(
        callers(conn, target.id), key=lambda item: _relevance_key(item, rel)
    ):
        if not usable(symbol):
            continue
        taken.add(symbol.id)
        sections["callers"].append(_signature_piece(symbol, reader))

    for symbol in sorted(
        callees(conn, target.id), key=lambda item: _relevance_key(item, rel)
    ):
        if not usable(symbol):
            continue
        taken.add(symbol.id)
        sections["callees"].append(_signature_piece(symbol, reader))

    sections["imports"] = _import_pieces(conn, rel, reader, taken)

    # Oversized targets become a signature plus an addressable outline. The
    # outline keeps the result honest: the caller still sees what the symbol
    # contains, but pays for the parts it actually needs.
    target_piece, split_segments = _outline_piece(
        target, reader, lang, budget, max_target
    )

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
    expansion_cap = min(cap, PACK_EXPAND_CAP_TOKENS)

    upgraded = 0
    # A split target must not trigger expansion: the budget freed by the
    # outline would otherwise be spent pasting every neighbour in full.
    if full_body and not split_segments and total < cap:
        for key in _UPGRADE_ORDER:
            items = sections[key]
            for index, piece in enumerate(items):
                if piece.full or piece.symbol is None:
                    continue
                expanded = _expand(piece, reader)
                delta = expanded.tokens - piece.tokens
                if delta <= 0:
                    items[index] = expanded
                    continue
                # Imports are not in _UPGRADE_ORDER, but if that ever changes,
                # only a small definition may be pulled in -- never a whole
                # file.
                if key == "imports" and (
                    expanded.end_line - expanded.start_line + 1
                    > _IMPORT_EXPAND_MAX_LINES
                    or expanded.tokens > _IMPORT_EXPAND_MAX_TOKENS
                ):
                    continue
                if total + delta <= expansion_cap:
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
    # Judge the pack on what the caller actually receives: slice bodies plus
    # the JSON envelope that repeats every slice's metadata.
    all_slices = [
        target_slice,
        *caller_slices,
        *callee_slices,
        *import_slices,
        *test_slices,
    ]
    response_tokens = token_total + tokens.envelope_tokens(
        [(item.file, item.symbol) for item in all_slices]
    )
    # A response that costs more than a plain read is never worth sending, even
    # when it is only marginally over: the caller would pay for the very tokens
    # this tool exists to save. Return the advice instead of the payload. A
    # forced call still returns something -- but when the neighbourhood is what
    # pushed the response over the baseline, return only the target and say so.
    if baseline > 0 and response_tokens >= baseline:
        if not force:
            return Pack(
                response_tokens=response_tokens,
                token_baseline=baseline,
                savings_pct=tokens.savings_pct(token_total, baseline),
                worth_using=False,
                note=(
                    f"packing would cost {response_tokens} tokens vs {baseline} "
                    "for the file; read the file directly"
                ),
            )
        return _target_only_pack(
            target,
            reader,
            lang,
            baseline,
            budget,
            max_target,
            force,
            reason=(
                f"full pack cost {response_tokens} tokens vs {baseline} "
                "for the file, so only the target was returned"
            ),
        )

    worth = tokens.worth_using(response_tokens, baseline)
    note = _build_note(
        token_total,
        baseline,
        budget,
        trimmed,
        upgraded,
        worth,
        split_segments,
        response_tokens=response_tokens,
    )
    if force:
        note = f"forced: {note}"

    return Pack(
        target=target_slice,
        callers=caller_slices,
        callees=callee_slices,
        imports=import_slices,
        tests=test_slices,
        token_total=token_total,
        response_tokens=response_tokens,
        token_baseline=baseline,
        savings_pct=tokens.savings_pct(token_total, baseline),
        worth_using=worth,
        note=note,
    )
