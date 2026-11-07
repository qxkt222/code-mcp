"""Surgical single-symbol rewrite.

This is the highest-risk write path of the project, so every guard from the
shared contract is enforced before a byte reaches disk:

1. the symbol is located through the index (``start_byte``/``end_byte``);
2. the sha256 of the raw file must match the indexed hash, and
   ``expect_hash`` must match as well when supplied;
3. only that byte range is replaced, every other byte is preserved; when
   ``new_code`` repeats the indentation that precedes the range, that
   indentation is consumed instead of being duplicated;
4. ``new_code`` is normalised to the file's existing newline style;
5. the rewritten file is re-parsed; a syntax error or a vanished symbol rolls
   the edit back;
6. the index hash/mtime are updated only after a successful write;
7. the unified diff is capped at 200 lines.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import re
import sqlite3
from pathlib import Path
from typing import cast

import tree_sitter_python
import tree_sitter_rust
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

from .db import open_index
from .graph.refs import resolve_refs
from .graph.tauri import build_bridges
from .index.builder import reindex_file
from .paths import resolve_within
from .types import EditResult

MAX_DIFF_LINES = 200

_EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "tsx",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".rs": "rust",
}

_TS_DEFS = frozenset(
    {
        "function_declaration",
        "function_signature",
        "generator_function_declaration",
        "class_declaration",
        "method_definition",
        "method_signature",
        "abstract_method_signature",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "variable_declarator",
        "lexical_declaration",
        "variable_declaration",
    }
)

_DEF_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"function_definition", "class_definition", "assignment"}),
    "typescript": _TS_DEFS,
    "javascript": _TS_DEFS,
    "tsx": _TS_DEFS,
    "rust": frozenset(
        {
            "function_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "impl_item",
            "type_item",
            "const_item",
            "static_item",
            "mod_item",
            "union_item",
            "macro_definition",
        }
    ),
}

_WEB_LANGS = frozenset({"typescript", "javascript", "tsx"})

# One import statement is the unit of replacement: editing an import must never
# rewrite the surrounding import block.
_IMPORT_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"import_statement", "import_from_statement"}),
    "typescript": frozenset({"import_statement"}),
    "javascript": frozenset({"import_statement"}),
    "tsx": frozenset({"import_statement"}),
    "rust": frozenset({"use_declaration"}),
}

_SYMBOL_SQL = (
    "SELECT s.id, s.name, s.qualname, s.kind, s.start_line, s.end_line, "
    "       s.start_byte, s.end_byte, f.id AS file_id, f.path AS file_path, "
    "       f.hash AS file_hash, f.lang AS lang "
    "FROM symbols s JOIN files f ON f.id = s.file_id "
    "WHERE f.path = ? AND (s.qualname = ? OR s.name = ?) "
    "ORDER BY CASE WHEN s.qualname = ? THEN 0 WHEN s.name = ? THEN 1 ELSE 2 END, "
    "         s.start_byte "
    "LIMIT 1"
)


def replace_symbol(
    root: Path,
    file: str,
    name: str,
    new_code: str,
    expect_hash: str | None = None,
) -> EditResult:
    """Replace one symbol's byte range with ``new_code`` after all guards pass."""
    root = Path(root)
    rel = _relative_path(root, file)
    result = EditResult(file=rel or str(file), symbol=name)

    if rel is None:
        result.reason = "file is outside the repository root"
        return result
    if not new_code.strip():
        result.reason = "new_code is empty"
        return result

    path = resolve_within(root, rel)
    if path is None:
        result.reason = "file is outside the repository root"
        return result
    try:
        data = path.read_bytes()
    except OSError as exc:
        result.reason = f"cannot read file: {exc}"
        return result

    conn = open_index(root)
    try:
        symbol = _lookup_symbol(conn, rel, name)
        file_row = _lookup_file(conn, rel)
        if file_row is None:
            result.reason = f"file '{rel}' is not in the index; reindex before editing"
            return result

        language = str(file_row["lang"] or _lang_from_path(rel) or "")
        kind = ""
        check_name = True
        if symbol is not None:
            start = int(symbol["start_byte"])
            end = int(symbol["end_byte"])
            kind = str(symbol["kind"] or "")
            if not 0 <= start <= end <= len(data):
                result.reason = "indexed byte range lies outside the file"
                return result
            result.start_line = data[:start].count(b"\n") + 1
            result.end_line = data[:end].count(b"\n") + 1
            if kind == "constant":
                start, end = _expand_constant(language, data, start, end)
        else:
            if not _import_is_indexed(conn, rel, name):
                result.reason = f"symbol '{name}' not found in the index for {rel}"
                return result
            located = _find_import_range(language, data, name)
            if located is None:
                result.reason = (
                    f"import '{name}' is indexed for {rel} but its statement could "
                    "not be located; reindex and retry"
                )
                return result
            start, end = located
            kind = "import"
            check_name = False
            result.start_line = data[:start].count(b"\n") + 1
            result.end_line = data[:end].count(b"\n") + 1

        current_hash = hashlib.sha256(data).hexdigest()
        result.old_hash = current_hash

        indexed_hash = _normalize_hash(file_row["file_hash"])
        if indexed_hash != current_hash:
            result.reason = "file hash differs from the index; reindex before editing"
            return result
        if expect_hash is not None and _normalize_hash(expect_hash) != current_hash:
            result.reason = "expect_hash does not match the current file content"
            return result

        newline = _detect_newline(data)
        normalized = _normalize_newlines(new_code, newline)
        start = _shift_to_line_start(data, start, new_code)
        normalized = _avoid_double_newline(normalized, data, end, newline)
        new_data = data[:start] + normalized.encode("utf-8") + data[end:]

        symbol_name = str(symbol["name"]) if symbol is not None else name
        ok, reason = _validate(language, data, new_data, symbol_name, kind, check_name)
        if not ok:
            result.reason = reason
            return result

        old_text = data.decode("utf-8", errors="replace")
        new_text = new_data.decode("utf-8", errors="replace")
        result.lines_added, result.lines_removed, result.diff = _build_diff(
            old_text, new_text, rel
        )

        try:
            path.write_bytes(new_data)
        except OSError as exc:
            result.reason = f"cannot write file: {exc}"
            return result

        new_hash = hashlib.sha256(new_data).hexdigest()
        # Re-parse rather than only refreshing the hash: every symbol after the
        # edited range has shifted, and a stale offset would make the next edit
        # rewrite the wrong bytes.
        reindex_file(root, conn, rel)
        # Replacing the file row cascaded its symbols away, so every ref that
        # pointed at them now has ``to_symbol_id = NULL``. Rebuild those edges
        # (and the Tauri bridges) before reporting success; otherwise impact()
        # silently goes empty after the first edit.
        resolve_refs(conn)
        build_bridges(conn)
        conn.commit()

        result.applied = True
        result.new_hash = new_hash
        result.end_line = result.start_line + max(0, len(normalized.rstrip("\r\n").splitlines()) - 1)
        return result
    finally:
        conn.close()


def _relative_path(root: Path, file: str) -> str | None:
    raw = (file or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        target = resolve_within(root, raw)
        if target is None:
            return None
        return target.relative_to(root.resolve()).as_posix()
    normalized = raw.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized or None


def _lookup_symbol(conn: sqlite3.Connection, rel: str, name: str) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        conn.execute(_SYMBOL_SQL, (rel, name, name, name, name)).fetchone(),
    )


def _lookup_file(conn: sqlite3.Connection, rel: str) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        conn.execute(
            "SELECT f.hash AS file_hash, f.lang AS lang FROM files f WHERE f.path = ?",
            (rel,),
        ).fetchone(),
    )


def _import_is_indexed(conn: sqlite3.Connection, rel: str, name: str) -> bool:
    """True when ``name`` is a local binding of an indexed import of ``rel``."""
    rows = conn.execute(
        "SELECT i.module AS module, i.alias AS alias FROM imports AS i "
        "JOIN files AS f ON f.id = i.file_id WHERE f.path = ?",
        (rel,),
    )
    for row in rows:
        module = str(row["module"] or "")
        alias = str(row["alias"] or "")
        if alias == name:
            return True
        if alias:
            continue
        parts = [part for part in re.split(r"[.:]+", module) if part]
        if name == module or name in parts:
            return True
    return False


def _find_import_range(
    language: str, data: bytes, name: str
) -> tuple[int, int] | None:
    """Byte range of the single import statement that binds ``name``."""
    grammar = _grammar(language)
    allowed = _IMPORT_NODE_TYPES.get(language, frozenset())
    if grammar is None or not allowed:
        return None
    root = Parser(grammar).parse(data).root_node
    pattern = re.compile(r"(?<![\w])" + re.escape(name) + r"(?![\w])")
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in allowed:
            text = data[node.start_byte : node.end_byte].decode(
                "utf-8", errors="replace"
            )
            if pattern.search(text):
                return node.start_byte, node.end_byte
        stack.extend(reversed(node.children))
    return None


def _expand_constant(
    language: str, data: bytes, start: int, end: int
) -> tuple[int, int]:
    """Widen a single-declarator TS/JS constant to its whole declaration.

    The index stores a ``const`` symbol as its ``variable_declarator`` span, so
    replacing only that range would leave the ``const`` keyword behind. When the
    declaration holds exactly one declarator the whole ``lexical_declaration``
    is editable; declarations with siblings keep the narrower span so a rename
    cannot drop the other bindings.
    """
    if language not in _WEB_LANGS:
        return start, end
    grammar = _grammar(language)
    if grammar is None:
        return start, end
    node = _node_at(Parser(grammar).parse(data).root_node, start, end)
    parent = node.parent if node is not None else None
    if node is None or parent is None:
        return start, end
    if node.type != "variable_declarator":
        return start, end
    if parent.type not in ("lexical_declaration", "variable_declaration"):
        return start, end
    declarators = [
        child for child in parent.named_children if child.type == "variable_declarator"
    ]
    if len(declarators) != 1:
        return start, end
    return parent.start_byte, parent.end_byte


def _node_at(root: Node, start: int, end: int) -> Node | None:
    """Node whose byte range is exactly ``[start, end)``, if the tree has one."""
    stack = [root]
    while stack:
        node = stack.pop()
        if node.start_byte == start and node.end_byte == end:
            return node
        if node.start_byte <= start and end <= node.end_byte:
            stack.extend(node.children)
    return None


def _normalize_hash(value: str | None) -> str:
    return (value or "").strip().lower().removeprefix("sha256:")


def _detect_newline(data: bytes) -> str:
    crlf = data.count(b"\r\n")
    lf = data.count(b"\n") - crlf
    cr = data.count(b"\r") - crlf
    if crlf > 0 and crlf >= lf and crlf >= cr:
        return "\r\n"
    return "\n"


def _normalize_newlines(text: str, newline: str) -> str:
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    if newline == "\r\n":
        return unified.replace("\n", "\r\n")
    return unified


def _shift_to_line_start(data: bytes, start: int, new_code: str) -> int:
    """Move ``start`` to the line start when ``new_code`` repeats that indent.

    ``slice`` returns a symbol with the indentation it has in the file, while
    the index points at the declaration keyword itself. Replacing from
    ``start`` would therefore apply the first line's indentation twice. When
    the first line's indent equals the whitespace that precedes the indexed
    range, the replacement consumes that whitespace instead.
    """
    first_line = _normalize_newlines(new_code, "\n").split("\n", 1)[0]
    if not first_line.strip():
        return start
    indent = first_line[: len(first_line) - len(first_line.lstrip(" \t"))]
    if not indent:
        return start
    line_start = data.rfind(b"\n", 0, start) + 1
    leading = data[line_start:start]
    if leading.strip(b" \t"):
        return start
    if leading.decode("utf-8", errors="replace") != indent:
        return start
    return line_start


def _avoid_double_newline(text: str, data: bytes, end: int, newline: str) -> str:
    """Drop one trailing newline when the source already has one right after the range.

    Indexed ``end_byte`` points at the end of the syntax node, which excludes the
    newline that terminates the last line. Keeping the caller's trailing newline
    in that case would silently add a blank line on every edit.
    """
    ending = newline.encode("utf-8")
    if text.endswith(newline) and data[end : end + len(ending)] == ending:
        return text[: -len(newline)]
    return text


def _lang_from_path(rel: str) -> str | None:
    return _EXT_LANG.get(Path(rel).suffix.lower())


def _grammar(language: str) -> Language | None:
    try:
        if language == "python":
            return Language(tree_sitter_python.language())
        if language in {"typescript", "javascript"}:
            return Language(tree_sitter_typescript.language_typescript())
        if language == "tsx":
            return Language(tree_sitter_typescript.language_tsx())
        if language == "rust":
            return Language(tree_sitter_rust.language())
    except Exception:  # pragma: no cover - defensive against grammar changes
        return None
    return None


def _validate(
    language: str,
    original: bytes,
    rewritten: bytes,
    symbol_name: str,
    kind: str,
    check_name: bool,
) -> tuple[bool, str]:
    grammar = _grammar(language)
    if grammar is None:
        return False, f"unsupported language '{language}': cannot verify the rewrite"

    parser = Parser(grammar)
    root = parser.parse(rewritten).root_node
    if root.has_error:
        return False, "rewritten file has syntax errors; edit rolled back"

    if language == "python":
        try:
            ast.parse(rewritten.decode("utf-8", errors="replace"))
        except (SyntaxError, ValueError) as exc:
            message = getattr(exc, "msg", str(exc))
            return False, f"rewritten file is not valid Python ({message}); edit rolled back"

    if not check_name:
        # An import has no definition to look up; the syntax check above is the
        # whole contract for it.
        return True, ""

    # Distinguish "this indexed symbol type is not editable" from "the rewrite
    # dropped the symbol": the old single message claimed a constant was gone
    # from a file that still defined it.
    if not _defines(parser.parse(original).root_node, original, symbol_name, language):
        kinds = ", ".join(sorted(_DEF_TYPES.get(language, frozenset())))
        label = f"a {kind}" if kind else "a symbol"
        return (
            False,
            f"symbol '{symbol_name}' is {label}; only definitions of kind "
            f"[{kinds}] are editable",
        )

    if not _defines(root, rewritten, symbol_name, language):
        return False, f"symbol '{symbol_name}' is no longer defined; edit rolled back"
    return True, ""


def _defines(root: Node, data: bytes, name: str, language: str) -> bool:
    allowed = _DEF_TYPES.get(language, frozenset())
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in allowed and _defines_name(node, data, name):
            return True
        stack.extend(node.children)
    return False


def _defines_name(node: Node, data: bytes, name: str) -> bool:
    """True when ``node`` binds ``name``, across the supported declaration forms."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _text_of(name_node, data) == name
    if node.type == "assignment":
        # Python constants are indexed from their ``assignment`` node, whose
        # first bound name lives in the ``left`` field rather than ``name``.
        left = node.child_by_field_name("left")
        return (
            left is not None
            and left.type == "identifier"
            and _text_of(left, data) == name
        )
    if node.type in ("lexical_declaration", "variable_declaration"):
        for child in node.named_children:
            if child.type != "variable_declarator":
                continue
            declarator_name = child.child_by_field_name("name")
            if declarator_name is not None and _text_of(declarator_name, data) == name:
                return True
    return False


def _text_of(node: Node, data: bytes) -> str:
    return data[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _build_diff(old_text: str, new_text: str, rel: str) -> tuple[int, int, str]:
    lines = list(
        difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
        )
    )
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    if len(lines) > MAX_DIFF_LINES:
        lines = [*lines[: MAX_DIFF_LINES - 1], "... (diff truncated)"]
    return added, removed, "\n".join(lines)
