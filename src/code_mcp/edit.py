"""Surgical single-symbol rewrite.

This is the highest-risk write path of the project, so every guard from the
shared contract is enforced before a byte reaches disk:

1. the symbol is located through the index (``start_byte``/``end_byte``);
2. the sha256 of the raw file must match the indexed hash, and
   ``expect_hash`` must match as well when supplied;
3. only that byte range is replaced, every other byte is preserved;
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
import sqlite3
from pathlib import Path

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
    }
)

_DEF_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"function_definition", "class_definition"}),
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
        if symbol is None:
            result.reason = f"symbol '{name}' not found in the index for {rel}"
            return result

        start = int(symbol["start_byte"])
        end = int(symbol["end_byte"])
        if not 0 <= start <= end <= len(data):
            result.reason = "indexed byte range lies outside the file"
            return result
        result.start_line = data[:start].count(b"\n") + 1
        result.end_line = data[:end].count(b"\n") + 1

        current_hash = hashlib.sha256(data).hexdigest()
        result.old_hash = current_hash

        indexed_hash = _normalize_hash(symbol["file_hash"])
        if indexed_hash != current_hash:
            result.reason = "file hash differs from the index; reindex before editing"
            return result
        if expect_hash is not None and _normalize_hash(expect_hash) != current_hash:
            result.reason = "expect_hash does not match the current file content"
            return result

        newline = _detect_newline(data)
        normalized = _normalize_newlines(new_code, newline)
        normalized = _avoid_double_newline(normalized, data, end, newline)
        new_data = data[:start] + normalized.encode("utf-8") + data[end:]

        language = str(symbol["lang"] or _lang_from_path(rel) or "")
        ok, reason = _validate(language, new_data, str(symbol["name"]))
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
    return conn.execute(_SYMBOL_SQL, (rel, name, name, name, name)).fetchone()


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


def _validate(language: str, data: bytes, symbol_name: str) -> tuple[bool, str]:
    grammar = _grammar(language)
    if grammar is None:
        return False, f"unsupported language '{language}': cannot verify the rewrite"

    tree = Parser(grammar).parse(data)
    root = tree.root_node
    if root.has_error:
        return False, "rewritten file has syntax errors; edit rolled back"

    if language == "python":
        try:
            ast.parse(data.decode("utf-8", errors="replace"))
        except (SyntaxError, ValueError) as exc:
            message = getattr(exc, "msg", str(exc))
            return False, f"rewritten file is not valid Python ({message}); edit rolled back"

    if not _defines(root, data, symbol_name, language):
        return False, f"symbol '{symbol_name}' is no longer defined; edit rolled back"
    return True, ""


def _defines(root: Node, data: bytes, name: str, language: str) -> bool:
    allowed = _DEF_TYPES.get(language, frozenset())
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in allowed:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                text = data[name_node.start_byte : name_node.end_byte].decode(
                    "utf-8", errors="replace"
                )
                if text == name:
                    return True
        stack.extend(node.children)
    return False


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
        lines = lines[: MAX_DIFF_LINES - 1] + ["... (diff truncated)"]
    return added, removed, "\n".join(lines)
