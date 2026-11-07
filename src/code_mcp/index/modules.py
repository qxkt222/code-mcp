"""Import-specifier resolution for the index.

An ``imports`` row stores the raw specifier exactly as it appears in source.
This module maps that specifier to an indexed file using the conventions of the
importing language, then persists the result through ``resolve_imports``.
``ModuleResolver`` exposes the same mapping in memory so callers that only need
import evidence (``graph.refs``) do not have to write to the table first.

Only an unambiguous match is accepted: when several indexed files could satisfy
one specifier, the import is left unresolved rather than guessed.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_WEB_LANGS = frozenset({"typescript", "tsx", "javascript"})

_PY_SUFFIXES = (".py", "/__init__.py")
_WEB_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mts", ".cts", ".mjs", ".cjs")
_RUST_SUFFIXES = (".rs", "/mod.rs")

# tsconfig files are JSONC in practice. ``json.loads`` is tried first so a
# comment-free file is never rewritten; the fallback strips the two constructs
# JSON itself rejects.


@dataclass(frozen=True, slots=True)
class _TsConfig:
    """The ``compilerOptions`` entries that affect module resolution."""

    base_url: str
    base_declared: bool
    paths: tuple[tuple[str, tuple[str, ...]], ...]


class ModuleResolver:
    """Resolve raw import specifiers against one snapshot of indexed files.

    ``files`` maps repo-relative paths to ``(file_id, language)``. ``root`` is
    used to load ``tsconfig.json``/``jsconfig.json`` once; without it only
    relative specifiers (and absolute Python/Rust paths) can be resolved.
    """

    def __init__(
        self, files: Mapping[str, tuple[int, str]], root: Path | None = None
    ) -> None:
        self._files = files
        self._tsconfig = _load_ts_config(root) if root is not None else None

    def resolve(self, from_path: str | None, module: str) -> str | None:
        """Return the single indexed path ``module`` denotes, or None."""
        if not from_path or from_path not in self._files:
            return None
        spec = (module or "").strip().strip("'\"")
        if not spec:
            return None

        lang = self._files[from_path][1]
        if lang == "python":
            found, _ambiguous = _unique_existing(
                _python_candidates(from_path, spec), self._files
            )
            return found
        if lang == "rust":
            found, _ambiguous = _unique_existing(
                _rust_candidates(from_path, spec), self._files
            )
            return found
        if lang in _WEB_LANGS:
            return self._resolve_web(from_path, spec)
        return None

    def _resolve_web(self, from_path: str, spec: str) -> str | None:
        directory = _dirname(from_path)
        if spec.startswith("."):
            found, _ambiguous = _unique_existing(
                _web_expand(_posix_join(directory, spec)), self._files
            )
            return found
        if spec.startswith("/"):
            found, _ambiguous = _unique_existing(
                _web_expand(spec.lstrip("/")), self._files
            )
            return found

        if self._tsconfig is None:
            return None

        alias_hits = self._alias_candidates(spec)
        if alias_hits:
            found, ambiguous = _unique_existing(alias_hits, self._files)
            if found is not None or ambiguous:
                return found

        # TypeScript falls back to ``baseUrl`` when no ``paths`` pattern hits.
        if not self._tsconfig.base_declared:
            return None
        base_hits = _web_expand(_posix_join(self._tsconfig.base_url, spec))
        found, _ambiguous = _unique_existing(base_hits, self._files)
        return found

    def _alias_candidates(self, spec: str) -> list[str]:
        if self._tsconfig is None:
            return []
        candidates: list[str] = []
        for pattern, targets in self._tsconfig.paths:
            captured = _match_pattern(pattern, spec)
            if captured is None:
                continue
            for target in targets:
                expanded = target.replace("*", captured)
                candidates.extend(
                    _web_expand(_posix_join(self._tsconfig.base_url, expanded))
                )
        return candidates


def resolve_imports(conn: sqlite3.Connection, root: Path) -> int:
    """Fill ``imports.resolved_file_id`` for every import row.

    Recomputes every row, so a stale target left behind by a reindex is
    corrected (or cleared) as well. Returns the number of rows whose
    ``resolved_file_id`` changed.
    """
    files: dict[str, tuple[int, str]] = {
        row["path"]: (int(row["id"]), row["lang"] or "")
        for row in conn.execute("SELECT id, path, lang FROM files")
    }
    by_id = {file_id: path for path, (file_id, _lang) in files.items()}
    resolver = ModuleResolver(files, Path(root))

    updates: list[tuple[int | None, int]] = []
    rows = conn.execute(
        "SELECT id, file_id, module, resolved_file_id FROM imports ORDER BY id"
    ).fetchall()
    for row in rows:
        from_path = by_id.get(int(row["file_id"]))
        target = resolver.resolve(from_path, row["module"])
        target_id = files[target][0] if target is not None else None
        if target_id != row["resolved_file_id"]:
            updates.append((target_id, int(row["id"])))

    if updates:
        conn.executemany(
            "UPDATE imports SET resolved_file_id = ? WHERE id = ?", updates
        )
        conn.commit()
    return len(updates)


# --------------------------------------------------------------------------- #
# Language-specific candidates
# --------------------------------------------------------------------------- #


def _python_candidates(from_path: str, module: str) -> list[str]:
    if module.startswith("."):
        dots = len(module) - len(module.lstrip("."))
        rest = module[dots:]
        base = _dirname(from_path)
        for _ in range(max(0, dots - 1)):
            base = _dirname(base)
        stem = _posix_join(base, rest.replace(".", "/")) if rest else base
        return [stem + suffix for suffix in _PY_SUFFIXES]
    dotted = module.replace(".", "/")
    return [dotted + suffix for suffix in _PY_SUFFIXES]


def _web_expand(stem: str) -> list[str]:
    stem = _normalize(stem)
    if not stem:
        return []
    candidates = [stem]
    for suffix in _WEB_SUFFIXES:
        candidates.append(stem + suffix)
        candidates.append(f"{stem}/index{suffix}")
    return candidates


def _rust_candidates(from_path: str, module: str) -> list[str]:
    parts = [part for part in module.split("::") if part]
    if not parts:
        return []

    head, rest = parts[0], parts[1:]
    if head == "crate":
        base = "src"
    elif head == "self":
        base = _rust_module_dir(from_path)
    elif head == "super":
        base = _dirname(_rust_module_dir(from_path))
    else:
        # An external crate (``std::...``, ``serde::...``) is not indexed.
        return []
    if not rest:
        return []

    stem = _posix_join(base, "/".join(rest))
    return [stem + suffix for suffix in _RUST_SUFFIXES]


def _rust_module_dir(from_path: str) -> str:
    """Module path of ``from_path`` in Rust 2018 layout."""
    path = _normalize(from_path)
    directory = _dirname(path)
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".rs"):
        name = name[:-3]
    if name in ("lib", "main", "mod"):
        return directory
    return _posix_join(directory, name)


def _unique_existing(
    candidates: list[str], files: Mapping[str, tuple[int, str]]
) -> tuple[str | None, bool]:
    """Return ``(path, ambiguous)`` for candidates that exist in ``files``."""
    seen: set[str] = set()
    hits: set[str] = set()
    for candidate in candidates:
        path = _normalize(candidate)
        if not path or path in seen:
            continue
        seen.add(path)
        if path in files:
            hits.add(path)
    if len(hits) == 1:
        return next(iter(hits)), False
    return None, len(hits) > 1


# --------------------------------------------------------------------------- #
# tsconfig / jsconfig aliases
# --------------------------------------------------------------------------- #


def _load_ts_config(root: Path) -> _TsConfig | None:
    for name in ("tsconfig.json", "jsconfig.json"):
        path = Path(root) / name
        if not path.is_file():
            continue
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        options = data.get("compilerOptions")
        if not isinstance(options, dict):
            options = {}

        base = options.get("baseUrl")
        if isinstance(base, str) and base.strip():
            base_declared = True
            base_url = _normalize(_posix_join("", base))
        else:
            base_declared = False
            base_url = ""

        entries: list[tuple[str, tuple[str, ...]]] = []
        raw_paths = options.get("paths")
        if isinstance(raw_paths, dict):
            for pattern, targets in raw_paths.items():
                if not isinstance(pattern, str) or not isinstance(targets, list):
                    continue
                cleaned = tuple(target for target in targets if isinstance(target, str))
                if cleaned:
                    entries.append((pattern, cleaned))
        return _TsConfig(
            base_url=base_url, base_declared=base_declared, paths=tuple(entries)
        )
    return None


def _read_json(path: Path) -> object | None:
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError:
        return None
    try:
        parsed: object = json.loads(raw)
        return parsed
    except ValueError:
        pass
    try:
        parsed = json.loads(_strip_jsonc(raw))
        return parsed
    except ValueError:
        return None


def _strip_jsonc(text: str) -> str:
    """Remove ``//``/``/* */`` comments and trailing commas, respecting strings."""
    out: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < len(text):
            following = text[index + 1]
            if following == "/":
                newline = text.find("\n", index)
                index = len(text) if newline < 0 else newline
                continue
            if following == "*":
                closing = text.find("*/", index + 2)
                index = len(text) if closing < 0 else closing + 2
                continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue
        out.append(char)
        index += 1
    return "".join(out)


def _match_pattern(pattern: str, spec: str) -> str | None:
    """Return the ``*`` capture when ``spec`` matches an alias pattern."""
    star = pattern.find("*")
    if star < 0:
        return "" if pattern == spec else None
    prefix, suffix = pattern[:star], pattern[star + 1 :]
    if not spec.startswith(prefix) or not spec.endswith(suffix):
        return None
    if len(spec) < len(prefix) + len(suffix):
        return None
    end = len(spec) - len(suffix) if suffix else len(spec)
    return spec[len(prefix) : end]


# --------------------------------------------------------------------------- #
# Path helpers (repo-relative, forward-slashed)
# --------------------------------------------------------------------------- #


def _normalize(path: str) -> str:
    cleaned = (path or "").replace("\\", "/").strip()
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def _posix_join(base: str, rel: str) -> str:
    stack = [part for part in _normalize(base).split("/") if part]
    for part in _normalize(rel).split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if stack:
                stack.pop()
        else:
            stack.append(part)
    return "/".join(stack)


def _dirname(path: str) -> str:
    normalized = _normalize(path)
    return normalized.rsplit("/", 1)[0] if "/" in normalized else ""
