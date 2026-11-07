"""Language detection and the public parse entry points for the index.

Byte offsets returned here always refer to the raw file bytes, never to the
decoded string: files may contain invalid UTF-8 or multi-byte characters, and
the edit layer relies on these offsets to splice the original file.
"""

from __future__ import annotations

from pathlib import Path

from .languages import (  # re-exported below for the documented contract
    PARSERS,
    Collector,
    ParsedFile,
    ParsedRef,
    ParsedSymbol,
    parse_python,
    parse_rust,
    parse_typescript,
)

_SUFFIXES: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "tsx",  # JSX needs the tsx grammar
    ".rs": "rust",
}

__all__ = [
    "PARSERS",
    "Collector",
    "ParsedFile",
    "ParsedRef",
    "ParsedSymbol",
    "detect_language",
    "parse_file",
    "parse_source",
]


def detect_language(path: Path) -> str | None:
    """Map a file path to one of ``types.LANGUAGES`` by suffix."""
    return _SUFFIXES.get(path.suffix.lower())


def parse_source(source: bytes, lang: str) -> ParsedFile:
    """Parse raw bytes with the grammar for ``lang``."""
    parser = PARSERS.get(lang)
    if parser is None:
        raise ValueError(f"unsupported language: {lang!r}")
    if lang == "python":
        return parse_python(source)
    if lang == "rust":
        return parse_rust(source)
    return parse_typescript(source, lang)


def parse_file(path: Path, rel: str, lang: str) -> ParsedFile:
    """Read ``path`` as bytes and parse it.

    ``rel`` is only used for error messages; it keeps callers from having to
    re-derive a repository-relative name just to parse.
    """
    try:
        source = Path(path).read_bytes()
    except OSError as exc:  # pragma: no cover - caller normally checks first
        raise OSError(f"cannot read {rel}: {exc}") from exc
    return parse_source(source, lang)
