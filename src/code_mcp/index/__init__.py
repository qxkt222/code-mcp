"""Index layer: language parsing, incremental building and full-text search."""

from __future__ import annotations

from .builder import IndexStats, build_index
from .parsers import (
    ParsedFile,
    ParsedRef,
    ParsedSymbol,
    detect_language,
    parse_file,
    parse_source,
)
from .search import fts_escape, search_comments, search_symbols

__all__ = [
    "IndexStats",
    "ParsedFile",
    "ParsedRef",
    "ParsedSymbol",
    "build_index",
    "detect_language",
    "fts_escape",
    "parse_file",
    "parse_source",
    "search_comments",
    "search_symbols",
]
