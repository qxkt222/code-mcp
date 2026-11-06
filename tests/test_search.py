"""Search: FTS escaping, BM25 merge across symbols/comments and filters."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from code_mcp.config import DEFAULT_EXCLUDES, Config
from code_mcp.db import connect, init_schema
from code_mcp.index.builder import build_index
from code_mcp.index.search import fts_escape, search_comments, search_symbols

SERVICE = '''"""Payment service."""
import os

def charge_card(amount):
    """Charge a customer card."""
    # charge_card uses the gateway for retries
    return gateway(amount)

def parse_payload(data):
    return decode(data)
'''

UI = '''/** Paints the widget. */
export function renderWidget() {
  // 网关重试逻辑
  return draw();
}
'''


def _indexed(tmp_path: Path) -> sqlite3.Connection:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "service.py").write_bytes(SERVICE.encode("utf-8"))
    (root / "ui.ts").write_bytes(UI.encode("utf-8"))

    conn = connect(tmp_path / "search.db")
    init_schema(conn)
    build_index(
        root,
        conn,
        Config(root=root, excludes=list(DEFAULT_EXCLUDES)),
    )
    return conn


def _empty_conn(tmp_path: Path) -> sqlite3.Connection:
    conn = connect(tmp_path / "empty.db")
    init_schema(conn)
    return conn


def test_search_symbols_by_name_and_docstring(tmp_path: Path) -> None:
    conn = _indexed(tmp_path)
    by_name = search_symbols(conn, "charge")
    assert [symbol.qualname for symbol in by_name] == ["charge_card"]
    assert by_name[0].file == "service.py"
    assert by_name[0].kind == "function"

    by_doc = search_symbols(conn, "customer card")
    assert [symbol.qualname for symbol in by_doc] == ["charge_card"]


def test_comment_hits_map_to_enclosing_symbol(tmp_path: Path) -> None:
    conn = _indexed(tmp_path)
    # "gateway" appears only in a comment inside charge_card.
    assert [symbol.qualname for symbol in search_symbols(conn, "gateway")] == [
        "charge_card"
    ]
    # CJK comment inside renderWidget.
    assert [symbol.qualname for symbol in search_symbols(conn, "网关")] == [
        "renderWidget"
    ]


def test_symbol_and_comment_hits_are_deduplicated(tmp_path: Path) -> None:
    conn = _indexed(tmp_path)
    hits = search_symbols(conn, "charge")
    assert len(hits) == 1
    assert len({symbol.id for symbol in hits}) == 1


def test_search_comments(tmp_path: Path) -> None:
    conn = _indexed(tmp_path)
    assert search_comments(conn, "gateway") == [
        ("service.py", 6, "# charge_card uses the gateway for retries")
    ]
    assert search_comments(conn, "网关") == [
        ("ui.ts", 3, "// 网关重试逻辑")
    ]


def test_kinds_filter(tmp_path: Path) -> None:
    conn = _indexed(tmp_path)
    assert search_symbols(conn, "render", kinds=["class"]) == []
    functions = search_symbols(conn, "render", kinds=["function"])
    assert [symbol.qualname for symbol in functions] == ["renderWidget"]


def test_limit_is_respected(tmp_path: Path) -> None:
    conn = _indexed(tmp_path)
    assert len(search_symbols(conn, "charge card parse")) >= 2
    assert len(search_symbols(conn, "charge card parse", limit=1)) == 1
    assert search_symbols(conn, "charge", limit=0) == []


def test_module_anchors_are_not_search_results(tmp_path: Path) -> None:
    conn = _indexed(tmp_path)
    # "service" is the module anchor's name; module rows stay out of results.
    assert search_symbols(conn, "service") == []
    for symbol in search_symbols(conn, "widget draw charge"):
        assert symbol.kind != "module"


def test_fts_escape_handles_special_characters() -> None:
    assert fts_escape('foo" OR * (bar) -baz') == '"foo"* OR "OR"* OR "bar"* OR "baz"*'
    assert fts_escape('"*()-') == ""
    assert fts_escape("") == ""
    assert fts_escape("   ") == ""
    # Single letters are dropped only when longer terms are present.
    assert fts_escape("a file") == '"file"*'
    assert fts_escape("a b") == '"a"* OR "b"*'
    assert fts_escape("x") == '"x"*'
    assert fts_escape("parse_file") == '"parse_file"*'


def test_special_queries_do_not_break_search(tmp_path: Path) -> None:
    conn = _indexed(tmp_path)
    assert search_symbols(conn, '" OR * (') == []
    assert [symbol.qualname for symbol in search_symbols(conn, 'charge"*')] == [
        "charge_card"
    ]
    assert search_comments(conn, '")-*(') == []


def test_search_on_empty_index(tmp_path: Path) -> None:
    conn = _empty_conn(tmp_path)
    assert search_symbols(conn, "anything") == []
    assert search_comments(conn, "anything") == []
