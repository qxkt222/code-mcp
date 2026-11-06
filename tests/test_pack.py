"""Behavioural tests for the context packer."""

from __future__ import annotations

import sqlite3

from code_mcp import db, tokens
from code_mcp.graph.refs import resolve_refs
from code_mcp.pack import pack
from code_mcp.types import CONF_NAME, REF_CALL

SERVICE = '''"""Service helpers."""
from app.util import format_value, parse


def save(path):
    """Persist a value."""
    return parse(path)


def load(path):
    """Load a value."""
    return parse(path)
'''

UTIL = '''def parse(value):
    """Parse a value."""
    return value.strip()


def format_value(value):
    """Format a value."""
    return f"[{value}]"
'''

VIEW = '''from app.service import load


def render(path):
    return load(path)
'''

TESTS = '''from app.service import load


def test_load():
    assert load(" x ") == "x"


def load_test():
    assert load("y") == "y"
'''


def _service_text() -> str:
    padding = "\n".join(f"# padding line {index}" for index in range(1, 81))
    return SERVICE + "\n" + padding + "\n"


def _db(tmp_path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "index.db")
    db.init_schema(conn)
    return conn


def _file(conn: sqlite3.Connection, path: str, text: str, lang: str = "python") -> int:
    cursor = conn.execute(
        "INSERT INTO files(path, lang, mtime, hash, loc) VALUES (?, ?, 0, '', ?)",
        (path, lang, len(text.splitlines())),
    )
    return int(cursor.lastrowid)


def _symbol(
    conn: sqlite3.Connection,
    file_id: int,
    name: str,
    start: int,
    end: int,
    signature: str = "",
    docstring: str = "",
    kind: str = "function",
) -> int:
    cursor = conn.execute(
        """INSERT INTO symbols(file_id, name, qualname, kind, start_line, end_line,
                               start_byte, end_byte, signature, docstring)
           VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?)""",
        (file_id, name, name, kind, start, end, signature, docstring),
    )
    return int(cursor.lastrowid)


def _ref(conn: sqlite3.Connection, from_id: int, to_name: str) -> None:
    conn.execute(
        """INSERT INTO refs(from_symbol_id, to_symbol_id, to_name, kind, confidence)
           VALUES (?, NULL, ?, ?, ?)""",
        (from_id, to_name, REF_CALL, CONF_NAME),
    )


def _fixture(tmp_path) -> tuple[sqlite3.Connection, dict[str, int]]:
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()

    service_text = _service_text()
    (tmp_path / "app" / "service.py").write_text(
        service_text, encoding="utf-8", newline="\n"
    )
    (tmp_path / "app" / "util.py").write_text(UTIL, encoding="utf-8", newline="\n")
    (tmp_path / "app" / "view.py").write_text(VIEW, encoding="utf-8", newline="\n")
    (tmp_path / "tests" / "test_service.py").write_text(
        TESTS, encoding="utf-8", newline="\n"
    )

    conn = _db(tmp_path)
    service = _file(conn, "app/service.py", service_text)
    util = _file(conn, "app/util.py", UTIL)
    view = _file(conn, "app/view.py", VIEW)
    tests = _file(conn, "tests/test_service.py", TESTS)

    ids = {
        "save": _symbol(conn, service, "save", 5, 7, "def save(path):", "Persist a value."),
        "load": _symbol(conn, service, "load", 10, 12, "def load(path):", "Load a value."),
        "parse": _symbol(conn, util, "parse", 1, 3, "def parse(value):", "Parse a value."),
        "format_value": _symbol(
            conn, util, "format_value", 6, 8, "def format_value(value):", "Format a value."
        ),
        # Deliberately stale signature: slices must come from disk, not the index.
        "render": _symbol(conn, view, "render", 4, 5, "def render():", ""),
        "test_load": _symbol(conn, tests, "test_load", 4, 5, "def test_load():", ""),
        "load_test": _symbol(conn, tests, "load_test", 8, 9, "def load_test():", ""),
    }

    for from_name, to_name in (
        ("save", "parse"),
        ("load", "parse"),
        ("render", "load"),
        ("test_load", "load"),
    ):
        _ref(conn, ids[from_name], to_name)

    conn.executemany(
        "INSERT INTO imports(file_id, module, alias, resolved_file_id) VALUES (?, ?, ?, NULL)",
        [
            (service, "app.util", "parse"),
            (service, "app.util", "format_value"),
            (view, "app.service", "load"),
        ],
    )
    conn.commit()

    resolve_refs(conn)
    return conn, ids


def _slices(packed):
    return [packed.target] + packed.callers + packed.callees + packed.imports + packed.tests


def test_pack_includes_every_section(tmp_path):
    conn, ids = _fixture(tmp_path)
    packed = pack(conn, tmp_path, "app/service.py", "load", budget_tokens=10**6)

    assert packed.target is not None
    assert packed.target.symbol == "load"
    assert "def load(path):" in packed.target.text
    assert "return parse(path)" in packed.target.text

    assert [slice.symbol for slice in packed.callers] == ["render"]
    assert [slice.symbol for slice in packed.callees] == ["parse"]
    assert [slice.symbol for slice in packed.imports] == ["format_value"]
    assert {slice.symbol for slice in packed.tests} == {"test_load", "load_test"}


def test_pack_reads_slice_text_from_disk(tmp_path):
    conn, _ids = _fixture(tmp_path)
    packed = pack(
        conn, tmp_path, "app/service.py", "load", budget_tokens=10**6, full_body=False
    )

    # Index says ``def render():`` but the file says ``def render(path):``.
    caller = packed.callers[0]
    assert caller.text == "def render(path):"

    # Signature slice keeps the docstring's first line, again from disk.
    callee = packed.callees[0]
    assert "def parse(value):" in callee.text
    assert "Parse a value." in callee.text


def test_pack_signature_slice_spans_multiline_declarations(tmp_path):
    source = '''@decorator
def alpha(
    first: int,
    second: str,
) -> None:
    """Alpha doc."""
    return None


def caller():
    return alpha(1, "x")
'''
    (tmp_path / "mod.py").write_text(source, encoding="utf-8", newline="\n")

    conn = _db(tmp_path)
    file_id = _file(conn, "mod.py", source)
    _symbol(
        conn,
        file_id,
        "alpha",
        1,
        7,
        "def alpha( first: int, second: str, ) -> None",
        "Alpha doc.",
    )
    caller = _symbol(conn, file_id, "caller", 10, 11, "def caller()", "")
    _ref(conn, caller, "alpha")
    conn.commit()
    resolve_refs(conn)

    packed = pack(
        conn, tmp_path, "mod.py", "caller", budget_tokens=10**6, full_body=False
    )

    assert [slice.symbol for slice in packed.callees] == ["alpha"]
    text = packed.callees[0].text
    assert "@decorator" in text
    assert "second: str," in text  # the whole multi-line signature
    assert '"""Alpha doc."""' in text  # first docstring line
    assert "return None" not in text  # body stays out


def test_pack_signature_slices_stay_short_without_full_body(tmp_path):
    conn, _ids = _fixture(tmp_path)
    packed = pack(
        conn, tmp_path, "app/service.py", "load", budget_tokens=10**6, full_body=False
    )

    assert "return load(path)" not in packed.callers[0].text
    assert "return value.strip()" not in packed.callees[0].text
    assert "return f\"[{value}]\"" not in packed.imports[0].text


def test_pack_expands_neighbours_when_budget_allows(tmp_path):
    conn, _ids = _fixture(tmp_path)
    packed = pack(conn, tmp_path, "app/service.py", "load", budget_tokens=10**6)

    assert "return load(path)" in packed.callers[0].text
    assert "return value.strip()" in packed.callees[0].text
    assert "return f\"[{value}]\"" in packed.imports[0].text


def test_pack_token_accounting_matches_slices(tmp_path):
    conn, _ids = _fixture(tmp_path)
    packed = pack(conn, tmp_path, "app/service.py", "load", budget_tokens=10**6)

    slices = [item for item in _slices(packed) if item is not None]
    assert packed.token_total == sum(item.tokens for item in slices)
    for item in slices:
        assert item.tokens == tokens.estimate(item.text)

    assert packed.token_baseline == tokens.estimate(_service_text())
    assert packed.savings_pct == tokens.savings_pct(packed.token_total, packed.token_baseline)
    assert packed.worth_using == tokens.worth_using(packed.token_total, packed.token_baseline)
    assert packed.worth_using is True


def test_pack_trims_imports_before_other_sections(tmp_path):
    conn, _ids = _fixture(tmp_path)
    generous = pack(
        conn, tmp_path, "app/service.py", "load", budget_tokens=10**6, full_body=False
    )
    import_tokens = sum(item.tokens for item in generous.imports)
    assert import_tokens > 0

    budget = generous.token_total - import_tokens
    packed = pack(
        conn, tmp_path, "app/service.py", "load", budget_tokens=budget, full_body=False
    )

    assert packed.imports == []
    assert packed.callers and packed.callees and packed.tests
    assert "trimmed 1 import(s)" in packed.note
    assert "over budget" in packed.note


def test_pack_trims_callees_before_callers(tmp_path):
    conn, _ids = _fixture(tmp_path)
    generous = pack(
        conn, tmp_path, "app/service.py", "load", budget_tokens=10**6, full_body=False
    )
    removable = sum(item.tokens for item in generous.imports) + sum(
        item.tokens for item in generous.callees
    )

    packed = pack(
        conn,
        tmp_path,
        "app/service.py",
        "load",
        budget_tokens=generous.token_total - removable,
        full_body=False,
    )

    assert packed.imports == []
    assert packed.callees == []
    assert packed.callers and packed.tests
    assert "trimmed 1 import(s), 1 callee(s)" in packed.note


def test_pack_trims_everything_but_the_target_when_budget_is_tight(tmp_path):
    conn, _ids = _fixture(tmp_path)
    generous = pack(
        conn, tmp_path, "app/service.py", "load", budget_tokens=10**6, full_body=False
    )

    packed = pack(
        conn,
        tmp_path,
        "app/service.py",
        "load",
        budget_tokens=generous.target.tokens,
        full_body=False,
    )

    assert packed.target is not None
    assert "def load(path):" in packed.target.text
    assert packed.callers == []
    assert packed.callees == []
    assert packed.imports == []
    assert packed.tests == []
    assert "1 import(s), 1 callee(s), 1 caller(s), 2 test(s)" in packed.note


def test_pack_notes_when_target_alone_exceeds_budget(tmp_path):
    conn, _ids = _fixture(tmp_path)
    packed = pack(
        conn, tmp_path, "app/service.py", "load", budget_tokens=0, full_body=False
    )

    assert packed.target is not None
    assert packed.callers == []
    assert packed.callees == []
    assert packed.imports == []
    assert packed.tests == []
    assert "target alone exceeds budget 0" in packed.note


def test_pack_can_skip_tests(tmp_path):
    conn, _ids = _fixture(tmp_path)
    packed = pack(
        conn,
        tmp_path,
        "app/service.py",
        "load",
        budget_tokens=10**6,
        include_tests=False,
    )
    assert packed.tests == []
    # A test that calls the target must not leak into the caller section.
    assert [slice.symbol for slice in packed.callers] == ["render"]


def test_pack_defaults_to_first_top_level_symbol(tmp_path):
    conn, _ids = _fixture(tmp_path)
    packed = pack(conn, tmp_path, "app/service.py", None, budget_tokens=10**6)

    assert packed.target is not None
    assert packed.target.symbol == "save"
    assert "def save(path):" in packed.target.text


def test_pack_reports_missing_targets(tmp_path):
    conn, _ids = _fixture(tmp_path)
    packed = pack(conn, tmp_path, "app/service.py", "ghost")

    assert packed.target is None
    assert packed.token_total == 0
    assert packed.worth_using is False
    assert "target not found" in packed.note


def test_pack_normalises_crlf_line_endings(tmp_path):
    raw = "def alpha():\r\n    return 1\r\n"
    (tmp_path / "mod.py").write_bytes(raw.encode("utf-8"))

    conn = _db(tmp_path)
    file_id = _file(conn, "mod.py", "def alpha():\n    return 1\n")
    _symbol(conn, file_id, "alpha", 1, 2, "def alpha():", "")
    conn.commit()

    packed = pack(conn, tmp_path, "mod.py", "alpha")

    assert packed.target is not None
    assert packed.target.text == "def alpha():\n    return 1"
    assert "\r" not in packed.target.text
    # The baseline still reflects the real bytes on disk.
    assert packed.token_baseline == tokens.estimate(raw)
