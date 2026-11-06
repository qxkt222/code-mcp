"""Behavioural tests for the context packer."""

from __future__ import annotations

import sqlite3

from code_mcp import db, tokens
from code_mcp.graph.refs import resolve_refs
from code_mcp.pack import PACK_EXPAND_CAP_TOKENS, pack
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


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line.encode("utf-8")))
    return offsets


def _symbol_span(
    conn: sqlite3.Connection,
    file_id: int,
    text: str,
    name: str,
    start_line: int,
    end_line: int,
    signature: str = "",
    docstring: str = "",
    kind: str = "function",
) -> int:
    """Insert a symbol with byte offsets derived from the real file text."""
    offsets = _line_offsets(text)
    start_byte = offsets[min(start_line - 1, len(offsets) - 1)]
    end_byte = offsets[min(end_line, len(offsets) - 1)]
    cursor = conn.execute(
        """INSERT INTO symbols(file_id, name, qualname, kind, start_line, end_line,
                               start_byte, end_byte, signature, docstring)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            file_id,
            name,
            name,
            kind,
            start_line,
            end_line,
            start_byte,
            end_byte,
            signature,
            docstring,
        ),
    )
    return int(cursor.lastrowid)


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

    # ``resolved_file_id`` is what a real index stores after resolving module
    # paths; refs may only cross files along these edges.
    conn.executemany(
        "INSERT INTO imports(file_id, module, alias, resolved_file_id) VALUES (?, ?, ?, ?)",
        [
            (service, "app.util", "parse", util),
            (service, "app.util", "format_value", util),
            (view, "app.service", "load", service),
            (tests, "app.service", "load", service),
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
    # Imports are never expanded to a full body, even with a huge budget.
    assert "return f\"[{value}]\"" not in packed.imports[0].text


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


# --------------------------------------------------------------------------- #
# Oversized targets (outline + segments)
# --------------------------------------------------------------------------- #


def _big_source(statements: int = 1200, callers: int = 0) -> str:
    lines = ["def big():", '    """Big."""', "    total = 0"]
    lines += [f"    total += {index}" for index in range(statements)]
    lines += ["    return total", ""]
    for index in range(callers):
        lines += ["", f"def caller{index}():", f"    value = {index}"]
        lines += [f"    value += {step}" for step in range(300)]
        lines += ["    return big()", ""]
    return "\n".join(lines) + "\n"


def _write(conn, root, path: str, text: str, lang: str = "python") -> int:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    return _file(conn, path, text, lang)


def _big_fixture(tmp_path, statements: int = 1200, callers: int = 0):
    source = _big_source(statements, callers)
    conn = _db(tmp_path)
    file_id = _write(conn, tmp_path, "big.py", source)
    line_count = len(source.splitlines())
    big_end = 3 + statements + 1
    _symbol_span(conn, file_id, source, "big", 1, big_end, "def big()", "Big.")
    caller_names: list[str] = []
    if callers:
        lines = source.splitlines()
        for index in range(callers):
            start = next(
                line_no
                for line_no, line in enumerate(lines, start=1)
                if line.startswith(f"def caller{index}()")
            )
            name = f"caller{index}"
            caller_id = _symbol_span(
                conn, file_id, source, name, start, line_count, f"def {name}()"
            )
            _ref(conn, caller_id, "big")
            caller_names.append(name)
    conn.commit()
    resolve_refs(conn)
    return conn, source, caller_names


def test_pack_outlines_an_oversized_target(tmp_path):
    conn, source, _ = _big_fixture(tmp_path)

    packed = pack(conn, tmp_path, "big.py", "big")

    assert packed.target is not None
    assert packed.target.truncated is True
    assert len(packed.target.segments) >= 2
    full_text = source.rstrip("\n")
    assert packed.target.total_tokens == tokens.estimate(full_text)
    assert packed.target.tokens < packed.target.total_tokens
    assert "def big()" in packed.target.text
    assert "total += 500" not in packed.target.text
    assert "target split into" in packed.note
    assert "fetch one with slice(file, start_line, end_line)" in packed.note


def test_pack_returns_an_outline_when_budget_is_small(tmp_path):
    conn, _, _ = _big_fixture(tmp_path)

    packed = pack(conn, tmp_path, "big.py", "big", budget_tokens=100)

    assert packed.target is not None
    assert packed.target.truncated is True
    assert packed.target.text.count("\n") < 5  # signature, not the body
    assert "total += 500" not in packed.target.text
    assert packed.token_total < 200


def test_pack_segment_returns_the_requested_block(tmp_path):
    conn, source, _ = _big_fixture(tmp_path)
    outline = pack(conn, tmp_path, "big.py", "big")
    assert outline.target is not None
    last_index = len(outline.target.segments)
    chosen = outline.target.segments[-1]

    packed = pack(conn, tmp_path, "big.py", "big", segment=last_index)

    assert packed.target is not None
    assert (packed.target.start_line, packed.target.end_line) == (
        chosen.start_line,
        chosen.end_line,
    )
    lines = source.splitlines()
    assert packed.target.text == "\n".join(
        lines[chosen.start_line - 1 : chosen.end_line]
    )
    assert "return total" in packed.target.text
    assert packed.target.truncated is False


def test_pack_segment_out_of_range_is_reported(tmp_path):
    conn, _, _ = _big_fixture(tmp_path)

    packed = pack(conn, tmp_path, "big.py", "big", segment=99)

    assert packed.target is None
    assert packed.worth_using is False
    assert "out of range" in packed.note


def test_pack_does_not_segment_a_small_target(tmp_path):
    conn, _ids = _fixture(tmp_path)

    packed = pack(conn, tmp_path, "app/service.py", "load")

    assert packed.target is not None
    assert packed.target.truncated is False
    assert packed.target.segments == []
    assert "target split into" not in packed.note


def test_split_target_skips_neighbour_expansion(tmp_path):
    conn, _, callers = _big_fixture(tmp_path, callers=1)

    packed = pack(conn, tmp_path, "big.py", "big", budget_tokens=10**6)

    assert packed.target is not None and packed.target.truncated is True
    assert [item.symbol for item in packed.callers] == callers
    # The budget freed by the outline must not pull in the caller's body.
    assert "return big()" not in packed.callers[0].text
    assert "expanded" not in packed.note


def _cap_source() -> str:
    lines = ["def target():", "    return 1", ""]
    for index in range(3):
        lines += ["", f"def caller{index}():", f"    value = {index}"]
        lines += [f"    value += {step}" for step in range(400)]
        lines += ["    return target()", ""]
    lines += ["# padding " + "x" * 40 for _ in range(1500)]
    return "\n".join(lines) + "\n"


def test_neighbour_expansion_has_an_absolute_cap(tmp_path):
    source = _cap_source()
    conn = _db(tmp_path)
    file_id = _write(conn, tmp_path, "cap.py", source)
    lines = source.splitlines()
    target_id = _symbol_span(conn, file_id, source, "target", 1, 2, "def target()")
    for index in range(3):
        start = next(
            line_no
            for line_no, line in enumerate(lines, start=1)
            if line.startswith(f"def caller{index}()")
        )
        caller_id = _symbol_span(
            conn, file_id, source, f"caller{index}", start, start + 402,
            f"def caller{index}()",
        )
        _ref(conn, caller_id, "target")
    conn.commit()
    resolve_refs(conn)

    packed = pack(conn, tmp_path, "cap.py", "target", budget_tokens=10**6)

    assert packed.token_total <= PACK_EXPAND_CAP_TOKENS
    assert any(
        "return target()" not in item.text for item in packed.callers
    ), "the cap must stop at least one caller from expanding"


def test_import_without_a_matching_symbol_stays_a_header(tmp_path):
    helpers = "".join(f"def helper_{index}():\n    return {index}\n\n" for index in range(100))
    service = "from app.helpers import add\n\n\ndef run():\n    return add(1)\n"

    conn = _db(tmp_path)
    helpers_id = _write(conn, tmp_path, "app/helpers.py", helpers)
    service_id = _write(conn, tmp_path, "app/service.py", service)
    run_id = _symbol_span(
        conn, service_id, service, "run", 4, 5, "def run()", kind="function"
    )
    conn.execute(
        "INSERT INTO imports(file_id, module, alias, resolved_file_id) VALUES (?, ?, ?, ?)",
        (service_id, "app.helpers", "add", helpers_id),
    )
    conn.commit()
    resolve_refs(conn)

    packed = pack(conn, tmp_path, "app/service.py", "run", budget_tokens=10**6)

    assert [item.file for item in packed.imports] == ["app/helpers.py"]
    assert packed.imports[0].end_line <= 15
    assert "def helper_99" not in packed.imports[0].text


# --------------------------------------------------------------------------- #
# Test discovery prefilter
# --------------------------------------------------------------------------- #


def test_test_prefilter_keeps_multi_part_names(tmp_path):
    target = "def fetch_user_data():\n    return 1\n"
    tests = (
        "def test_fetch_user_data():\n    pass\n\n\n"
        "def fetch_user_data_test():\n    pass\n\n\n"
        "def test_fetch_user_data_roundtrip():\n    pass\n\n\n"
        "def test_data_fetch_user():\n    pass\n\n\n"
        "def helper_fetch_user_data():\n    pass\n"
    )

    conn = _db(tmp_path)
    target_file = _write(conn, tmp_path, "app/api.py", target)
    _symbol_span(
        conn, target_file, target, "fetch_user_data", 1, 2, "def fetch_user_data()"
    )
    tests_file = _write(conn, tmp_path, "tests/test_users.py", tests)
    for name, start in (
        ("test_fetch_user_data", 1),
        ("fetch_user_data_test", 5),
        ("test_fetch_user_data_roundtrip", 9),
        ("test_data_fetch_user", 13),
        ("helper_fetch_user_data", 17),
    ):
        _symbol_span(conn, tests_file, tests, name, start, start + 1, f"def {name}()")
    conn.commit()

    packed = pack(conn, tmp_path, "app/api.py", "fetch_user_data")

    # Underscores are identifier characters here: only the exact variants and
    # the "test_*" prefix form match, and the prefilter must not discard them.
    assert {item.symbol for item in packed.tests} == {
        "test_fetch_user_data",
        "fetch_user_data_test",
    }
