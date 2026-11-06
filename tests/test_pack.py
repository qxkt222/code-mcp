"""Behavioural tests for the context packer."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from code_mcp import db, tokens
from code_mcp.config import load_config
from code_mcp.graph.refs import find_symbol, resolve_refs
from code_mcp.index.builder import _SIZES_META_KEY, build_index
from code_mcp.pack import (
    MIN_PACK_BASELINE_TOKENS,
    PACK_EXPAND_CAP_TOKENS,
    _Reader,
    _signature_piece,
    ensure_fresh,
    pack,
)
from code_mcp.serialize import pack_dict
from code_mcp.types import CONF_EXACT, CONF_NAME, REF_CALL

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
    # Well past MIN_PACK_BASELINE_TOKENS so the file is a realistic pack target
    # rather than being rejected by the small-file short-circuit.
    padding = "\n".join(f"# padding line {index}" for index in range(1, 301))
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


def _ref_to(conn: sqlite3.Connection, from_id: int, to_id: int, name: str) -> None:
    """Insert an already-resolved edge, bypassing import-based resolution."""
    conn.execute(
        """INSERT INTO refs(from_symbol_id, to_symbol_id, to_name, kind, confidence)
           VALUES (?, ?, ?, ?, ?)""",
        (from_id, to_id, name, REF_CALL, CONF_EXACT),
    )


def _mark_fresh(conn: sqlite3.Connection, root: Path, *paths: str) -> None:
    """Record real mtime/size so ``pack`` does not see the fixture as stale."""
    sizes = json.loads(db.get_meta(conn, _SIZES_META_KEY) or "{}")
    for path in paths:
        stat = (root / path).stat()
        conn.execute(
            "UPDATE files SET mtime = ? WHERE path = ?", (stat.st_mtime, path)
        )
        sizes[path] = stat.st_size
    db.set_meta(conn, _SIZES_META_KEY, json.dumps(sizes, sort_keys=True))


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
    _mark_fresh(
        conn,
        tmp_path,
        "app/service.py",
        "app/util.py",
        "app/view.py",
        "tests/test_service.py",
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
    padding = "\n".join(f"# padding line {index}" for index in range(1, 121))
    source = '''@decorator
def alpha(
    first: int,
    second: str,
) -> None:
    """Alpha doc."""
    return None


def caller():
    return alpha(1, "x")


''' + padding + "\n"
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
    _mark_fresh(conn, tmp_path, "mod.py")
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
    # ``worth_using`` is judged on the serialised response, not the bodies.
    assert packed.response_tokens == packed.token_total + tokens.envelope_tokens(
        [(item.file, item.symbol) for item in slices]
    )
    assert packed.response_tokens > packed.token_total
    assert packed.worth_using == tokens.worth_using(
        packed.response_tokens, packed.token_baseline
    )
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


def test_pack_rejects_files_below_the_baseline_floor(tmp_path):
    conn, _ids = _fixture(tmp_path)
    packed = pack(conn, tmp_path, "app/util.py", "parse")

    assert packed.target is None
    assert packed.token_total == 0
    assert packed.response_tokens == 0
    assert packed.token_baseline == tokens.estimate(UTIL)
    assert packed.token_baseline < 300
    assert packed.worth_using is False
    assert f"file is only {packed.token_baseline} tokens" in packed.note
    assert "read it directly" in packed.note


def test_pack_note_explains_when_envelope_breaks_the_deal(tmp_path, monkeypatch):
    conn, _ids = _fixture(tmp_path)
    # Force the envelope to dominate while leaving the bodies untouched.
    monkeypatch.setattr(tokens, "envelope_tokens", lambda slices: 10**6)

    packed = pack(conn, tmp_path, "app/service.py", "load", budget_tokens=10**6)

    assert packed.worth_using is False
    assert packed.savings_pct > 0  # the bodies alone are smaller than the file
    # The payload is withheld entirely rather than sent with a warning.
    assert packed.target is None
    assert "would cost" in packed.note
    assert "read the file directly" in packed.note


def test_pack_dict_exposes_envelope_accounting(tmp_path):
    conn, _ids = _fixture(tmp_path)
    packed = pack(conn, tmp_path, "app/service.py", "load", budget_tokens=10**6)
    data = pack_dict(packed)

    assert data["tokens_used"] == packed.token_total
    assert data["response_tokens"] == packed.response_tokens
    assert data["savings_pct"] == packed.savings_pct
    assert data["effective_savings_pct"] == tokens.savings_pct(
        packed.response_tokens, packed.token_baseline
    )

    small = pack_dict(pack(conn, tmp_path, "app/util.py", "parse"))
    assert small["worth_using"] is False
    assert small["response_tokens"] == 0
    assert small["effective_savings_pct"] == 0.0


def test_pack_normalises_crlf_line_endings(tmp_path):
    padding = "".join(f"# padding line {index}\r\n" for index in range(1, 121))
    raw = "def alpha():\r\n    return 1\r\n" + padding
    (tmp_path / "mod.py").write_bytes(raw.encode("utf-8"))

    conn = _db(tmp_path)
    file_id = _file(conn, "mod.py", "def alpha():\n    return 1\n")
    _symbol(conn, file_id, "alpha", 1, 2, "def alpha():", "")
    _mark_fresh(conn, tmp_path, "mod.py")
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
    file_id = _file(conn, path, text, lang)
    _mark_fresh(conn, root, path)
    return file_id


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
    padding = "\n".join(f"# padding line {index}" for index in range(1, 121))
    service = (
        "from app.helpers import add\n\n\ndef run():\n    return add(1)\n\n" + padding + "\n"
    )

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
    padding = "\n".join(f"# padding line {index}" for index in range(1, 121))
    target = "def fetch_user_data():\n    return 1\n\n" + padding + "\n"
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


# --------------------------------------------------------------------------- #
# Index freshness (external edits)
# --------------------------------------------------------------------------- #

_PADDING_300 = "\n".join(f"# padding line {index}" for index in range(1, 301))


def _built_index(tmp_path, path: str, source: str):
    """Write ``source`` and index it through the real builder."""
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8", newline="\n")
    conn = _db(tmp_path)
    build_index(tmp_path, conn, load_config(tmp_path))
    conn.commit()
    return conn, target


def test_pack_refreshes_a_file_changed_outside_the_index(tmp_path):
    source = (
        'def first():\n    return "first-original"\n\n\n'
        'def second():\n    return "second-original"\n\n\n'
        'def third():\n    return "third-original"\n\n\n'
        + _PADDING_300
        + "\n"
    )
    conn, target = _built_index(tmp_path, "svc.py", source)

    before = pack(conn, tmp_path, "svc.py", "second")
    assert before.target is not None
    assert before.target.text == 'def second():\n    return "second-original"'

    # An edit that never went through ``edit`` shifts every byte offset below
    # it; the stale index would slice ``first`` instead of ``second``.
    target.write_text(
        '"""Service."""\n\n\n# inserted outside the index\n# another line\n\n\n'
        + source,
        encoding="utf-8",
        newline="\n",
    )

    after = pack(conn, tmp_path, "svc.py", "second")

    assert after.target is not None
    assert after.target.text == 'def second():\n    return "second-original"'
    assert "first-original" not in after.target.text


def test_pack_reports_a_symbol_deleted_outside_the_index(tmp_path):
    source = (
        "def first():\n    return 1\n\n\n"
        "def second():\n    return 2\n\n\n"
        + _PADDING_300
        + "\n"
    )
    conn, target = _built_index(tmp_path, "svc.py", source)
    assert pack(conn, tmp_path, "svc.py", "second").target is not None

    target.write_text(
        "def first():\n    return 1\n\n\n" + _PADDING_300 + "\n",
        encoding="utf-8",
        newline="\n",
    )

    packed = pack(conn, tmp_path, "svc.py", "second")

    # The vanished symbol must be reported, never replaced by a neighbour.
    assert packed.target is None
    assert packed.worth_using is False
    assert "target not found" in packed.note


def test_pack_only_reindexes_after_real_drift(tmp_path, monkeypatch):
    from code_mcp.index import builder

    calls: list[str] = []
    original = builder.reindex_file

    def counting(root, conn, rel):
        calls.append(rel)
        return original(root, conn, rel)

    monkeypatch.setattr(builder, "reindex_file", counting)
    source = (
        "def first():\n    return 1\n\n\n"
        "def second():\n    return 2\n\n\n"
        + _PADDING_300
        + "\n"
    )
    conn, target = _built_index(tmp_path, "svc.py", source)

    assert pack(conn, tmp_path, "svc.py", "second").target is not None
    assert calls == []  # an unchanged file is never re-parsed

    target.write_text(
        "# shifted\n" + source, encoding="utf-8", newline="\n"
    )
    assert pack(conn, tmp_path, "svc.py", "second").target is not None
    assert calls == ["svc.py"]

    # The reindex updated mtime/size, so the next pack is stable again.
    assert pack(conn, tmp_path, "svc.py", "second").target is not None
    assert calls == ["svc.py"]


def test_ensure_fresh_returns_false_when_the_file_is_gone(tmp_path):
    source = "def alpha():\n    return 1\n\n" + _PADDING_300 + "\n"
    (tmp_path / "mod.py").write_text(source, encoding="utf-8", newline="\n")
    conn = _db(tmp_path)
    build_index(tmp_path, conn, load_config(tmp_path))
    conn.commit()

    assert ensure_fresh(conn, tmp_path, "mod.py") is True
    (tmp_path / "mod.py").unlink()

    assert ensure_fresh(conn, tmp_path, "mod.py") is False


# --------------------------------------------------------------------------- #
# Relevance ordering
# --------------------------------------------------------------------------- #


def test_callers_and_callees_are_ranked_by_distance(tmp_path):
    target_text = (
        "def target():\n    return 1\n\n\n"
        "def same_file_caller():\n    return target()\n\n\n"
        "def same_file_callee():\n    return 2\n\n\n"
        + _PADDING_300
        + "\n"
    )
    dir_text = (
        "def same_dir_caller_late():\n    return target()\n\n\n"
        "def same_dir_caller_early():\n    return target()\n\n\n"
        "def same_dir_callee():\n    return 2\n\n\n"
        + _PADDING_300
        + "\n"
    )
    pkg_text = (
        "def same_pkg_caller():\n    return target()\n\n\n"
        "def same_pkg_callee():\n    return 2\n\n\n"
        + _PADDING_300
        + "\n"
    )
    remote_text = (
        "def remote_caller():\n    return target()\n\n\n"
        "def remote_callee():\n    return 2\n\n\n"
        + _PADDING_300
        + "\n"
    )

    conn = _db(tmp_path)
    files = {
        "pkg/a.py": _write(conn, tmp_path, "pkg/a.py", target_text),
        "pkg/b.py": _write(conn, tmp_path, "pkg/b.py", dir_text),
        "pkg/sub/c.py": _write(conn, tmp_path, "pkg/sub/c.py", pkg_text),
        "other/d.py": _write(conn, tmp_path, "other/d.py", remote_text),
    }
    texts = {
        "pkg/a.py": target_text,
        "pkg/b.py": dir_text,
        "pkg/sub/c.py": pkg_text,
        "other/d.py": remote_text,
    }

    def add(path: str, name: str, start: int) -> int:
        return _symbol_span(
            conn, files[path], texts[path], name, start, start + 1, f"def {name}()"
        )

    target_id = add("pkg/a.py", "target", 1)
    callers = {
        "same_file_caller": add("pkg/a.py", "same_file_caller", 5),
        "same_dir_caller_late": add("pkg/b.py", "same_dir_caller_late", 1),
        "same_dir_caller_early": add("pkg/b.py", "same_dir_caller_early", 5),
        "same_pkg_caller": add("pkg/sub/c.py", "same_pkg_caller", 1),
        "remote_caller": add("other/d.py", "remote_caller", 1),
    }
    callees = {
        "same_file_callee": add("pkg/a.py", "same_file_callee", 9),
        "same_dir_callee": add("pkg/b.py", "same_dir_callee", 9),
        "same_pkg_callee": add("pkg/sub/c.py", "same_pkg_callee", 5),
        "remote_callee": add("other/d.py", "remote_callee", 5),
    }
    for from_id in callers.values():
        _ref_to(conn, from_id, target_id, "target")
    for name, callee_id in callees.items():
        _ref_to(conn, target_id, callee_id, name)
    conn.commit()

    packed = pack(
        conn,
        tmp_path,
        "pkg/a.py",
        "target",
        budget_tokens=10**6,
        include_tests=False,
    )

    # Same file > same directory > same top-level package > everywhere else;
    # within one tier, earlier lines win.
    assert [item.symbol for item in packed.callers] == [
        "same_file_caller",
        "same_dir_caller_late",
        "same_dir_caller_early",
        "same_pkg_caller",
        "remote_caller",
    ]
    assert [item.symbol for item in packed.callees] == [
        "same_file_callee",
        "same_dir_callee",
        "same_pkg_callee",
        "remote_callee",
    ]


# --------------------------------------------------------------------------- #
# Forced packs
# --------------------------------------------------------------------------- #


def test_force_returns_a_payload_below_the_baseline_floor(tmp_path):
    conn, _ids = _fixture(tmp_path)
    assert tokens.estimate(UTIL) < MIN_PACK_BASELINE_TOKENS

    packed = pack(conn, tmp_path, "app/util.py", "parse", force=True)

    assert packed.target is not None
    assert "def parse(value):" in packed.target.text
    assert packed.token_baseline < MIN_PACK_BASELINE_TOKENS
    assert packed.response_tokens > 0
    assert packed.savings_pct == tokens.savings_pct(
        packed.token_total, packed.token_baseline
    )
    assert packed.worth_using is False
    assert packed.note.startswith("forced: ")
    assert "read the file directly" in packed.note


def test_force_returns_a_payload_the_envelope_would_withhold(tmp_path, monkeypatch):
    conn, _ids = _fixture(tmp_path)
    # Force the envelope to dominate while leaving the bodies untouched.
    monkeypatch.setattr(tokens, "envelope_tokens", lambda slices: 10**6)

    packed = pack(conn, tmp_path, "app/service.py", "load", budget_tokens=10**6, force=True)

    assert packed.target is not None
    assert "def load(path):" in packed.target.text
    assert packed.response_tokens > packed.token_baseline
    assert packed.savings_pct > 0  # the bodies alone are smaller than the file
    assert packed.worth_using is False
    assert packed.note.startswith("forced: ")
    assert "not recommended" in packed.note


def test_forced_segment_notes_that_it_was_forced(tmp_path):
    conn, _ids = _fixture(tmp_path)

    packed = pack(conn, tmp_path, "app/service.py", "load", segment=1, force=True)

    assert packed.target is not None
    assert packed.note.startswith("forced: ")


# --------------------------------------------------------------------------- #
# Clipped signatures (arrow-function constants)
# --------------------------------------------------------------------------- #

# Ten-line arrow-function head whose indexed signature is clipped at 200 chars:
# the collapsed head ends with ``merge?: (left:...``, so exact substring
# matching cannot find the declaration end. The body is closed by ``};``, the
# point an unordered brace check used to swallow it into the signature slice.
_ARROW_HEAD_END = 10
_ARROW_BODY_END = 16

_ARROW_SOURCE = '''export const pick = <
  T extends Record<string, unknown>,
  K extends keyof T,
  R extends Partial<Pick<T, K>>,
  S extends Record<K, unknown> = Record<K, unknown>,
>(
  source: T,
  keys: K[],
  merge?: (left: R, right: R) => R,
): R => {
  const result = {} as R
  for (const key of keys) {
    result[key] = source[key]
  }
  return result
};

export const other = 1;

''' + "\n".join(f"// padding line {index}" for index in range(1, 121)) + "\n"


def test_clipped_arrow_signature_stops_at_the_head(tmp_path):
    conn, _target = _built_index(
        tmp_path, "src/stores/useSimulationStore.ts", _ARROW_SOURCE
    )
    symbol = find_symbol(conn, "src/stores/useSimulationStore.ts", "pick")
    assert symbol is not None
    assert symbol.signature.endswith("...")  # the index clips the long head

    assert symbol.end_line == _ARROW_BODY_END
    piece = _signature_piece(symbol, _Reader(tmp_path))

    assert piece.start_line == 1
    assert piece.end_line == _ARROW_HEAD_END
    assert piece.text.rstrip().endswith("): R => {")
    # The arrow body must never be swallowed into the signature slice.
    assert "const result = {} as R" not in piece.text
    assert "return result" not in piece.text
    assert "export const other" not in piece.text


# --------------------------------------------------------------------------- #
# CLI gate --json
# --------------------------------------------------------------------------- #


def test_cli_gate_json_matches_the_server_shape():
    from code_mcp import cli
    from code_mcp.types import GateResult, GateStep

    result = GateResult(
        steps=[
            GateStep(
                name="unit",
                command="pytest -q",
                exit_code=0,
                duration_s=1.234,
                output="ok",
                passed=True,
            )
        ]
    )

    payload = cli._gate_payload(result, detected=True)

    assert payload["passed"] is True
    assert payload["detected_config"] is True
    assert payload["steps"] == [
        {
            "name": "unit",
            "command": "pytest -q",
            "passed": True,
            "exit_code": 0,
            "duration_s": 1.23,
            "output": "ok",
        }
    ]
