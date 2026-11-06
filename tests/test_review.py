"""External-linter and structural-rule behaviour of the review layer."""

from __future__ import annotations

import os
import shlex
import stat
from pathlib import Path

import pytest

from code_mcp import db
from code_mcp.review.external import detect_tools, run_external
from code_mcp.review.structural import analyze


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(db.DB_HOME_ENV, str(tmp_path / "_home"))


# --------------------------------------------------------------------------- #
# external tools
# --------------------------------------------------------------------------- #


def test_detect_tools_empty(tmp_path: Path) -> None:
    assert detect_tools(tmp_path) == []


def test_detect_tools_python_project(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    assert detect_tools(tmp_path) == ["ruff", "mypy", "bandit"]


def test_detect_tools_script_only(tmp_path: Path) -> None:
    (tmp_path / "script.py").write_text("print(1)\n", encoding="utf-8")
    assert detect_tools(tmp_path) == ["ruff", "mypy", "bandit"]


def test_detect_tools_mixed_project(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'x'\n", encoding="utf-8")
    assert detect_tools(tmp_path) == ["ruff", "mypy", "bandit", "tsc", "cargo"]


def test_run_external_missing_tool(tmp_path: Path) -> None:
    findings, ran, missing = run_external(tmp_path, ["definitely-not-a-real-tool-xyz"])
    assert findings == []
    assert ran == []
    assert missing == ["definitely-not-a-real-tool-xyz"]


def test_run_external_empty_tools(tmp_path: Path) -> None:
    assert run_external(tmp_path, []) == ([], [], [])


def _write_fake_tool(
    bin_dir: Path, name: str, stdout: str, exit_code: int = 0, to_stderr: bool = False
) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        redirect = " 1>&2" if to_stderr else ""
        script = bin_dir / f"{name}.cmd"
        script.write_text(
            f"@echo off\necho {stdout}{redirect}\nexit /b {exit_code}\n",
            encoding="ascii",
        )
        return
    redirect = " >&2" if to_stderr else ""
    script = bin_dir / name
    script.write_text(
        f"#!/bin/sh\nprintf '%s\\n' {shlex.quote(stdout)}{redirect}\nexit {exit_code}\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)


def _fake_tool_on_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    stdout: str,
    exit_code: int = 0,
    to_stderr: bool = False,
) -> None:
    bin_dir = tmp_path / "fake-bin"
    _write_fake_tool(bin_dir, name, stdout, exit_code, to_stderr)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))


def test_run_external_parses_ruff_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = (
        '[{"code":"F401","message":"unused import","filename":"a.py",'
        '"location":{"row":3,"column":1}}]'
    )
    _fake_tool_on_path(tmp_path, monkeypatch, "ruff", payload, exit_code=1)

    findings, ran, missing = run_external(tmp_path, ["ruff"])

    assert ran == ["ruff"]
    assert missing == []
    assert len(findings) == 1
    assert findings[0].rule == "ruff:F401"
    assert findings[0].severity == "P1"
    assert findings[0].file == "a.py"
    assert findings[0].line == 3


def test_run_external_parses_mypy_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_tool_on_path(
        tmp_path,
        monkeypatch,
        "mypy",
        "a.py:3: error: Incompatible types [assignment]",
        exit_code=1,
    )

    findings, ran, _ = run_external(tmp_path, ["mypy"])

    assert ran == ["mypy"]
    assert len(findings) == 1
    assert findings[0].rule == "mypy:assignment"
    assert findings[0].severity == "P1"
    assert findings[0].line == 3


def test_run_external_parses_bandit_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = (
        '{"results": [{"filename": "svc.py", "line_number": 7, '
        '"issue_text": "subprocess call", "issue_severity": "HIGH", '
        '"test_id": "B602"}]}'
    )
    _fake_tool_on_path(tmp_path, monkeypatch, "bandit", payload, exit_code=1)

    findings, ran, _ = run_external(tmp_path, ["bandit"])

    assert ran == ["bandit"]
    assert len(findings) == 1
    assert findings[0].rule == "bandit:B602"
    assert findings[0].severity == "P1"
    assert findings[0].line == 7


def test_run_external_parses_tsc_and_cargo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_tool_on_path(
        tmp_path,
        monkeypatch,
        "tsc",
        "src/a.ts(4,5): error TS2322: Type 'string' is not assignable to type 'number'.",
        exit_code=1,
    )
    _fake_tool_on_path(
        tmp_path,
        monkeypatch,
        "cargo",
        "src/main.rs:2:5: warning: unused variable",
        exit_code=1,
    )

    findings, ran, missing = run_external(tmp_path, ["tsc", "cargo"])

    assert ran == ["tsc", "cargo"]
    assert missing == []
    rules = {finding.rule for finding in findings}
    assert "tsc:TS2322" in rules
    assert "cargo" in rules
    tsc = next(finding for finding in findings if finding.rule == "tsc:TS2322")
    assert tsc.line == 4 and tsc.severity == "P1"
    cargo = next(finding for finding in findings if finding.rule == "cargo")
    assert cargo.line == 2 and cargo.severity == "P2"


def test_run_external_reads_cargo_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_tool_on_path(
        tmp_path,
        monkeypatch,
        "cargo",
        "src/main.rs:2:5: error[E0308]: mismatched types",
        exit_code=1,
        to_stderr=True,
    )

    findings, ran, _ = run_external(tmp_path, ["cargo"])

    assert ran == ["cargo"]
    assert len(findings) == 1
    assert findings[0].rule == "cargo:E0308"
    assert findings[0].severity == "P1"


def test_run_external_continues_when_one_tool_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_tool_on_path(
        tmp_path,
        monkeypatch,
        "ruff",
        '[{"code":"E501","message":"line too long","filename":"a.py",'
        '"location":{"row":1,"column":1}}]',
    )

    findings, ran, missing = run_external(tmp_path, ["missing-tool-abc", "ruff"])

    assert ran == ["ruff"]
    assert missing == ["missing-tool-abc"]
    assert len(findings) == 1


# --------------------------------------------------------------------------- #
# structural rules
# --------------------------------------------------------------------------- #


def _add_file(
    conn, root: Path, rel: str, text: str, lang: str = "python"
) -> tuple[int, bytes]:
    data = text.encode("utf-8")
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    cursor = conn.execute(
        "INSERT INTO files(path, lang, mtime, hash, loc) VALUES (?, ?, ?, ?, ?)",
        (
            rel,
            lang,
            path.stat().st_mtime,
            "0" * 64,
            data.count(b"\n"),
        ),
    )
    return int(cursor.lastrowid), data


def _add_symbol(
    conn,
    file_id: int,
    data: bytes,
    name: str,
    start_marker: str,
    end_marker: str,
    *,
    kind: str = "function",
    qualname: str | None = None,
    docstring: str = "",
    start_byte: int | None = None,
    end_byte: int | None = None,
) -> int:
    if start_byte is None:
        start_byte = data.index(start_marker.encode("utf-8"))
    if end_byte is None:
        end_byte = data.index(end_marker.encode("utf-8"), start_byte) + len(
            end_marker.encode("utf-8")
        )
    start_line = data[:start_byte].count(b"\n") + 1
    end_line = data[:end_byte].count(b"\n") + 1
    cursor = conn.execute(
        "INSERT INTO symbols(file_id, name, qualname, kind, start_line, end_line, "
        "start_byte, end_byte, signature, docstring) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            file_id,
            name,
            qualname or name,
            kind,
            start_line,
            end_line,
            start_byte,
            end_byte,
            "",
            docstring,
        ),
    )
    return int(cursor.lastrowid)


def test_long_function(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    body = "".join(f"    value_{index} = {index}\n" for index in range(80))
    text = "def long_one():\n" + body + "    return 0\n"
    file_id, data = _add_file(conn, tmp_path, "pkg/long.py", text)
    _add_symbol(conn, file_id, data, "long_one", "def long_one", "return 0")
    conn.commit()

    findings = [finding for finding in analyze(conn) if finding.rule == "long-function"]

    # 82 lines over the Python limit of 80, but under twice the limit.
    assert len(findings) == 1
    assert findings[0].severity == "P2"
    assert findings[0].symbol == "long_one"
    assert findings[0].file == "pkg/long.py"


def test_deep_nesting(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    nested = "".join("    " * (level + 1) + "if x:\n" for level in range(6))
    text = "def deep_one(x):\n" + nested + "    " * 7 + "return 1\n"
    file_id, data = _add_file(conn, tmp_path, "pkg/deep.py", text)
    _add_symbol(conn, file_id, data, "deep_one", "def deep_one", "return 1")
    conn.commit()

    findings = [finding for finding in analyze(conn) if finding.rule == "deep-nesting"]

    assert len(findings) == 1
    assert findings[0].severity == "P1"
    assert findings[0].symbol == "deep_one"


def test_high_fanout(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    text = "def caller():\n    pass\n\ndef calm():\n    pass\n"
    file_id, data = _add_file(conn, tmp_path, "pkg/calls.py", text)
    caller_id = _add_symbol(conn, file_id, data, "caller", "def caller", "pass")
    calm_id = _add_symbol(conn, file_id, data, "calm", "def calm", "pass")
    for index in range(16):
        conn.execute(
            "INSERT INTO refs(from_symbol_id, to_symbol_id, to_name, kind, confidence) "
            "VALUES (?, NULL, ?, 'call', 'name_match')",
            (caller_id, f"callee_{index}"),
        )
    for index in range(15):
        conn.execute(
            "INSERT INTO refs(from_symbol_id, to_symbol_id, to_name, kind, confidence) "
            "VALUES (?, NULL, ?, 'call', 'name_match')",
            (calm_id, f"other_{index}"),
        )
    conn.commit()

    findings = [finding for finding in analyze(conn) if finding.rule == "high-fanout"]

    assert len(findings) == 1
    assert findings[0].symbol == "caller"
    assert findings[0].severity == "P2"


def test_high_fanin(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    hub_file, hub_data = _add_file(conn, tmp_path, "pkg/hub.py", "def hub():\n    pass\n")
    hub_id = _add_symbol(conn, hub_file, hub_data, "hub", "def hub", "pass")

    caller_text = "".join(f"def caller_{index}():\n    pass\n" for index in range(21))
    caller_file, caller_data = _add_file(conn, tmp_path, "pkg/callers.py", caller_text)
    for index in range(21):
        caller_id = _add_symbol(
            conn,
            caller_file,
            caller_data,
            f"caller_{index}",
            f"def caller_{index}()",
            "pass",
        )
        conn.execute(
            "INSERT INTO refs(from_symbol_id, to_symbol_id, to_name, kind, confidence) "
            "VALUES (?, ?, 'hub', 'call', 'name_match')",
            (caller_id, hub_id),
        )
    conn.commit()

    findings = [finding for finding in analyze(conn) if finding.rule == "high-fanin"]

    assert len(findings) == 1
    assert findings[0].symbol == "hub"
    assert "21" in findings[0].message


def test_duplicate_body(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    body = "\n".join(f"    total = total + {index}" for index in range(1, 12))
    text = (
        f"def alpha():\n{body}\n    return total\n"
        f"\n\n"
        f"def beta():\n{body}\n    return total\n"
    )
    file_id, data = _add_file(conn, tmp_path, "pkg/dup.py", text)
    _add_symbol(conn, file_id, data, "alpha", "def alpha", "return total")
    _add_symbol(conn, file_id, data, "beta", "def beta", "return total")
    conn.commit()

    findings = [finding for finding in analyze(conn) if finding.rule == "duplicate-body"]

    assert len(findings) == 1
    assert findings[0].symbol == "beta"
    assert "alpha" in findings[0].message
    assert findings[0].severity == "P2"


def test_missing_docstring(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    text = (
        'def documented():\n    """Does a thing."""\n    return 1\n'
        "\n"
        "def undocumented():\n    return 2\n"
    )
    file_id, data = _add_file(conn, tmp_path, "pkg/docs.py", text)
    _add_symbol(
        conn, file_id, data, "documented", "def documented", "return 1", docstring="Does a thing."
    )
    _add_symbol(conn, file_id, data, "undocumented", "def undocumented", "return 2")
    conn.commit()

    findings = [finding for finding in analyze(conn) if finding.rule == "missing-docstring"]

    assert len(findings) == 1
    assert findings[0].symbol == "undocumented"
    assert findings[0].severity == "P3"


def test_no_tests(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    service_text = (
        "def covered():\n    return 1\n"
        "\n"
        "def uncovered():\n    return 2\n"
    )
    service_file, service_data = _add_file(conn, tmp_path, "pkg/svc.py", service_text)
    _add_symbol(conn, service_file, service_data, "covered", "def covered", "return 1")
    _add_symbol(conn, service_file, service_data, "uncovered", "def uncovered", "return 2")

    test_text = "def test_covered():\n    assert True\n"
    test_file, test_data = _add_file(conn, tmp_path, "pkg/tests/test_svc.py", test_text)
    _add_symbol(
        conn, test_file, test_data, "test_covered", "def test_covered", "assert True"
    )
    conn.commit()

    findings = [finding for finding in analyze(conn) if finding.rule == "no-tests"]

    assert [finding.symbol for finding in findings] == ["uncovered"]
    assert findings[0].severity == "P3"


def test_files_filter(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    text_a = "def alpha():\n    return 1\n"
    text_b = "def beta():\n    return 2\n"
    file_a, data_a = _add_file(conn, tmp_path, "pkg/a.py", text_a)
    file_b, data_b = _add_file(conn, tmp_path, "pkg/b.py", text_b)
    _add_symbol(conn, file_a, data_a, "alpha", "def alpha", "return 1")
    _add_symbol(conn, file_b, data_b, "beta", "def beta", "return 2")
    conn.commit()

    findings = analyze(conn, ["pkg/a.py"])

    assert findings
    assert all(finding.file == "pkg/a.py" for finding in findings)
    assert any(finding.symbol == "alpha" for finding in findings)
    assert not any(finding.symbol == "beta" for finding in findings)


def test_missing_source_file_does_not_crash(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    file_id, data = _add_file(conn, tmp_path, "pkg/gone.py", "def gone():\n    return 1\n")
    _add_symbol(conn, file_id, data, "gone", "def gone", "return 1")
    conn.commit()
    (tmp_path / "pkg/gone.py").unlink()

    findings = analyze(conn)

    assert any(finding.rule == "missing-docstring" for finding in findings)
