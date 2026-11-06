"""Calibration of the structural rules: language limits, real nesting, tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from code_mcp import db
from code_mcp.review.structural import analyze


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(db.DB_HOME_ENV, str(tmp_path / "_home"))


def _add_file(
    conn, root: Path, rel: str, text: str, lang: str = "python"
) -> tuple[int, bytes]:
    data = text.encode("utf-8")
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    cursor = conn.execute(
        "INSERT INTO files(path, lang, mtime, hash, loc) VALUES (?, ?, ?, ?, ?)",
        (rel, lang, path.stat().st_mtime, "0" * 64, data.count(b"\n")),
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
    docstring: str = "",
) -> int:
    start_byte = data.index(start_marker.encode("utf-8"))
    end_byte = data.index(end_marker.encode("utf-8"), start_byte) + len(
        end_marker.encode("utf-8")
    )
    start_line = data[:start_byte].count(b"\n") + 1
    end_line = data[:end_byte].count(b"\n") + 1
    cursor = conn.execute(
        "INSERT INTO symbols(file_id, name, qualname, kind, start_line, end_line, "
        "start_byte, end_byte, signature, docstring) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (file_id, name, name, kind, start_line, end_line, start_byte, end_byte, "", docstring),
    )
    return int(cursor.lastrowid)


def _add_function_file(conn, root: Path, rel: str, lang: str, total_lines: int) -> None:
    """Index a function symbol exactly ``total_lines`` lines long."""
    if lang == "python":
        header, filler, footer, end_marker = (
            "def huge():\n",
            "    value = 1\n",
            "    return 0\n",
            "return 0",
        )
    else:
        header, filler, footer, end_marker = (
            "function huge() {\n",
            "  const value = 1;\n",
            "}\n",
            "}",
        )
    text = header + filler * (total_lines - 2) + footer
    file_id, data = _add_file(conn, root, rel, text, lang)
    _add_symbol(conn, file_id, data, "huge", header.strip(), end_marker)


def _rule(conn, rule: str):
    return [finding for finding in analyze(conn) if finding.rule == rule]


def test_long_function_p1_requires_twice_the_language_limit(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    _add_function_file(conn, tmp_path, "pkg/medium.py", "python", 120)
    _add_function_file(conn, tmp_path, "pkg/huge.py", "python", 200)
    _add_function_file(conn, tmp_path, "src/Medium.ts", "typescript", 200)
    _add_function_file(conn, tmp_path, "src/Long.ts", "typescript", 400)
    _add_function_file(conn, tmp_path, "src/Medium.tsx", "tsx", 300)
    _add_function_file(conn, tmp_path, "src/Long.tsx", "tsx", 600)
    conn.commit()

    findings = {finding.file: finding for finding in _rule(conn, "long-function")}

    # Python and Rust keep the 80-line limit, TS 150 and TSX 250.
    assert findings["pkg/medium.py"].severity == "P2"
    assert findings["pkg/huge.py"].severity == "P1"
    assert findings["src/Medium.ts"].severity == "P2"
    assert findings["src/Long.ts"].severity == "P1"
    assert findings["src/Medium.tsx"].severity == "P2"
    assert findings["src/Long.tsx"].severity == "P1"
    assert "(limit 250)" in findings["src/Medium.tsx"].message


def test_deep_nesting_ignores_jsx_attribute_indentation(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    text = (
        "function Card() {\n"
        "  return (\n"
        "    <section\n"
        "      className=\"card\"\n"
        "      data-a={1}\n"
        "      data-b={2}\n"
        "      data-c={3}\n"
        "    >\n"
        "      <span>hi</span>\n"
        "    </section>\n"
        "  );\n"
        "}\n"
    )
    file_id, data = _add_file(conn, tmp_path, "src/Card.tsx", text, "tsx")
    _add_symbol(conn, file_id, data, "Card", "function Card", ");\n}")
    conn.commit()

    assert _rule(conn, "deep-nesting") == []


def test_deep_nesting_ignores_rust_macro_continuation(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    text = (
        "fn build() -> String {\n"
        "    format!(\n"
        "        \"{}{}\",\n"
        "        alpha,\n"
        "        beta,\n"
        "    )\n"
        "}\n"
    )
    file_id, data = _add_file(conn, tmp_path, "src/format.rs", text, "rust")
    _add_symbol(conn, file_id, data, "build", "fn build", "}")
    conn.commit()

    assert _rule(conn, "deep-nesting") == []


def test_deep_nesting_reports_real_control_flow(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    nested = "".join("    " * (level + 1) + "if x:\n" for level in range(6))
    text = "def deep(x):\n" + nested + "    " * 7 + "return 1\n"
    file_id, data = _add_file(conn, tmp_path, "pkg/deep.py", text)
    _add_symbol(conn, file_id, data, "deep", "def deep", "return 1")
    conn.commit()

    findings = _rule(conn, "deep-nesting")

    assert len(findings) == 1
    assert findings[0].symbol == "deep"
    assert findings[0].severity == "P1"
    # Six levels of real 4-space Python nesting must still count as six.
    assert "depth 6" in findings[0].message


def test_deep_nesting_uses_two_space_indent_for_typescript(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    nested = "".join("  " * (level + 1) + "if (x) {\n" for level in range(6))
    text = (
        "function deep(x: number) {\n"
        + nested
        + "  " * 7
        + "return 1;\n"
        + "".join("  " * (level + 1) + "}\n" for level in reversed(range(6)))
        + "}\n"
    )
    file_id, data = _add_file(conn, tmp_path, "src/deep.ts", text, "typescript")
    _add_symbol(conn, file_id, data, "deep", "function deep", "\n}")
    conn.commit()

    findings = _rule(conn, "deep-nesting")

    # Two-space indentation must not halve the measured depth.
    assert len(findings) == 1
    assert "depth 6" in findings[0].message


def test_no_tests_is_skipped_for_typescript(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    text = "export function helper() { return 1; }\n"
    file_id, data = _add_file(conn, tmp_path, "src/helper.ts", text, "typescript")
    _add_symbol(conn, file_id, data, "helper", "export function helper", "}")
    conn.commit()

    assert _rule(conn, "no-tests") == []


def test_dot_test_suffix_is_recognised_as_test_path(tmp_path: Path) -> None:
    conn = db.open_index(tmp_path)
    text = "export function helper() { return 1; }\n"
    file_id, data = _add_file(conn, tmp_path, "src/helper.test.tsx", text, "tsx")
    _add_symbol(conn, file_id, data, "helper", "export function helper", "}")
    conn.commit()

    # Test files are exempt from the docstring rule; the old local predicate
    # only knew Python naming and would have flagged this file.
    assert _rule(conn, "missing-docstring") == []
