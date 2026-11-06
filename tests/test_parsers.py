"""Parser behaviour: symbols, byte offsets, imports, refs and comments."""

from __future__ import annotations

from pathlib import Path

from code_mcp.index.parsers import detect_language, parse_file, parse_source
from code_mcp.types import REF_CALL, REF_IMPORT, REF_INHERIT, REF_TYPE

PY_SOURCE = '''"""Module docstring."""
# module comment
import os.path
import numpy as np
from pkg.models import User as U
from .local import helper

# 中文注释：常量
MAX_SIZE = 10

def decorator(func):
    return func

@decorator
async def outer(a: int, b=1) -> Result:
    """Outer doc."""
    def inner():
        return helper(a)
    return inner()

class Base:
    pass

class Store(Base):
    """Store doc."""
    LIMIT = 5

    def __init__(self):
        self.cache = {}

    @property
    async def size(self, user: U) -> Result:
        return compute(self.cache)
'''.encode()

TS_SOURCE = '''import { Alpha as Beta } from "./alpha";
import fs from "fs";
import * as ns from "pkg";
// 中文注释：组件
/** Shape docs */
export interface Shape extends Base {
  area(): number;
}
export type ID = string | number;
/** Arrow docs */
export const scale = (value: number): number => helper(value);
export const MAX = 10;
export default class Widget extends Base implements Shape {
  constructor(private size: number) { super(); }
  @Input() area(): number { return compute(this.size); }
}
export function helper(value: string = "x"): void { log(value); }
enum Color { Red, Green }
const LOCAL_MAX = 3;
'''.encode()

RUST_SOURCE = '''//! module docs
use std::collections::HashMap;
use crate::foo::{Bar as Baz, self};

/// Max size.
pub const MAX: u32 = 10;

pub struct Point<T> {
    pub x: T,
    y: T,
}

pub enum Color { Red, Green }

pub trait Shape {
    fn area(&self) -> f64;
}

impl<T> Shape for Point<T> {
    /// Area doc.
    pub fn area(&self) -> f64 {
        helper(&self.x)
    }
    fn helper(&mut self) {}
}

#[tauri::command]
pub async fn cmd(name: String) -> Result<(), String> {
    helper(&name);
    Ok(())
}

mod inner {
    pub fn nested() {}
}

pub type Alias<T> = Vec<T>;
'''.encode()


def _by_qual(parsed):
    return {symbol.qualname: symbol for symbol in parsed.symbols}


def test_detect_language_by_suffix() -> None:
    assert detect_language(Path("a.py")) == "python"
    assert detect_language(Path("a.pyi")) == "python"
    assert detect_language(Path("a.ts")) == "typescript"
    assert detect_language(Path("a.mts")) == "typescript"
    assert detect_language(Path("a.tsx")) == "tsx"
    assert detect_language(Path("a.js")) == "javascript"
    assert detect_language(Path("a.jsx")) == "tsx"
    assert detect_language(Path("a.rs")) == "rust"
    assert detect_language(Path("a.PY")) == "python"
    assert detect_language(Path("a.txt")) is None
    assert detect_language(Path("Makefile")) is None


def test_python_symbols_docstrings_and_decorators() -> None:
    parsed = parse_source(PY_SOURCE, "python")
    symbols = _by_qual(parsed)

    assert symbols["outer"].kind == "function"
    assert symbols["outer"].docstring == "Outer doc."
    assert symbols["outer.inner"].kind == "function"
    assert symbols["outer.inner"].qualname == "outer.inner"
    assert symbols["Store"].kind == "class"
    assert symbols["Store"].docstring == "Store doc."
    assert symbols["Store.__init__"].kind == "method"
    assert symbols["Store.size"].kind == "method"
    assert symbols["Store.LIMIT"].kind == "constant"
    assert symbols["MAX_SIZE"].kind == "constant"

    # Decorated definitions start at the decorator, so the span can be replaced.
    outer = symbols["outer"]
    assert PY_SOURCE[outer.start_byte : outer.end_byte].startswith(b"@decorator")
    assert b"return inner()" in PY_SOURCE[outer.start_byte : outer.end_byte]
    assert PY_SOURCE.decode("utf-8").splitlines()[outer.start_line - 1].strip() == "@decorator"

    size = symbols["Store.size"]
    assert PY_SOURCE[size.start_byte : size.end_byte].startswith(b"@property")
    assert size.signature == "async def size(self, user: U) -> Result"

    # Offsets are byte offsets even when earlier lines are multi-byte.
    max_size = symbols["MAX_SIZE"]
    assert PY_SOURCE[max_size.start_byte : max_size.end_byte] == b"MAX_SIZE = 10"
    assert PY_SOURCE.decode("utf-8").splitlines()[max_size.start_line - 1] == "MAX_SIZE = 10"
    assert parsed.loc == len(PY_SOURCE.decode("utf-8").splitlines())


def test_python_imports_comments_and_refs() -> None:
    parsed = parse_source(PY_SOURCE, "python")

    assert parsed.imports == [
        ("os.path", ""),
        ("numpy", "np"),
        ("pkg.models", "U"),
        (".local", "helper"),
    ]
    assert any("中文注释" in text for _line, text in parsed.comments)
    assert any(text == "# module comment" for _line, text in parsed.comments)

    refs = {(ref.from_qualname, ref.to_name, ref.kind) for ref in parsed.refs}
    assert ("", "User", REF_IMPORT) in refs
    assert ("", "path", REF_IMPORT) in refs
    assert ("outer", "int", REF_TYPE) in refs
    assert ("outer", "Result", REF_TYPE) in refs
    assert ("outer.inner", "helper", REF_CALL) in refs
    assert ("outer", "inner", REF_CALL) in refs
    assert ("Store", "Base", REF_INHERIT) in refs
    assert ("Store.size", "compute", REF_CALL) in refs
    assert ("Store.size", "U", REF_TYPE) in refs


def test_typescript_symbols_and_refs() -> None:
    parsed = parse_source(TS_SOURCE, "typescript")
    symbols = _by_qual(parsed)

    assert symbols["Shape"].kind == "interface"
    assert symbols["Shape"].docstring == "Shape docs"
    assert symbols["Shape.area"].kind == "method"
    assert symbols["ID"].kind == "type"
    assert symbols["scale"].kind == "function"
    assert symbols["scale"].docstring == "Arrow docs"
    assert symbols["scale"].signature.startswith("export const scale = (value: number)")
    assert symbols["MAX"].kind == "constant"
    assert symbols["MAX"].signature == "export const MAX = 10"
    assert symbols["Widget"].kind == "class"
    assert symbols["Widget.constructor"].kind == "method"
    assert symbols["Widget.area"].kind == "method"
    assert symbols["helper"].kind == "function"
    assert symbols["helper"].signature.startswith("export function helper")
    assert symbols["Color"].kind == "enum"
    assert symbols["LOCAL_MAX"].kind == "constant"
    assert symbols["LOCAL_MAX"].docstring == ""
    assert symbols["LOCAL_MAX"].signature == "const LOCAL_MAX = 3"

    # A const-arrow symbol covers only the declarator, not the export keyword.
    scale = symbols["scale"]
    raw = TS_SOURCE[scale.start_byte : scale.end_byte]
    assert raw.startswith(b"scale = (value: number)")
    assert not raw.startswith(b"export")
    assert TS_SOURCE.decode("utf-8").splitlines()[scale.start_line - 1].startswith("export const scale")

    refs = {(ref.from_qualname, ref.to_name, ref.kind) for ref in parsed.refs}
    assert ("", "Alpha", REF_IMPORT) in refs
    assert ("", "fs", REF_IMPORT) in refs
    assert ("Shape", "Base", REF_INHERIT) in refs
    assert ("Widget", "Base", REF_INHERIT) in refs
    assert ("Widget", "Shape", REF_INHERIT) in refs
    assert ("Widget", "Input", REF_CALL) in refs
    assert ("Widget.area", "compute", REF_CALL) in refs
    assert ("helper", "log", REF_CALL) in refs
    assert ("scale", "helper", REF_CALL) in refs


def test_typescript_imports_and_comments() -> None:
    parsed = parse_source(TS_SOURCE, "typescript")
    assert parsed.imports == [
        ("./alpha", "Beta"),
        ("fs", "fs"),
        ("pkg", "ns"),
    ]
    assert any(text == "// 中文注释：组件" for _line, text in parsed.comments)
    assert any(text.startswith("/** Shape docs */") for _line, text in parsed.comments)


def test_typescript_anonymous_default_export() -> None:
    parsed = parse_source(b"export default () => 42;\n", "typescript")
    symbols = _by_qual(parsed)
    assert symbols["default"].kind == "function"
    assert symbols["default"].signature.startswith("export default")


def test_typescript_export_from() -> None:
    parsed = parse_source(b'export { Gamma } from "./gamma";\n', "typescript")
    assert parsed.imports == [("./gamma", "Gamma")]
    assert any(ref.to_name == "Gamma" and ref.kind == REF_IMPORT for ref in parsed.refs)


def test_typescript_anonymous_functions_scope_local_constants() -> None:
    parsed = parse_source(
        b"const top = 1;\n"
        b"it(() => { const in_arrow = 2; });\n"
        b"const fn = function () { const in_function = 3; };\n",
        "typescript",
    )

    qualnames = {symbol.qualname for symbol in parsed.symbols}
    assert qualnames == {"top", "fn"}
    assert not any(symbol.name.startswith("in_") for symbol in parsed.symbols)


def test_typescript_object_arrow_members_become_methods() -> None:
    source = (
        b"export const useStore = create<TavernState>((set) => ({\n"
        b"  sendMessage: async (text: string): Promise<void> => { set(text); },\n"
        b"  ui: {\n"
        b"    toggle: () => {},\n"
        b"  },\n"
        b"  plain: 1,\n"
        b"}));\n"
    )
    parsed = parse_source(source, "typescript")
    symbols = _by_qual(parsed)

    assert symbols["useStore.sendMessage"].kind == "method"
    assert symbols["useStore.ui.toggle"].kind == "method"
    assert "useStore.plain" not in symbols

    # The symbol span covers the whole pair so it can be replaced in place.
    member = symbols["useStore.sendMessage"]
    raw = source[member.start_byte : member.end_byte]
    assert raw.startswith(b"sendMessage:")
    assert raw.endswith(b"}")

    refs = {(ref.from_qualname, ref.to_name, ref.kind) for ref in parsed.refs}
    assert ("useStore.sendMessage", "set", REF_CALL) in refs


def test_typescript_object_arrow_member_without_outer_symbol() -> None:
    parsed = parse_source(b"create((set) => ({ foo: () => {} }));\n", "typescript")

    assert [symbol.qualname for symbol in parsed.symbols] == ["foo"]
    assert parsed.symbols[0].kind == "method"
    assert parsed.symbols[0].signature == "foo: ()"


def test_rust_symbols_docstrings_and_attributes() -> None:
    parsed = parse_source(RUST_SOURCE, "rust")
    symbols = _by_qual(parsed)

    assert symbols["MAX"].kind == "constant"
    assert symbols["MAX"].docstring == "Max size."
    assert symbols["Point"].kind == "struct"
    assert symbols["Color"].kind == "enum"
    assert symbols["Shape"].kind == "trait"
    assert symbols["Shape::area"].kind == "method"
    assert symbols["Point::area"].kind == "method"
    assert symbols["Point::area"].docstring == "Area doc."
    assert symbols["Point::helper"].kind == "method"
    assert symbols["cmd"].kind == "function"
    assert symbols["inner"].kind == "module"
    assert symbols["inner::nested"].kind == "function"
    assert symbols["Alias"].kind == "type"

    assert symbols["Point"].signature == "pub struct Point<T>"
    assert symbols["Point::area"].signature == "pub fn area(&self) -> f64"

    # Attributes are part of the item span and precede the signature.
    cmd = symbols["cmd"]
    raw = RUST_SOURCE[cmd.start_byte : cmd.end_byte]
    assert raw.startswith(b"#[tauri::command]")
    assert cmd.signature == "pub async fn cmd(name: String) -> Result<(), String>"

    lines = RUST_SOURCE.decode("utf-8").splitlines()
    assert lines[cmd.start_line - 1] == "#[tauri::command]"
    assert lines[symbols["MAX"].start_line - 1].strip() == "pub const MAX: u32 = 10;"


def test_rust_imports_and_refs() -> None:
    parsed = parse_source(RUST_SOURCE, "rust")
    assert parsed.imports == [
        ("std::collections", "HashMap"),
        ("crate::foo", "Baz"),
    ]

    refs = {(ref.from_qualname, ref.to_name, ref.kind) for ref in parsed.refs}
    assert ("", "HashMap", REF_IMPORT) in refs
    assert ("", "Bar", REF_IMPORT) in refs
    assert ("Point::area", "helper", REF_CALL) in refs
    assert ("cmd", "helper", REF_CALL) in refs
    assert ("cmd", "String", REF_TYPE) in refs
    assert ("cmd", "Result", REF_TYPE) in refs
    assert ("Point", "Shape", REF_INHERIT) in refs


def test_rust_use_list_expansion() -> None:
    parsed = parse_source(
        b"use std::io::{self, Read};\nuse a::{b::c, d as e};\nuse x::*;\n",
        "rust",
    )
    assert parsed.imports == [
        ("std::io", "Read"),
        ("a::b", "c"),
        ("a", "e"),
        ("x", "*"),
    ]
    assert {ref.to_name for ref in parsed.refs} == {"Read", "c", "d"}


def test_parse_file_reads_raw_bytes(tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    path.write_bytes("# 中文\n\ndef greet(name):\n    return name\n".encode("utf-8"))
    parsed = parse_file(path, "sample.py", "python")
    assert [symbol.name for symbol in parsed.symbols] == ["greet"]
    symbol = parsed.symbols[0]
    assert path.read_bytes()[symbol.start_byte : symbol.end_byte].startswith(b"def greet")
    assert symbol.start_line == 3


def test_tsx_and_javascript_share_the_front_end() -> None:
    tsx = parse_source(b"export const App = () => <div>hi</div>;\n", "tsx")
    assert [symbol.qualname for symbol in tsx.symbols] == ["App"]

    js = parse_source(b"function f(a) { return g(a); }\nconst h = () => 1;\n", "javascript")
    symbols = _by_qual(js)
    assert symbols["f"].kind == "function"
    assert symbols["h"].kind == "function"
    assert any(ref.to_name == "g" and ref.kind == REF_CALL for ref in js.refs)


def test_large_python_file_parses() -> None:
    # Regression guard for py-tree-sitter 0.26: reading Point.row/column on
    # Windows crashes the interpreter once enough nodes have been visited.
    lines: list[str] = []
    for index in range(300):
        lines.append(f"def fn_{index}(value: int) -> Result:")
        lines.append(f"    # comment {index}")
        lines.append(f"    return helper_{index}(value)")
        lines.append("")
    source = "\n".join(lines).encode()
    parsed = parse_source(source, "python")
    assert len(parsed.symbols) == 300
    assert parsed.symbols[0].start_line == 1
    assert parsed.symbols[-1].qualname == "fn_299"


def test_malformed_sources_do_not_raise() -> None:
    for source, lang in (
        (b"def broken(:\n", "python"),
        (b"export function ( {", "typescript"),
        (b"fn broken( {", "rust"),
    ):
        parsed = parse_source(source, lang)
        assert parsed.loc >= 0


def test_unknown_language_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        parse_source(b"x", "cobol")
