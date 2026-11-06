"""Behavioural tests for import-specifier resolution."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from code_mcp import db
from code_mcp.index.modules import ModuleResolver, resolve_imports


def _db(tmp_path: Path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "index.db")
    db.init_schema(conn)
    return conn


def _file(conn: sqlite3.Connection, path: str, lang: str) -> int:
    cursor = conn.execute(
        "INSERT INTO files(path, lang, mtime, hash, loc) VALUES (?, ?, 0, '', 0)",
        (path, lang),
    )
    return int(cursor.lastrowid)


def _import(
    conn: sqlite3.Connection,
    file_id: int,
    module: str,
    alias: str = "",
    resolved_file_id: int | None = None,
) -> int:
    cursor = conn.execute(
        """INSERT INTO imports(file_id, module, alias, resolved_file_id)
           VALUES (?, ?, ?, ?)""",
        (file_id, module, alias, resolved_file_id),
    )
    return int(cursor.lastrowid)


def _resolved(conn: sqlite3.Connection, import_id: int) -> int | None:
    row = conn.execute(
        "SELECT resolved_file_id FROM imports WHERE id = ?", (import_id,)
    ).fetchone()
    return row["resolved_file_id"]


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    return root


def _tsconfig(root: Path, payload: dict, name: str = "tsconfig.json") -> None:
    (root / name).write_text(json.dumps(payload), encoding="utf-8")


def test_python_absolute_imports(tmp_path):
    conn = _db(tmp_path)
    root = _root(tmp_path)
    a = _file(conn, "pkg/a.py", "python")
    b = _file(conn, "pkg/b.py", "python")
    init = _file(conn, "pkg/__init__.py", "python")

    to_module = _import(conn, a, "pkg.b")
    to_package = _import(conn, a, "pkg")
    missing = _import(conn, a, "pkg.nope")
    conn.commit()

    assert resolve_imports(conn, root) == 2
    assert _resolved(conn, to_module) == b
    assert _resolved(conn, to_package) == init
    assert _resolved(conn, missing) is None


def test_python_relative_imports(tmp_path):
    conn = _db(tmp_path)
    root = _root(tmp_path)
    c = _file(conn, "pkg/sub/c.py", "python")
    sibling = _file(conn, "pkg/sub/d.py", "python")
    parent = _file(conn, "pkg/e.py", "python")
    sub_init = _file(conn, "pkg/sub/__init__.py", "python")
    pkg_init = _file(conn, "pkg/__init__.py", "python")

    same_package = _import(conn, c, ".d")
    up_one = _import(conn, c, "..e")
    package_itself = _import(conn, c, ".")
    parent_package = _import(conn, c, "..")
    conn.commit()

    assert resolve_imports(conn, root) == 4
    assert _resolved(conn, same_package) == sibling
    assert _resolved(conn, up_one) == parent
    assert _resolved(conn, package_itself) == sub_init
    assert _resolved(conn, parent_package) == pkg_init


def test_python_ambiguous_module_stays_unresolved(tmp_path):
    conn = _db(tmp_path)
    root = _root(tmp_path)
    a = _file(conn, "pkg/a.py", "python")
    _file(conn, "pkg/b.py", "python")
    _file(conn, "pkg/b/__init__.py", "python")
    ambiguous = _import(conn, a, "pkg.b")
    conn.commit()

    assert resolve_imports(conn, root) == 0
    assert _resolved(conn, ambiguous) is None


def test_web_relative_imports(tmp_path):
    conn = _db(tmp_path)
    root = _root(tmp_path)
    caller = _file(conn, "src/deep/a.ts", "typescript")
    same_dir = _file(conn, "src/deep/b.ts", "typescript")
    sibling_dir = _file(conn, "src/lib/c.jsx", "javascript")
    index_file = _file(conn, "src/deep/widgets/index.ts", "typescript")

    to_same = _import(conn, caller, "./b")
    to_explicit = _import(conn, caller, "./b.ts")
    to_parent = _import(conn, caller, "../lib/c")
    to_index = _import(conn, caller, "./widgets")
    to_missing = _import(conn, caller, "./nope")
    conn.commit()

    assert resolve_imports(conn, root) == 4
    assert _resolved(conn, to_same) == same_dir
    assert _resolved(conn, to_explicit) == same_dir
    assert _resolved(conn, to_parent) == sibling_dir
    assert _resolved(conn, to_index) == index_file
    assert _resolved(conn, to_missing) is None


def test_web_tsconfig_path_alias(tmp_path):
    conn = _db(tmp_path)
    root = _root(tmp_path)
    _tsconfig(
        root,
        {
            "compilerOptions": {
                "baseUrl": ".",
                "paths": {"@/*": ["./src/*"]},
            }
        },
    )
    caller = _file(conn, "src/app/main.ts", "typescript")
    button = _file(conn, "src/components/Button.tsx", "tsx")
    client = _file(conn, "src/api/client.ts", "typescript")

    to_button = _import(conn, caller, "@/components/Button")
    to_client = _import(conn, caller, "@/api/client")
    to_missing = _import(conn, caller, "@/api/missing")
    conn.commit()

    assert resolve_imports(conn, root) == 2
    assert _resolved(conn, to_button) == button
    assert _resolved(conn, to_client) == client
    assert _resolved(conn, to_missing) is None


def test_web_base_url_fallback(tmp_path):
    conn = _db(tmp_path)
    root = _root(tmp_path)
    _tsconfig(root, {"compilerOptions": {"baseUrl": "src"}})
    caller = _file(conn, "src/main.ts", "typescript")
    button = _file(conn, "src/components/Button.tsx", "tsx")

    bare = _import(conn, caller, "components/Button")
    conn.commit()

    assert resolve_imports(conn, root) == 1
    assert _resolved(conn, bare) == button


def test_web_jsconfig_alias_is_read(tmp_path):
    conn = _db(tmp_path)
    root = _root(tmp_path)
    _tsconfig(
        root,
        {
            "compilerOptions": {
                "baseUrl": ".",
                "paths": {"~/*": ["app/*"]},
            }
        },
        name="jsconfig.json",
    )
    caller = _file(conn, "src/main.js", "javascript")
    helper = _file(conn, "app/helper.js", "javascript")

    aliased = _import(conn, caller, "~/helper")
    conn.commit()

    assert resolve_imports(conn, root) == 1
    assert _resolved(conn, aliased) == helper


def test_web_tsconfig_with_comments_and_trailing_commas(tmp_path):
    conn = _db(tmp_path)
    root = _root(tmp_path)
    (root / "tsconfig.json").write_text(
        """
        {
          // path aliases
          "compilerOptions": {
            "baseUrl": ".",
            "paths": { "@/*": ["./src/*"], },
          },
        }
        """,
        encoding="utf-8",
    )
    caller = _file(conn, "src/a.ts", "typescript")
    target = _file(conn, "src/b.ts", "typescript")
    aliased = _import(conn, caller, "@/b")
    conn.commit()

    assert resolve_imports(conn, root) == 1
    assert _resolved(conn, aliased) == target


def test_web_ambiguous_candidates_stay_unresolved(tmp_path):
    conn = _db(tmp_path)
    root = _root(tmp_path)
    caller = _file(conn, "src/a.ts", "typescript")
    _file(conn, "src/x.ts", "typescript")
    _file(conn, "src/x/index.ts", "typescript")
    ambiguous = _import(conn, caller, "./x")
    conn.commit()

    assert resolve_imports(conn, root) == 0
    assert _resolved(conn, ambiguous) is None


def test_rust_crate_paths(tmp_path):
    conn = _db(tmp_path)
    root = _root(tmp_path)
    main = _file(conn, "src/main.rs", "rust")
    module = _file(conn, "src/foo/bar.rs", "rust")
    dir_module = _file(conn, "src/baz/mod.rs", "rust")

    nested = _import(conn, main, "crate::foo::bar")
    directory_mod = _import(conn, main, "crate::baz")
    external = _import(conn, main, "std::collections")
    conn.commit()

    assert resolve_imports(conn, root) == 2
    assert _resolved(conn, nested) == module
    assert _resolved(conn, directory_mod) == dir_module
    assert _resolved(conn, external) is None


def test_rust_super_paths_are_relative_to_the_module_directory(tmp_path):
    conn = _db(tmp_path)
    root = _root(tmp_path)
    deep = _file(conn, "src/deep/mod.rs", "rust")
    parent = _file(conn, "src/parent.rs", "rust")

    up = _import(conn, deep, "super::parent")
    conn.commit()

    assert resolve_imports(conn, root) == 1
    assert _resolved(conn, up) == parent


def test_resolve_imports_is_idempotent_and_corrects_stale_targets(tmp_path):
    conn = _db(tmp_path)
    root = _root(tmp_path)
    caller = _file(conn, "src/a.ts", "typescript")
    target = _file(conn, "src/b.ts", "typescript")
    other = _file(conn, "src/c.ts", "typescript")
    row = _import(conn, caller, "./b")
    conn.commit()

    assert resolve_imports(conn, root) == 1
    assert _resolved(conn, row) == target
    assert resolve_imports(conn, root) == 0

    conn.execute(
        "UPDATE imports SET resolved_file_id = ? WHERE id = ?", (other, row)
    )
    conn.commit()
    assert resolve_imports(conn, root) == 1
    assert _resolved(conn, row) == target


def test_unknown_language_and_empty_module_are_ignored(tmp_path):
    conn = _db(tmp_path)
    root = _root(tmp_path)
    odd = _file(conn, "notes.xyz", "unknown")
    first = _import(conn, odd, "./b")
    _import(conn, odd, "")
    conn.commit()

    assert resolve_imports(conn, root) == 0
    assert _resolved(conn, first) is None


def test_module_resolver_accepts_none_from_path(tmp_path):
    resolver = ModuleResolver({}, tmp_path)
    assert resolver.resolve(None, "./x") is None
    assert resolver.resolve("missing.ts", "./x") is None
