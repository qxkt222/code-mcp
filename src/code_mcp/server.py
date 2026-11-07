"""MCP server: exposes the code-context surface to an AI coding agent.

The tool descriptions are the routing layer for the model, so each one states
when it should be preferred over reading files directly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from . import tokens as token_util
from .config import load_config
from .db import open_index
from .edit import replace_symbol
from .graph.impact import impact as compute_impact
from .graph.refs import find_symbol, resolve_refs
from .graph.tauri import build_bridges
from .index.builder import build_index
from .index.modules import resolve_imports
from .index.search import search_comments, search_symbols
from .pack import ensure_fresh
from .pack import pack as build_pack
from .paths import resolve_within
from .review.external import detect_tools, run_external
from .review.gate import run_gate
from .review.structural import analyze as analyze_structure
from .serialize import pack_dict, with_response_tokens
from .types import Symbol

INSTRUCTIONS = (
    "Local code-context server. Use it before reading whole files: it returns only the "
    "symbol you asked for plus the code causally connected to it, and reports how many "
    "tokens that saved versus reading the file. Run `index` once per repository first."
)

mcp = MCPServer("code-mcp", instructions=INSTRUCTIONS)


# --------------------------------------------------------------------------- helpers


def _root(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"not a directory: {resolved}")
    return resolved


def _conn(root: Path) -> sqlite3.Connection:
    return open_index(root)


def _symbol_dict(symbol: Symbol) -> dict:
    return {
        "file": symbol.file,
        "name": symbol.name,
        "qualname": symbol.qualname,
        "kind": symbol.kind,
        "start_line": symbol.start_line,
        "end_line": symbol.end_line,
        "signature": symbol.signature,
        "docstring": symbol.docstring[:200],
    }


# ----------------------------------------------------------------------------- tools


def index(path: str, force: bool = False) -> dict:
    """Build or incrementally refresh the code index for a repository.

    Call this once before any other tool, and again after large external changes.
    Incremental by default: only files whose mtime/size changed are re-parsed.

    Args:
        path: Repository root directory.
        force: Re-parse every file even when unchanged.
    """
    root = _root(path)
    config = load_config(root)
    conn = _conn(root)
    try:
        stats = build_index(root, conn, config, force=force)
        resolve_imports(conn, root)
        resolve_refs(conn)
        build_bridges(conn)
        conn.commit()
        totals = conn.execute(
            "SELECT (SELECT COUNT(*) FROM files) AS files, "
            "(SELECT COUNT(*) FROM symbols) AS symbols, "
            "(SELECT COUNT(*) FROM refs) AS refs"
        ).fetchone()
        return with_response_tokens(
            {
                "indexed": stats.files_indexed,
                "skipped": stats.files_skipped,
                "removed": stats.files_removed,
                "symbols": totals["symbols"],
                "refs": totals["refs"],
                "files_total": totals["files"],
                "duration_s": round(stats.duration_s, 2),
            }
        )
    finally:
        conn.close()


def find(path: str, query: str, kinds: list[str] | None = None, limit: int = 20) -> dict:
    """Locate candidate symbols by name, signature or comment text.

    Use this instead of grep when you know what you are looking for but not
    where it lives. Returns summaries only, never full bodies — follow up with
    `pack` or `slice` for the code itself.

    Args:
        path: Repository root directory.
        query: Symbol name, partial name, or words from a comment/docstring.
        kinds: Optional filter, e.g. ["function", "class"].
        limit: Maximum results, default 20.
    """
    root = _root(path)
    conn = _conn(root)
    try:
        symbols = search_symbols(conn, query, kinds=kinds, limit=limit)
        comments = search_comments(conn, query, limit=max(1, limit // 2))
        return with_response_tokens(
            {
                "symbols": [_symbol_dict(symbol) for symbol in symbols],
                "comments": [
                    {"file": file, "line": line, "text": text}
                    for file, line, text in comments
                ],
            }
        )
    finally:
        conn.close()


def pack(
    path: str,
    file: str,
    name: str | None = None,
    budget_tokens: int = 30000,
    include_tests: bool = True,
    segment: int | None = None,
    max_target_tokens: int = 4000,
    force: bool = False,
    target_only: bool = False,
) -> dict:
    """Return the target symbol plus its causal neighbourhood, trimmed to a budget.

    This is the main replacement for reading a whole file. It returns the target
    body, its callers, callees, imports and matching tests, and reports the token
    cost versus reading the containing file. If `worth_using` is false, read the
    file directly — the packed result would not save anything.

    A target larger than `max_target_tokens` is reduced to its signature plus an
    outline: `target.segments` lists the addressable blocks and `target.truncated`
    is true. Fetch one block with `segment=N` (1-based) or with
    `slice_symbol(start_line=..., end_line=...)`. A `segment` that is out of
    range returns `worth_using=false` and an explanatory note.

    Args:
        path: Repository root directory.
        file: Repository-relative file path.
        name: Symbol name. Omit for the first top-level symbol in the file.
        budget_tokens: Upper bound on returned tokens, default 30000.
        include_tests: Include tests that reference the symbol.
        segment: 1-based index of one outline block to fetch instead of the pack.
        max_target_tokens: Target size above which an outline is returned,
            default 4000.
        force: return the packed context even when it costs more than reading the
            file
        target_only: return just the target (or its outline); omit callers,
            callees, imports and tests
    """
    root = _root(path)
    conn = _conn(root)
    try:
        result = build_pack(
            conn,
            root,
            file,
            name,
            budget_tokens=budget_tokens,
            include_tests=include_tests,
            max_target_tokens=max_target_tokens,
            segment=segment,
            force=force,
            target_only=target_only,
        )
        return with_response_tokens(pack_dict(result))
    finally:
        conn.close()


def slice_symbol(
    path: str,
    file: str,
    name: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict:
    """Read exactly one symbol, or an explicit line range of one file.

    Prefer this over reading the whole file when you already know the symbol
    name. For change preparation use `pack` instead, which also pulls in the
    surrounding dependency context.

    Args:
        path: Repository root directory.
        file: Repository-relative file path.
        name: Symbol name to read.
        start_line: Alternative to `name`; 1-based inclusive start.
        end_line: 1-based inclusive end, only with `start_line`.
    """
    root = _root(path)
    conn = _conn(root)
    try:
        source_path = resolve_within(root, file)
        if source_path is None:
            raise ValueError("file is outside the repository root")
        if not source_path.is_file():
            raise ValueError(f"not a file: {file}")

        lines = source_path.read_bytes().decode("utf-8", errors="replace").splitlines()

        if name:
            if not ensure_fresh(conn, root, file):
                raise ValueError(
                    "file changed on disk and could not be re-indexed; run index"
                )
            symbol = find_symbol(conn, file, name)
            if symbol is None:
                raise ValueError(f"symbol not found: {name} in {file}")
            first, last, label = symbol.start_line, symbol.end_line, symbol.qualname
        elif start_line is not None:
            first = max(1, start_line)
            last = min(len(lines), end_line or start_line)
            label = ""
        else:
            raise ValueError("provide either name or start_line")

        text = "\n".join(lines[first - 1 : last])
        return with_response_tokens(
            {
                "file": file,
                "symbol": label,
                "start_line": first,
                "end_line": last,
                "tokens": token_util.estimate(text),
                "text": text,
            }
        )
    finally:
        conn.close()


def edit(
    path: str,
    file: str,
    name: str,
    new_code: str,
    expect_hash: str | None = None,
) -> dict:
    """Replace one symbol's implementation without rewriting the whole file.

    The write is refused when the file changed since indexing, when the result
    fails to parse, or when `expect_hash` does not match. Line endings are
    normalised to the file's existing style.

    Args:
        path: Repository root directory.
        file: Repository-relative file path.
        name: Symbol to replace.
        new_code: Full replacement source for that symbol.
        expect_hash: Optional sha256 of the file you based the edit on.
    """
    root = _root(path)
    conn = _conn(root)
    try:
        result = replace_symbol(root, file, name, new_code, expect_hash=expect_hash)
        conn.commit()
        return with_response_tokens(
            {
                "file": result.file,
                "symbol": result.symbol,
                "applied": result.applied,
                "reason": result.reason,
                "lines_added": result.lines_added,
                "lines_removed": result.lines_removed,
                "new_hash": result.new_hash,
                "diff": result.diff,
            }
        )
    finally:
        conn.close()


def impact(
    path: str,
    file: str,
    name: str,
    depth: int = 2,
    limit: int = 100,
    include_tests: bool = False,
    include_imports: bool = False,
) -> dict:
    """List what breaks if this symbol changes, ranked by distance.

    Use before changing a signature, renaming, or deleting a symbol. Test
    callers and import pseudo-symbols are hidden by default because they drown
    out the actionable call sites; `filtered` reports how many were hidden.

    Args:
        path: Repository root directory.
        file: Repository-relative file path.
        name: Symbol name.
        depth: How many hops to follow, default 2.
        limit: Maximum affected symbols, default 100.
        include_tests: Also list callers that live in test files.
        include_imports: Also list import pseudo-symbols (empty qualnames).
    """
    root = _root(path)
    conn = _conn(root)
    try:
        symbol = find_symbol(conn, file, name)
        if symbol is None:
            raise ValueError(f"symbol not found: {name} in {file}")
        nodes = compute_impact(
            conn,
            symbol.id,
            depth=depth,
            limit=limit,
            include_tests=include_tests,
            include_imports=include_imports,
        )
        return with_response_tokens(
            {
                "target": _symbol_dict(symbol),
                "affected": [
                    {
                        **_symbol_dict(node.symbol),
                        "depth": node.depth,
                        "via": node.via,
                        "line": node.line,
                    }
                    for node in nodes
                ],
                "count": len(nodes),
                "filtered": nodes.filtered,
            }
        )
    finally:
        conn.close()


def review(path: str, min_severity: str = "P2", paths: list[str] | None = None) -> dict:
    """Run health and sustainability checks over the indexed code.

    Combines external linters (ruff/mypy/bandit/tsc/cargo, whichever exist) with
    structural rules: long functions, deep nesting, fan-in/fan-out, duplicated
    bodies, missing docstrings and symbols without tests.

    Args:
        path: Repository root directory.
        min_severity: Lowest severity to report; P0 is the most severe.
        paths: Optional repository-relative files to restrict the structural pass.
    """
    root = _root(path)
    conn = _conn(root)
    try:
        order = ["P0", "P1", "P2", "P3"]
        cutoff = order.index(min_severity) if min_severity in order else len(order)

        findings = analyze_structure(conn, files=paths)

        tools = detect_tools(root)
        external_findings, ran, missing = run_external(root, tools, paths=paths)
        findings.extend(external_findings)

        kept = [item for item in findings if order.index(item.severity) <= cutoff]
        kept.sort(key=lambda item: (order.index(item.severity), item.file, item.line))

        counts = dict.fromkeys(order, 0)
        for item in kept:
            counts[item.severity] += 1

        return with_response_tokens(
            {
                "counts": counts,
                "findings": [
                    {
                        "rule": item.rule,
                        "severity": item.severity,
                        "file": item.file,
                        "line": item.line,
                        "symbol": item.symbol,
                        "message": item.message,
                    }
                    for item in kept
                ],
                "external_ran": ran,
                "external_missing": missing,
            }
        )
    finally:
        conn.close()


def gate(path: str, only: list[str] | None = None, timeout_s: int = 600) -> dict:
    """Run the project's gate commands (tests, lint, type check) and report each step.

    Commands come from `.code-mcp.toml`, or are auto-detected from marker files.
    A failing step does not stop the remaining steps.

    Args:
        path: Repository root directory.
        only: Optional step names to run.
        timeout_s: Per-step timeout in seconds, default 600.
    """
    root = _root(path)
    config = load_config(root)
    if timeout_s:
        for step in config.gate_steps:
            step.timeout = min(step.timeout, timeout_s)

    result = run_gate(config, only=only)
    return with_response_tokens(
        {
            "passed": result.passed,
            "detected_config": config.detected,
            "steps": [
                {
                    "name": step.name,
                    "command": step.command,
                    "passed": step.passed,
                    "exit_code": step.exit_code,
                    "duration_s": round(step.duration_s, 2),
                    "output": step.output,
                }
                for step in result.steps
            ],
        }
    )


def _register() -> None:
    mcp.tool()(index)
    mcp.tool()(find)
    mcp.tool()(pack)
    mcp.tool()(slice_symbol)
    mcp.tool()(edit)
    mcp.tool()(impact)
    mcp.tool()(review)
    mcp.tool()(gate)


_register()


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
