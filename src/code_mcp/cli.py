"""Command line entry point mirroring the MCP tools for manual use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .db import open_index
from .graph.impact import impact as compute_impact
from .graph.refs import find_symbol, resolve_refs
from .graph.tauri import build_bridges
from .index.builder import build_index
from .index.modules import resolve_imports
from .index.search import search_comments, search_symbols
from .config import load_config
from .pack import ensure_fresh, pack as build_pack
from .paths import resolve_within
from .review.external import detect_tools, run_external
from .review.gate import run_gate
from .review.structural import analyze as analyze_structure
from .serialize import pack_dict
from .edit import replace_symbol
from . import tokens as token_util


def _open(path: str):
    root = Path(path).expanduser().resolve()
    return root, open_index(root)


def _print_json(payload: object) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


# ----------------------------------------------------------------- commands


def cmd_index(args: argparse.Namespace) -> int:
    root, conn = _open(args.path)
    try:
        stats = build_index(root, conn, load_config(root), force=args.force)
        resolve_imports(conn, root)
        resolve_refs(conn)
        build_bridges(conn)
        conn.commit()
        _print_json(
            {
                "indexed": stats.files_indexed,
                "skipped": stats.files_skipped,
                "removed": stats.files_removed,
                "symbols": stats.symbols,
                "refs": stats.refs,
                "duration_s": round(stats.duration_s, 2),
            }
        )
        return 0
    finally:
        conn.close()


def cmd_find(args: argparse.Namespace) -> int:
    _, conn = _open(args.path)
    try:
        symbols = search_symbols(conn, args.query, kinds=args.kind, limit=args.limit)
        comments = search_comments(conn, args.query, limit=max(1, args.limit // 2))
        if args.json:
            _print_json(
                {
                    "symbols": [
                        {
                            "file": item.file,
                            "qualname": item.qualname,
                            "kind": item.kind,
                            "line": item.start_line,
                            "signature": item.signature,
                        }
                        for item in symbols
                    ],
                    "comments": [
                        {"file": file, "line": line, "text": text}
                        for file, line, text in comments
                    ],
                }
            )
            return 0
        for item in symbols:
            print(f"{item.file}:{item.start_line}  {item.kind:<10} {item.qualname}")
            if item.signature:
                print(f"    {item.signature}")
        for file, line, text in comments:
            print(f"{file}:{line}  comment  {text[:100]}")
        return 0
    finally:
        conn.close()


def _print_slice(label: str, item) -> None:
    """Print one packed slice; an oversized target is shown as its outline."""
    print(f"\n=== {label} {item.header()} ===")
    if item.truncated:
        for segment in item.segments:
            print(
                f"  #{segment.index}  L{segment.start_line}-{segment.end_line}  "
                f"{segment.label}"
            )
        return
    print(item.text)


def cmd_pack(args: argparse.Namespace) -> int:
    root, conn = _open(args.path)
    try:
        result = build_pack(
            conn,
            root,
            args.file,
            args.name,
            budget_tokens=args.budget,
            max_target_tokens=args.max_target_tokens,
            segment=args.segment,
            force=args.force,
        )
        if args.json:
            _print_json(pack_dict(result))
            return 0
        for label, items in result.sections().items():
            for item in items:
                _print_slice(label, item)
        print(
            f"\n[pack] {result.token_total} tokens vs {result.token_baseline} "
            f"for a plain read ({result.savings_pct}% saved)"
        )
        if result.note:
            print(f"[note] {result.note}")
        return 0
    finally:
        conn.close()


def cmd_slice(args: argparse.Namespace) -> int:
    root, conn = _open(args.path)
    try:
        source = resolve_within(root, args.file)
        if source is None:
            print("file is outside the repository root", file=sys.stderr)
            return 1
        if not source.is_file():
            print(f"not a file: {args.file}", file=sys.stderr)
            return 1
        lines = source.read_bytes().decode("utf-8", errors="replace").splitlines()
        if args.name:
            if not ensure_fresh(conn, root, args.file):
                print(
                    "file changed on disk and could not be re-indexed; run index",
                    file=sys.stderr,
                )
                return 1
            symbol = find_symbol(conn, args.file, args.name)
            if symbol is None:
                print(f"symbol not found: {args.name}", file=sys.stderr)
                return 1
            first, last = symbol.start_line, symbol.end_line
        else:
            first, last = args.start, args.end or args.start
        text = "\n".join(lines[first - 1 : last])
        tokens = token_util.estimate(text)
        if args.json:
            _print_json(
                {
                    "file": args.file,
                    "start_line": first,
                    "end_line": last,
                    "tokens": tokens,
                    "text": text,
                    "segments": [],
                    "truncated": False,
                    "total_tokens": tokens,
                }
            )
        else:
            print(text)
        return 0
    finally:
        conn.close()


def cmd_edit(args: argparse.Namespace) -> int:
    root, conn = _open(args.path)
    try:
        new_code = Path(args.code_file).read_text(encoding="utf-8")
        result = replace_symbol(root, args.file, args.name, new_code)
        conn.commit()
        if args.json:
            _print_json(
                {
                    "applied": result.applied,
                    "reason": result.reason,
                    "lines_added": result.lines_added,
                    "lines_removed": result.lines_removed,
                    "diff": result.diff,
                }
            )
        else:
            status = "applied" if result.applied else f"refused: {result.reason}"
            print(f"{status}  +{result.lines_added} -{result.lines_removed}")
            if result.diff:
                print(result.diff)
        return 0 if result.applied else 1
    finally:
        conn.close()


def cmd_impact(args: argparse.Namespace) -> int:
    _, conn = _open(args.path)
    try:
        symbol = find_symbol(conn, args.file, args.name)
        if symbol is None:
            print(f"symbol not found: {args.name}", file=sys.stderr)
            return 1
        nodes = compute_impact(
            conn,
            symbol.id,
            depth=args.depth,
            include_tests=args.include_tests,
            include_imports=args.include_imports,
        )
        if args.json:
            _print_json(
                {
                    "target": symbol.qualname,
                    "affected": [
                        {
                            "file": node.symbol.file,
                            "qualname": node.symbol.qualname,
                            "depth": node.depth,
                            "via": node.via,
                            "line": node.line,
                        }
                        for node in nodes
                    ],
                    "filtered": nodes.filtered,
                }
            )
            return 0
        print(f"target: {symbol.file}:{symbol.start_line} {symbol.qualname}")
        for node in nodes:
            print(
                f"  d{node.depth} {node.via:<10} {node.symbol.file}:{node.symbol.start_line} "
                f"{node.symbol.qualname}"
            )
        print(f"{len(nodes)} affected")
        return 0
    finally:
        conn.close()


def cmd_review(args: argparse.Namespace) -> int:
    root, conn = _open(args.path)
    try:
        order = ["P0", "P1", "P2", "P3"]
        cutoff = order.index(args.min_severity) if args.min_severity in order else len(order)
        findings = analyze_structure(conn, files=args.files)
        tools = detect_tools(root)
        external, ran, missing = run_external(root, tools, paths=args.files)
        findings.extend(external)
        kept = [item for item in findings if order.index(item.severity) <= cutoff]
        kept.sort(key=lambda item: (order.index(item.severity), item.file, item.line))

        if args.json:
            _print_json(
                {
                    "findings": [
                        {
                            "rule": item.rule,
                            "severity": item.severity,
                            "file": item.file,
                            "line": item.line,
                            "message": item.message,
                        }
                        for item in kept
                    ],
                    "external_ran": ran,
                    "external_missing": missing,
                }
            )
            return 0
        for item in kept:
            print(f"{item.severity} {item.rule:<18} {item.file}:{item.line}  {item.message}")
        print(f"{len(kept)} findings  (ran: {', '.join(ran) or 'none'})")
        return 0
    finally:
        conn.close()


def _gate_payload(result, detected: bool) -> dict:
    """JSON-ready gate result, shared with ``server.gate``.

    ``serialize.gate_dict`` is the single source of truth when present; the
    inline branch keeps the CLI working on its own.
    """
    try:
        from .serialize import gate_dict
    except ImportError:  # pragma: no cover - an older serialize without it
        gate_dict = None
    if gate_dict is not None:
        try:
            return gate_dict(result, detected_config=detected)
        except TypeError:  # older signature without the optional argument
            return gate_dict(result)
    return {  # pragma: no cover - replaced by serialize.gate_dict
        "passed": result.passed,
        "detected_config": detected,
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


def cmd_gate(args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser().resolve()
    config = load_config(root)
    only = args.only.split(",") if args.only else None
    result = run_gate(config, only=only)
    if args.json:
        _print_json(_gate_payload(result, config.detected))
        return 0 if result.passed else 1
    for step in result.steps:
        mark = "PASS" if step.passed else "FAIL"
        print(f"[{mark}] {step.name}: {step.command}  ({step.duration_s:.1f}s)")
        if not step.passed and step.output:
            print(step.output)
    return 0 if result.passed else 1


def cmd_serve(_: argparse.Namespace) -> int:
    from .server import main as serve

    serve()
    return 0


# --------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="code-mcp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(name: str, help_text: str) -> argparse.ArgumentParser:
        child = sub.add_parser(name, help=help_text)
        child.add_argument("-p", "--path", default=".", help="repository root (default: .)")
        child.add_argument("--json", action="store_true", help="emit JSON")
        return child

    index_parser = shared("index", "build or refresh the index")
    index_parser.add_argument("--force", action="store_true", help="re-parse unchanged files")
    index_parser.set_defaults(func=cmd_index)

    find_parser = shared("find", "search symbols and comments")
    find_parser.add_argument("query")
    find_parser.add_argument("-k", "--kind", action="append", help="filter by kind")
    find_parser.add_argument("-n", "--limit", type=int, default=20)
    find_parser.set_defaults(func=cmd_find)

    pack_parser = shared("pack", "pack the causal neighbourhood of a symbol")
    pack_parser.add_argument("file")
    pack_parser.add_argument("name", nargs="?", default=None)
    pack_parser.add_argument("-b", "--budget", type=int, default=30000, dest="budget")
    pack_parser.add_argument(
        "--segment",
        type=int,
        default=None,
        help="fetch one outline block by 1-based index",
    )
    pack_parser.add_argument(
        "--max-target-tokens",
        type=int,
        default=4000,
        dest="max_target_tokens",
        help="target size above which an outline is returned (default: 4000)",
    )
    pack_parser.add_argument(
        "--force",
        action="store_true",
        help="return the pack even when it costs more than reading the file",
    )
    pack_parser.set_defaults(func=cmd_pack)

    slice_parser = shared("slice", "read one symbol or a line range")
    slice_parser.add_argument("file")
    slice_parser.add_argument("name", nargs="?", default=None)
    slice_parser.add_argument("--start", type=int, default=1)
    slice_parser.add_argument("--end", type=int, default=None)
    slice_parser.set_defaults(func=cmd_slice)

    edit_parser = shared("edit", "replace one symbol in place")
    edit_parser.add_argument("file")
    edit_parser.add_argument("name")
    edit_parser.add_argument("--code-file", required=True, help="file holding the new source")
    edit_parser.set_defaults(func=cmd_edit)

    impact_parser = shared("impact", "list what a change would break")
    impact_parser.add_argument("file")
    impact_parser.add_argument("name")
    impact_parser.add_argument("-d", "--depth", type=int, default=2)
    impact_parser.add_argument(
        "--include-tests",
        action="store_true",
        help="also list callers that live in test files",
    )
    impact_parser.add_argument(
        "--include-imports",
        action="store_true",
        help="also list import pseudo-symbols",
    )
    impact_parser.set_defaults(func=cmd_impact)

    review_parser = shared("review", "run health and sustainability checks")
    review_parser.add_argument("-s", "--min-severity", default="P2", dest="min_severity")
    review_parser.add_argument("--files", nargs="*", default=None)
    review_parser.set_defaults(func=cmd_review)

    gate_parser = shared("gate", "run the project gate commands")
    gate_parser.add_argument("--only", default=None, help="comma separated step names")
    gate_parser.set_defaults(func=cmd_gate)

    serve_parser = sub.add_parser("serve", help="run the MCP server over stdio")
    serve_parser.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
