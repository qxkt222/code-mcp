"""JSON-ready views of the shared result objects.

Both the MCP server and the CLI return the same data, so the shape is defined
once here rather than maintained twice. Only plain dicts and lists are
produced: the caller can hand the result straight to ``json.dump``.
"""

from __future__ import annotations

from .types import Pack, Segment, Slice


def segment_dict(segment: Segment) -> dict:
    """One addressable block of an oversized symbol."""
    return {
        "index": segment.index,
        "start_line": segment.start_line,
        "end_line": segment.end_line,
        "tokens": segment.tokens,
        "label": segment.label,
    }


def slice_dict(slice_: Slice) -> dict:
    """One bounded piece of source text, including its outline metadata."""
    return {
        "file": slice_.file,
        "symbol": slice_.symbol,
        "kind": slice_.kind,
        "start_line": slice_.start_line,
        "end_line": slice_.end_line,
        "tokens": slice_.tokens,
        "text": slice_.text,
        "segments": [segment_dict(segment) for segment in slice_.segments],
        "truncated": slice_.truncated,
        "total_tokens": slice_.total_tokens,
    }


def pack_dict(pack: Pack) -> dict:
    """A whole packed neighbourhood: target, neighbours, tests and budget data."""
    return {
        "target": slice_dict(pack.target) if pack.target else None,
        "callers": [slice_dict(item) for item in pack.callers],
        "callees": [slice_dict(item) for item in pack.callees],
        "imports": [slice_dict(item) for item in pack.imports],
        "tests": [slice_dict(item) for item in pack.tests],
        "tokens_used": pack.token_total,
        "tokens_if_read_file": pack.token_baseline,
        "savings_pct": pack.savings_pct,
        "worth_using": pack.worth_using,
        "note": pack.note,
    }
