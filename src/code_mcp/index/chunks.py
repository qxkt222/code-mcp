"""Split oversized symbols into contiguous, addressable segments.

A symbol that costs thousands of tokens is more useful as a signature plus an
outline than as one multi-thousand-token blob: the caller can fetch just the
block it needs by line range. Boundaries land on real syntax nodes parsed with
the same grammars the index uses, so a segment rarely splits an expression in
half.

``segment_symbol`` is intentionally total: it never raises, falling back to
fixed-size line windows when the language is unknown or the tree cannot be
parsed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import tokens
from ..types import Segment
from .parsers import PARSERS

try:  # pragma: no cover - exercised through the public function
    import tree_sitter_python
    import tree_sitter_rust
    import tree_sitter_typescript
    from tree_sitter import Language, Node, Parser

    _TREE_PARSERS: dict[str, Parser] = {
        "python": Parser(Language(tree_sitter_python.language())),
        "typescript": Parser(Language(tree_sitter_typescript.language_typescript())),
        "tsx": Parser(Language(tree_sitter_typescript.language_tsx())),
        "javascript": Parser(Language(tree_sitter_typescript.language_typescript())),
        "rust": Parser(Language(tree_sitter_rust.language())),
    }
except ImportError:  # pragma: no cover - tree-sitter is a hard dependency
    Node = object  # type: ignore[assignment,misc]
    _TREE_PARSERS = {}

SEGMENT_MAX_TOKENS = 1500
SEGMENT_THRESHOLD_TOKENS = 4000
SEGMENT_MIN_TOKENS = 120

# Fallback window for unparseable sources, in lines.
_FALLBACK_WINDOW_LINES = 60
_LABEL_CHARS = 56
_MAX_DEPTH = 64

_FUNCTION_NODES = frozenset(
    {
        "function_definition",
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
        "method_declaration",
        "generator_function",
        "generator_function_declaration",
    }
)
_CLASS_NODES = frozenset(
    {"class_definition", "class_declaration", "abstract_class_declaration"}
)
_JSX_NODES = frozenset({"jsx_element", "jsx_self_closing_element"})
_JSX_WRAPPERS = frozenset(
    {"return_statement", "expression_statement", "parenthesized_expression", "jsx_expression"}
)

__all__ = [
    "SEGMENT_MAX_TOKENS",
    "SEGMENT_MIN_TOKENS",
    "SEGMENT_THRESHOLD_TOKENS",
    "segment_symbol",
]


@dataclass(slots=True)
class _Raw:
    start_line: int
    end_line: int
    tokens: int
    label: str


def segment_symbol(
    source: bytes,
    lang: str,
    start_byte: int,
    end_byte: int,
    max_tokens: int = SEGMENT_MAX_TOKENS,
) -> list[Segment]:
    """Split one symbol into contiguous addressable blocks. Never raises.

    ``start_byte``/``end_byte`` are the raw byte offsets stored by the index.
    The returned segments are 1-based, ordered, non-overlapping and cover the
    symbol's line range. A parse failure or unknown language falls back to
    fixed 60-line windows.
    """
    try:
        return _segment_symbol(source, lang, start_byte, end_byte, max_tokens)
    except Exception:  # noqa: BLE001 - the contract is "never raises"
        try:
            return _fallback(source, start_byte, end_byte)
        except Exception:  # noqa: BLE001
            return []


def _segment_symbol(
    source: bytes, lang: str, start_byte: int, end_byte: int, max_tokens: int
) -> list[Segment]:
    if not isinstance(source, (bytes, bytearray)):
        return _fallback(b"", start_byte, end_byte)
    data = bytes(source)
    if not data or lang not in PARSERS:
        return _fallback(data, start_byte, end_byte)

    parser = _TREE_PARSERS.get(lang)
    if parser is None:
        return _fallback(data, start_byte, end_byte)

    budget = int(max_tokens)
    if budget <= 0:
        budget = SEGMENT_MAX_TOKENS
    if budget < SEGMENT_MIN_TOKENS:
        budget = SEGMENT_MIN_TOKENS

    start, end = _requested_bytes(data, start_byte, end_byte)
    if start >= end:
        return _fallback(data, start_byte, end_byte)

    tree = parser.parse(data)
    node = tree.root_node.descendant_for_byte_range(start, end)
    if node is None:
        return _fallback(data, start_byte, end_byte)
    node = _covering_node(node, start, end)

    request_start = _line_at(data, start)
    request_end = _line_at(data, end - 1)
    lines = data.decode("utf-8", errors="replace").splitlines()
    raw = _split_node(data, lines, node, budget, 0)
    if not raw:
        return _fallback(data, start_byte, end_byte)
    segments = _finalize(lines, raw, request_start, request_end)
    if not segments:
        return _fallback(data, start_byte, end_byte)
    return segments


# --------------------------------------------------------------------------- #
# Node traversal
# --------------------------------------------------------------------------- #


def _covering_node(node: Node, start: int, end: int) -> Node:
    """Ascend until the node covers the requested byte range."""
    current = node
    while current.parent is not None and (
        current.start_byte > start or current.end_byte < end
    ):
        current = current.parent
    return current


def _split_node(
    source: bytes, lines: list[str], node: Node, budget: int, depth: int
) -> list[_Raw]:
    """Greedy recursion: one node, its only child, or contiguous child groups."""
    node_weight = _node_weight(lines, node)
    if node_weight <= budget or depth >= _MAX_DEPTH:
        return [_raw_from_node(source, lines, node)]

    children = [
        child for child in node.named_children if child.end_byte > child.start_byte
    ]
    if len(children) == 1:
        child = children[0]
        if child.start_byte <= node.start_byte and child.end_byte >= node.end_byte:
            return [_raw_from_node(source, lines, node)]
        return _split_node(source, lines, child, budget, depth + 1)
    if not children:
        return [_raw_from_node(source, lines, node)]

    result: list[_Raw] = []
    current: list[Node] = []
    current_start = 0
    current_end = 0
    current_weight = 0.0
    for child in children:
        child_weight = _node_weight(lines, child)
        if child_weight > budget:
            if current:
                result.append(_group_raw(source, lines, current))
                current, current_weight = [], 0.0
            result.extend(_split_node(source, lines, child, budget, depth + 1))
            continue
        if not current:
            current = [child]
            current_start = _start_line(child)
            current_end = _end_line(child)
            current_weight = child_weight
            continue
        addition = _grow_weight(lines, current_start, current_end, child)
        if current_weight + addition <= budget:
            current.append(child)
            current_end = max(current_end, _end_line(child))
            current_weight += addition
        else:
            result.append(_group_raw(source, lines, current))
            current = [child]
            current_start = _start_line(child)
            current_end = _end_line(child)
            current_weight = child_weight
    if current:
        result.append(_group_raw(source, lines, current))
    return result


def _raw_from_node(source: bytes, lines: list[str], node: Node) -> _Raw:
    start_line = _start_line(node)
    end_line = _end_line(node)
    return _Raw(
        start_line=start_line,
        end_line=end_line,
        tokens=_line_tokens(lines, start_line, end_line),
        label=_node_label(source, node),
    )


def _group_raw(source: bytes, lines: list[str], children: list[Node]) -> _Raw:
    first = children[0]
    last = children[-1]
    start_line = _start_line(first)
    end_line = _end_line(last)
    return _Raw(
        start_line=start_line,
        end_line=end_line,
        tokens=_line_tokens(lines, start_line, end_line),
        label=_node_label(source, first),
    )


def _line_text(lines: list[str], start_line: int, end_line: int) -> str:
    if not lines:
        return ""
    first = min(max(1, start_line), len(lines))
    last = min(max(first, end_line), len(lines))
    return "\n".join(lines[first - 1 : last])


def _line_tokens(lines: list[str], start_line: int, end_line: int) -> int:
    return tokens.estimate(_line_text(lines, start_line, end_line))


def _line_weight(lines: list[str], start_line: int, end_line: int) -> float:
    """Unrounded token cost; per-line integer rounding would underestimate."""
    text = _line_text(lines, start_line, end_line)
    if not text:
        return 0.0
    non_ascii = sum(1 for char in text if ord(char) > 127)
    ascii_count = len(text) - non_ascii
    return (
        ascii_count / tokens.ASCII_CHARS_PER_TOKEN
        + non_ascii / tokens.NON_ASCII_CHARS_PER_TOKEN
    )


def _node_weight(lines: list[str], node: Node) -> float:
    """Token cost of the full lines a node touches.

    Line ranges rather than exact byte ranges: indentation between sibling
    statements is real content, and a byte-slice sum undercounts it badly.
    """
    return _line_weight(lines, _start_line(node), _end_line(node))


def _grow_weight(
    lines: list[str], start_line: int, end_line: int, child: Node
) -> float:
    """Extra cost of extending a line range to include ``child``.

    Joining lines adds one newline per line boundary, and any uncovered lines
    between siblings belong to the group too; both are missed by summing the
    per-child weights.
    """
    child_start = _start_line(child)
    child_end = _end_line(child)
    if child_start > end_line:
        extra = (child_end - end_line) / tokens.ASCII_CHARS_PER_TOKEN
        if child_start > end_line + 1:
            extra += _line_weight(lines, end_line + 1, child_start - 1)
        return extra + _line_weight(lines, child_start, child_end)
    # Shared or overlapping line: recompute the union exactly (rare).
    return _line_weight(lines, start_line, max(end_line, child_end)) - _line_weight(
        lines, start_line, end_line
    )


def _start_line(node: Node) -> int:
    # Points are tuple subclasses; ``.row`` crashes py-tree-sitter 0.26 on
    # Windows, so index the point on purpose.
    return node.start_point[0] + 1


def _end_line(node: Node) -> int:
    row = node.end_point[0]
    if node.end_point[1] == 0 and row > node.start_point[0]:
        return row
    return row + 1


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #


def _finalize(
    lines: list[str], raw: list[_Raw], request_start: int, request_end: int
) -> list[Segment]:
    merged: list[_Raw] = []
    for piece in raw:
        if merged and (
            piece.tokens < SEGMENT_MIN_TOKENS
            or merged[-1].tokens < SEGMENT_MIN_TOKENS
        ):
            merged[-1] = _merge(merged[-1], piece)
        else:
            merged.append(piece)
    if len(merged) > 1 and merged[-1].tokens < SEGMENT_MIN_TOKENS:
        merged[-2] = _merge(merged[-2], merged[-1])
        merged.pop()

    clamped: list[_Raw] = []
    for piece in merged:
        piece.start_line = max(piece.start_line, request_start)
        piece.end_line = min(piece.end_line, request_end)
        if piece.start_line <= piece.end_line:
            clamped.append(piece)
    if not clamped:
        return []
    clamped[0].start_line = request_start
    clamped[-1].end_line = request_end

    ordered: list[_Raw] = [clamped[0]]
    for piece in clamped[1:]:
        previous = ordered[-1]
        if piece.end_line <= previous.end_line:
            # Fully contained: its lines already belong to the previous block.
            continue
        # Trim a shared boundary line, or pull the start forward over a gap,
        # so the outline is a true partition with no double-served lines.
        piece.start_line = previous.end_line + 1
        ordered.append(piece)

    segments: list[Segment] = []
    for position, piece in enumerate(ordered, start=1):
        text = "\n".join(lines[piece.start_line - 1 : piece.end_line])
        segments.append(
            Segment(
                index=position,
                start_line=piece.start_line,
                end_line=piece.end_line,
                tokens=tokens.estimate(text),
                label=piece.label,
            )
        )
    return segments


def _merge(left: _Raw, right: _Raw) -> _Raw:
    return _Raw(
        start_line=left.start_line,
        end_line=max(left.end_line, right.end_line),
        tokens=left.tokens + right.tokens,
        label=left.label,
    )


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def _node_label(source: bytes, node: Node) -> str:
    # The name identifier of a function/class declaration labels the signature
    # block as ``def name`` instead of the bare identifier.
    parent = node.parent
    if parent is not None and parent.type in _FUNCTION_NODES:
        name_node = parent.child_by_field_name("name")
        if name_node is not None and _same_node(name_node, node):
            return _clip(f"def {_node_text(source, node)}")

    jsx = _first_jsx(node)
    if jsx is not None:
        opening = jsx.child_by_field_name("open_tag")
        if opening is None:
            opening = jsx
        name = opening.child_by_field_name("name")
        if name is not None:
            return _clip(f"<{_node_text(source, name)}>")

    kind = node.type
    if kind == "pair":
        key = node.child_by_field_name("key")
        if key is not None:
            return _clip(_node_text(source, key))
    if kind in _FUNCTION_NODES or kind == "variable_declarator":
        name = node.child_by_field_name("name")
        if name is not None:
            return _clip(f"def {_node_text(source, name)}")
    if kind in _CLASS_NODES:
        name = node.child_by_field_name("name")
        if name is not None:
            return _clip(f"class {_node_text(source, name)}")
    return _clip(_first_line(source, node))


def _same_node(left: Node, right: Node) -> bool:
    return (
        left.type == right.type
        and left.start_byte == right.start_byte
        and left.end_byte == right.end_byte
    )


def _first_jsx(node: Node) -> Node | None:
    current = node
    for _ in range(8):
        if current.type in _JSX_NODES:
            return current
        if current.type not in _JSX_WRAPPERS:
            return None
        children = [child for child in current.named_children if child.type != "comment"]
        if len(children) != 1:
            return None
        current = children[0]
    return None


def _node_text(source: bytes, node: Node) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _first_line(source: bytes, node: Node) -> str:
    raw = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    # A node can start exactly at a line end (for example the tail of a JSX
    # expression); use the first line that carries text.
    for line in raw.splitlines()[:4]:
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _clip(text: str) -> str:
    return " ".join(text.split())[:_LABEL_CHARS]


# --------------------------------------------------------------------------- #
# Fallback and byte helpers
# --------------------------------------------------------------------------- #


def _requested_bytes(source: bytes, start_byte: int, end_byte: int) -> tuple[int, int]:
    start = max(0, int(start_byte or 0))
    end = int(end_byte or 0)
    if end > len(source):
        end = len(source)
    if start >= len(source) or end <= start:
        return 0, len(source)
    return start, end


def _line_at(source: bytes, offset: int) -> int:
    offset = min(max(0, int(offset)), len(source))
    return source.count(b"\n", 0, offset) + 1


def _fallback(source: bytes, start_byte: int, end_byte: int) -> list[Segment]:
    data = bytes(source) if isinstance(source, (bytes, bytearray)) else b""
    if not data:
        return [Segment(index=1, start_line=1, end_line=1, tokens=0, label="")]

    start, end = _requested_bytes(data, start_byte, end_byte)
    if start >= end:
        start, end = 0, len(data)
    lines = data.decode("utf-8", errors="replace").splitlines()
    if not lines:
        return [Segment(index=1, start_line=1, end_line=1, tokens=0, label="")]

    start_line = min(_line_at(data, start), len(lines))
    end_line = min(_line_at(data, max(start, end - 1)), len(lines))
    end_line = max(start_line, end_line)

    segments: list[Segment] = []
    cursor = start_line
    while cursor <= end_line:
        last = min(cursor + _FALLBACK_WINDOW_LINES - 1, end_line)
        text = "\n".join(lines[cursor - 1 : last])
        segments.append(
            Segment(
                index=len(segments) + 1,
                start_line=cursor,
                end_line=last,
                tokens=tokens.estimate(text),
                label=_clip(lines[cursor - 1]) if cursor - 1 < len(lines) else "",
            )
        )
        cursor = last + 1
    return segments
