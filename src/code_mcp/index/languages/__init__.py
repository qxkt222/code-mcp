"""Language front ends for the index layer.

The parsed dataclasses live here and are re-exported from ``index.parsers`` so
the per-language modules can import them without creating an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...types import CONF_NAME

# Node is only used for annotations; importing it keeps type hints honest.
try:  # pragma: no cover - tree-sitter is a hard dependency in practice
    from tree_sitter import Node
except ImportError:  # pragma: no cover
    Node = object  # type: ignore[assignment,misc]


@dataclass
class ParsedSymbol:
    """A definition extracted from a single file."""

    name: str
    qualname: str
    kind: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    signature: str
    docstring: str = ""


@dataclass
class ParsedRef:
    """A reference from a containing symbol (``""`` at module level) to a name."""

    from_qualname: str
    to_name: str
    kind: str
    confidence: str = CONF_NAME


@dataclass
class ParsedFile:
    """Everything the index stores for one source file."""

    symbols: list[ParsedSymbol] = field(default_factory=list)
    refs: list[ParsedRef] = field(default_factory=list)
    imports: list[tuple[str, str]] = field(default_factory=list)
    comments: list[tuple[int, str]] = field(default_factory=list)
    loc: int = 0


def node_start_line(node: Node) -> int:
    """1-based line of a node's first byte.

    Points are tuple subclasses and are indexed on purpose: the ``row`` and
    ``column`` properties crash py-tree-sitter 0.26 on Windows under load.
    """
    return node.start_point[0] + 1


def node_end_line(node: Node) -> int:
    """1-based line of a node's last byte.

    tree-sitter's ``end_point`` is the position *after* the node, so a node
    ending at column 0 belongs to the previous line.
    """
    row = node.end_point[0]
    if node.end_point[1] == 0 and row > node.start_point[0]:
        return row
    return row + 1


def collapse_ws(text: str) -> str:
    """Collapse runs of whitespace (including newlines) into single spaces."""
    return " ".join(text.split())


def clean_block_doc(raw: str) -> str:
    """Turn a ``/** ... */`` body into plain text without leading stars."""
    body = raw
    if body.startswith("/**"):
        body = body[3:]
    if body.endswith("*/"):
        body = body[:-2]
    lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("*"):
            stripped = stripped[1:].strip()
        lines.append(stripped)
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def strip_quotes(text: str) -> str:
    """Remove one layer of matching quotes from a string literal."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        return text[1:-1]
    return text


class Collector:
    """Accumulates parsed pieces for one file and de-duplicates refs."""

    def __init__(self, source: bytes) -> None:
        self.source = source
        self.text = source.decode("utf-8", errors="replace")
        self.result = ParsedFile(loc=len(self.text.splitlines()))
        self._seen_refs: set[tuple[str, str, str]] = set()

    def snippet(self, start: int, end: int) -> str:
        return self.source[start:end].decode("utf-8", errors="replace")

    def node_text(self, node: Node) -> str:
        return self.snippet(node.start_byte, node.end_byte)

    def add_symbol(
        self,
        *,
        name: str,
        qualname: str,
        kind: str,
        start_byte: int,
        end_byte: int,
        start_line: int,
        end_line: int,
        signature: str = "",
        docstring: str = "",
    ) -> ParsedSymbol:
        symbol = ParsedSymbol(
            name=name,
            qualname=qualname,
            kind=kind,
            start_line=start_line,
            end_line=end_line,
            start_byte=start_byte,
            end_byte=end_byte,
            signature=signature,
            docstring=docstring,
        )
        self.result.symbols.append(symbol)
        return symbol

    def add_ref(
        self,
        from_qualname: str,
        to_name: str,
        kind: str,
        confidence: str = CONF_NAME,
    ) -> None:
        if not to_name:
            return
        key = (from_qualname, to_name, kind)
        if key in self._seen_refs:
            return
        self._seen_refs.add(key)
        self.result.refs.append(
            ParsedRef(
                from_qualname=from_qualname,
                to_name=to_name,
                kind=kind,
                confidence=confidence,
            )
        )

    def add_import(self, module: str, alias: str = "") -> None:
        if not module:
            return
        self.result.imports.append((module, alias))

    def add_comment(self, node: Node, text: str | None = None) -> None:
        raw = self.node_text(node) if text is None else text
        raw = raw.strip()
        if raw:
            self.result.comments.append((node_start_line(node), raw))


# Imported last so the dataclasses above exist while the submodules load.
from .python_lang import parse_python  # noqa: E402
from .rust_lang import parse_rust  # noqa: E402
from .typescript_lang import parse_typescript  # noqa: E402

PARSERS = {
    "python": parse_python,
    "typescript": parse_typescript,
    "tsx": parse_typescript,
    "javascript": parse_typescript,
    "rust": parse_rust,
}

__all__ = [
    "PARSERS",
    "Collector",
    "ParsedFile",
    "ParsedRef",
    "ParsedSymbol",
    "clean_block_doc",
    "collapse_ws",
    "node_end_line",
    "node_start_line",
    "parse_python",
    "parse_rust",
    "parse_typescript",
    "strip_quotes",
]
