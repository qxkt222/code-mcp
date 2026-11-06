"""Reverse dependency traversal: which symbols a change can affect."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ..paths import is_test_path
from ..types import CONF_BRIDGE, REF_IMPORT, Symbol
from .refs import SYMBOL_COLUMNS, row_to_symbol

_MODULE_KIND = "module"


@dataclass(slots=True)
class ImpactNode:
    """One symbol reached while walking the reverse dependency graph."""

    symbol: Symbol
    depth: int
    via: str  # call / import / bridge

    @property
    def line(self) -> int:
        """Definition line, taken from ``symbol.start_line``."""
        return self.symbol.start_line


class ImpactNodes(list[ImpactNode]):
    """Impact nodes plus counts of the candidates the default filters hid.

    A ``list`` subclass so existing callers that iterate, index or take
    ``len()`` keep working unchanged. ``filtered`` holds disjoint counts: a
    node matching both filters is counted under ``tests``, so the values add
    up to the number of hidden nodes.
    """

    def __init__(
        self,
        nodes: list[ImpactNode] | None = None,
        filtered: dict[str, int] | None = None,
    ) -> None:
        super().__init__(nodes or [])
        self.filtered: dict[str, int] = (
            filtered if filtered is not None else {"tests": 0, "imports": 0}
        )


# Strongest edge wins when a node is reachable through several refs.
_VIA_PRIORITY = ("bridge", "import", "call")

_IMPACT_QUERY = f"""
SELECT {SYMBOL_COLUMNS}, r.kind AS ref_kind, r.confidence AS ref_confidence
FROM refs AS r
JOIN symbols AS s ON s.id = r.from_symbol_id
JOIN files AS f ON f.id = s.file_id
WHERE r.to_symbol_id = ?
ORDER BY f.path, s.start_line, s.id
"""


def _via(kind: str, confidence: str) -> str:
    if confidence == CONF_BRIDGE:
        return "bridge"
    if kind == REF_IMPORT:
        return "import"
    return "call"


def _best_via(vias: set[str]) -> str:
    for candidate in _VIA_PRIORITY:
        if candidate in vias:
            return candidate
    return "call"


def impact(
    conn: sqlite3.Connection,
    symbol_id: int,
    depth: int = 2,
    limit: int = 100,
    include_tests: bool = False,
    include_imports: bool = False,
) -> ImpactNodes:
    """Walk callers transitively up to ``depth`` hops from ``symbol_id``.

    Nodes are returned breadth-first, ordered by file and line inside each
    level. The origin symbol is never included, and each symbol appears once
    at its shortest distance.

    Import pseudo-symbols (``kind='module'``) and callers that live in test
    files are hidden by default: they dominate the output without naming an
    actionable call site. ``include_tests`` and ``include_imports`` restore
    them. ``filtered`` records how many candidates each filter hid; hidden
    nodes still count as graph paths, so symbols behind them are reached at
    their true depth.
    """
    nodes = ImpactNodes()
    if depth <= 0 or limit <= 0:
        return nodes

    seen = {symbol_id}
    frontier = [symbol_id]

    for level in range(1, depth + 1):
        found: dict[int, tuple[Symbol, set[str]]] = {}
        for current in frontier:
            for row in conn.execute(_IMPACT_QUERY, (current,)):
                symbol = row_to_symbol(row)
                if symbol.id in seen:
                    continue
                entry = found.get(symbol.id)
                via = _via(row["ref_kind"], row["ref_confidence"])
                if entry is None:
                    found[symbol.id] = (symbol, {via})
                else:
                    entry[1].add(via)

        if not found:
            break

        ordered = sorted(
            found.values(),
            key=lambda item: (item[0].file, item[0].start_line, item[0].id),
        )
        next_frontier: list[int] = []
        for symbol, vias in ordered:
            seen.add(symbol.id)
            next_frontier.append(symbol.id)
            if not include_tests and is_test_path(symbol.file):
                nodes.filtered["tests"] += 1
                continue
            if not include_imports and symbol.kind == _MODULE_KIND:
                nodes.filtered["imports"] += 1
                continue
            if len(nodes) >= limit:
                return nodes
            nodes.append(ImpactNode(symbol=symbol, depth=level, via=_best_via(vias)))

        frontier = next_frontier

    return nodes
