"""Reverse dependency traversal: which symbols a change can affect."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ..types import CONF_BRIDGE, REF_IMPORT, Symbol
from .refs import SYMBOL_COLUMNS, row_to_symbol


@dataclass(slots=True)
class ImpactNode:
    """One symbol reached while walking the reverse dependency graph."""

    symbol: Symbol
    depth: int
    via: str  # call / import / bridge


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
    conn: sqlite3.Connection, symbol_id: int, depth: int = 2, limit: int = 100
) -> list[ImpactNode]:
    """Walk callers transitively up to ``depth`` hops from ``symbol_id``.

    Nodes are returned breadth-first, ordered by file and line inside each
    level. The origin symbol is never included, and each symbol appears once
    at its shortest distance.
    """
    if depth <= 0 or limit <= 0:
        return []

    seen = {symbol_id}
    frontier = [symbol_id]
    nodes: list[ImpactNode] = []

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
            if len(nodes) >= limit:
                return nodes
            seen.add(symbol.id)
            nodes.append(ImpactNode(symbol=symbol, depth=level, via=_best_via(vias)))
            next_frontier.append(symbol.id)

        frontier = next_frontier

    return nodes
