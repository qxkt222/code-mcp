"""Measure how much `pack` actually saves on a real repository.

Usage:
    uv run python scripts/bench.py <repo-root> [--sample 30]

Reports, per sampled symbol, the packed token count against the cost of reading
the containing file. The whole point of the tool is this ratio; if the median
saving is small the design is not paying for itself.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from code_mcp.config import load_config
from code_mcp.db import open_index
from code_mcp.graph.refs import resolve_refs
from code_mcp.graph.tauri import build_bridges
from code_mcp.index.builder import build_index
from code_mcp.pack import pack


def _sample_symbols(conn, count: int) -> list[tuple[str, str]]:
    rows = conn.execute(
        "SELECT f.path, s.name FROM symbols s JOIN files f ON f.id = s.file_id "
        "WHERE s.kind IN ('function', 'method') ORDER BY s.id"
    ).fetchall()
    pairs = [(row["path"], row["name"]) for row in rows]
    if len(pairs) <= count:
        return pairs
    return random.Random(0).sample(pairs, count)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--sample", type=int, default=30)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1

    conn = open_index(root)
    try:
        stats = build_index(root, conn, load_config(root))
        resolve_refs(conn)
        build_bridges(conn)
        conn.commit()
        print(
            f"indexed {stats.files_indexed} files, skipped {stats.files_skipped}, "
            f"{stats.symbols} symbols in {stats.duration_s:.1f}s"
        )

        samples = _sample_symbols(conn, args.sample)
        if not samples:
            print("no symbols to sample", file=sys.stderr)
            return 1

        savings: list[float] = []
        print(f"\n{'symbol':<48} {'packed':>8} {'file':>8} {'saved':>7}")
        print("-" * 74)
        for file, name in samples:
            result = pack(conn, root, file, name)
            if result.token_total == 0:
                continue
            savings.append(result.savings_pct)
            label = f"{file}:{name}"[:48]
            print(
                f"{label:<48} {result.token_total:>8} "
                f"{result.token_baseline:>8} {result.savings_pct:>6.1f}%"
            )

        print("-" * 74)
        if savings:
            print(f"samples:      {len(savings)}")
            print(f"median saved: {statistics.median(savings):.1f}%")
            print(f"mean saved:   {statistics.mean(savings):.1f}%")
            print(f"worst case:   {min(savings):.1f}%")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
