"""Review layer: external linters, structural rules and the project gate."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..types import SEVERITY_ORDER, Finding
from .external import detect_tools, run_external
from .gate import run_gate
from .structural import analyze


def collect(
    root: Path,
    conn: sqlite3.Connection,
    min_severity: str,
    paths: list[str] | None = None,
) -> tuple[list[Finding], list[str], list[str]]:
    """Run structural and external checks, then filter and rank the findings.

    ``SEVERITY_ORDER`` is the single ordering: anything at or below
    ``min_severity`` is kept and sorted by severity, file and line. Returns the
    kept findings together with the external runners that ran and those missing.
    """
    findings = analyze(conn, files=paths)
    tools = detect_tools(root)
    external, ran, missing = run_external(root, tools, paths=paths)
    findings.extend(external)

    cutoff = (
        SEVERITY_ORDER.index(min_severity)
        if min_severity in SEVERITY_ORDER
        else len(SEVERITY_ORDER)
    )
    kept = [item for item in findings if SEVERITY_ORDER.index(item.severity) <= cutoff]
    kept.sort(key=lambda item: (SEVERITY_ORDER.index(item.severity), item.file, item.line))
    return kept, ran, missing


__all__ = ["analyze", "collect", "detect_tools", "run_external", "run_gate"]
