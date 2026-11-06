"""Optional Claude Code PreToolUse hook: nudge the agent toward `pack`.

Reads the hook payload from stdin. Exits 0 either way unless ``--enforce`` is
passed, in which case a large read is blocked so the message reaches the model.
The hook never inspects file contents and never fails a session: any parse or
I/O problem degrades to "allow".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LINE_THRESHOLD = 300


def _read_payload() -> dict:
    try:
        raw = sys.stdin.read()
    except OSError:
        return {}
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _target_path(payload: dict) -> Path | None:
    if payload.get("tool_name") != "Read":
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    raw = tool_input.get("file_path") or tool_input.get("path") or tool_input.get("notebook_path")
    if not isinstance(raw, str) or not raw:
        return None
    return Path(raw)


def _line_count(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    enforce = "--enforce" in args

    payload = _read_payload()
    if payload.get("hook_event_name") not in (None, "PreToolUse"):
        return 0

    target = _target_path(payload)
    if target is None:
        return 0

    lines = _line_count(target)
    if lines is None or lines < LINE_THRESHOLD:
        return 0

    message = (
        f"{target.name} is {lines} lines. If you only need one part of it, call the "
        "code-mcp `pack` tool first: it returns that symbol plus its callers, callees "
        "and tests, and reports how many tokens it saved versus reading the file."
    )
    print(message, file=sys.stderr)
    return 2 if enforce else 0


if __name__ == "__main__":
    raise SystemExit(main())
