"""Path containment helpers shared by every tool that touches a user file.

The MCP surface is driven by a model, which may be steered by untrusted content
inside the indexed repository. Every path that reaches the filesystem therefore
has to be proven to live under the repository root.
"""

from __future__ import annotations

from pathlib import Path


def resolve_within(root: Path, rel: str) -> Path | None:
    """Resolve ``rel`` under ``root``, or return None when it escapes the root."""
    raw = (rel or "").strip()
    if not raw:
        return None

    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate

    try:
        resolved = candidate.resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return resolved


def is_test_path(path: str) -> bool:
    """True for any file the main test conventions treat as a test module."""
    normalized = path.replace("\\", "/")
    lowered = normalized.lower()
    name = lowered.rsplit("/", 1)[-1]

    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    if ".test." in name or ".spec." in name:
        return True
    parts = lowered.split("/")[:-1]
    return "tests" in parts or "__tests__" in parts
