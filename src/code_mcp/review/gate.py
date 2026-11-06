"""Run the configured project gate step by step.

Every step runs even when an earlier one fails; the caller gets the full list
of exit codes and truncated output. Commands are split into an argument vector
and executed with ``shell=False``.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..config import Config, GateStepSpec
from ..types import GateResult, GateStep

MAX_OUTPUT_CHARS = 12000
_TRUNCATION_MARKER = "\n...[truncated]...\n"
_CMD_SUFFIXES = {".cmd", ".bat"}


def run_gate(config: Config, only: list[str] | None = None) -> GateResult:
    """Run all configured steps (or just ``only``), collecting every result."""
    selected = [step for step in config.gate_steps if only is None or step.name in only]
    result = GateResult()
    for spec in selected:
        result.steps.append(_run_step(spec, config.root))
    return result


def _run_step(spec: GateStepSpec, root: Path) -> GateStep:
    started = time.monotonic()
    exit_code = 0
    output = ""
    passed = False

    try:
        argv = _split_command(spec.command)
        if not argv:
            raise ValueError("empty command")
        resolved = shutil.which(argv[0])
        if resolved is None and argv[0] in {"python", "python3"}:
            resolved = sys.executable
        executable = resolved or argv[0]
        completed = subprocess.run(
            _spawn(executable, argv[1:]),
            cwd=str(root),
            capture_output=True,
            timeout=spec.timeout,
            shell=False,
        )
        exit_code = completed.returncode
        output = _decode(completed.stdout) + _decode(completed.stderr)
        passed = exit_code == 0
    except subprocess.TimeoutExpired as exc:
        exit_code = -1
        output = _decode(getattr(exc, "stdout", None)) + _decode(getattr(exc, "stderr", None))
        output += f"\n[gate] timed out after {spec.timeout}s"
    except (OSError, ValueError) as exc:
        exit_code = 127
        output = f"[gate] failed to start: {exc}"

    duration = time.monotonic() - started
    return GateStep(
        name=spec.name,
        command=spec.command,
        exit_code=exit_code,
        duration_s=round(duration, 3),
        output=_truncate(output),
        passed=passed,
    )


def _split_command(command: str) -> list[str]:
    """Split a gate command without invoking a shell."""
    parts = shlex.split(command, posix=os.name != "nt")
    if os.name == "nt":
        parts = [part.strip("\"'") for part in parts]
    return parts


def _spawn(executable: str, argv: list[str]) -> list[str]:
    if os.name == "nt" and Path(executable).suffix.lower() in _CMD_SUFFIXES:
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return [comspec, "/c", executable, *argv]
    return [executable, *argv]


def _decode(raw: bytes | str | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8", errors="replace")


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep both ends of an oversized output.

    Test runners and compilers report their failure summary at the end, while
    the beginning still carries the command echo and the first errors, so the
    head (one third) and the tail (two thirds) are preserved.
    """
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head - len(_TRUNCATION_MARKER)
    if tail <= 0:
        # Degenerate limit: keeping any tail would exceed the budget.
        return text[:limit]
    return text[:head] + _TRUNCATION_MARKER + text[-tail:]
