"""Run the external linters that actually exist for a repository.

Detection is based on marker files, execution goes through ``shutil.which`` and
``subprocess.run`` with ``shell=False``. Captured output is decoded with
``errors="replace"`` because Windows tools may answer in a non-UTF-8 codepage.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from ..types import Finding

TOOL_TIMEOUT_S = 300

# Marker files that make the Python tools worth probing.
_PY_MARKERS: tuple[str, ...] = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "tox.ini",
    "ruff.toml",
    ".ruff.toml",
    "mypy.ini",
    ".mypy.ini",
    ".bandit",
    "Pipfile",
)

_CMD_SUFFIXES = {".cmd", ".bat"}

# Cargo (and some TypeScript setups) print diagnostics on stderr.
_STDERR_DIAGNOSTICS = {"mypy", "tsc", "cargo"}

_MYPY_RE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?:\s*(?P<col>\d+):)?\s*"
    r"(?P<sev>error|warning|note):\s*(?P<msg>.*)$"
)
_MYPY_CODE_RE = re.compile(r"\[(?P<code>[a-z0-9_-]+)\]\s*$")
_TSC_RE = re.compile(
    r"^(?P<file>.+?)\((?P<line>\d+),(?P<col>\d+)\):\s*"
    r"(?P<sev>error|warning)\s+(?P<code>TS\d+):\s*(?P<msg>.*)$"
)
_CARGO_RE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s*"
    r"(?P<sev>error|warning)(?:\[(?P<code>[A-Z]\d+)\])?:\s*(?P<msg>.*)$"
)


def detect_tools(root: Path) -> list[str]:
    """Return the external tools applicable to ``root``, in a stable order."""
    root = Path(root)
    tools: list[str] = []

    pythonish = any((root / marker).is_file() for marker in _PY_MARKERS)
    if not pythonish:
        pythonish = next(root.glob("*.py"), None) is not None
    if pythonish:
        tools.extend(["ruff", "mypy", "bandit"])

    if (root / "tsconfig.json").is_file() or _package_uses_typescript(root):
        tools.append("tsc")

    if (root / "Cargo.toml").is_file():
        tools.append("cargo")

    return tools


def run_external(
    root: Path,
    tools: list[str],
    paths: list[str] | None = None,
) -> tuple[list[Finding], list[str], list[str]]:
    """Run ``tools`` against ``root``.

    Returns ``(findings, ran, missing)``. Tools missing from PATH are collected
    in ``missing`` instead of raising; one failing tool never stops the others.
    """
    root = Path(root)
    targets = list(paths) if paths else ["."]
    findings: list[Finding] = []
    ran: list[str] = []
    missing: list[str] = []

    for tool in tools:
        executable = shutil.which(tool)
        if executable is None:
            missing.append(tool)
            continue

        argv = _argv_for(tool, targets)
        if argv is None:
            missing.append(tool)
            continue

        try:
            completed = subprocess.run(
                _spawn(executable, argv),
                cwd=str(root),
                capture_output=True,
                timeout=TOOL_TIMEOUT_S,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            ran.append(tool)
            continue
        except OSError:
            missing.append(tool)
            continue

        ran.append(tool)
        stdout = (completed.stdout or b"").decode("utf-8", errors="replace")
        findings.extend(_parse_output(tool, stdout, root))
        if tool in _STDERR_DIAGNOSTICS:
            stderr = (completed.stderr or b"").decode("utf-8", errors="replace")
            findings.extend(_parse_output(tool, stderr, root))

    return _dedupe(findings), ran, missing


def _package_uses_typescript(root: Path) -> bool:
    package = root / "package.json"
    if not package.is_file():
        return False
    try:
        raw = json.loads(package.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(raw, dict):
        return False
    deps: dict[str, object] = {}
    for key in ("dependencies", "devDependencies"):
        value = raw.get(key)
        if isinstance(value, dict):
            deps.update(value)
    return "typescript" in deps


def _argv_for(tool: str, targets: list[str]) -> list[str] | None:
    if tool == "ruff":
        return ["check", "--output-format=json", "--no-cache", *targets]
    if tool == "mypy":
        return ["--no-error-summary", "--show-column-numbers", "--no-color-output", *targets]
    if tool == "bandit":
        return ["-q", "-r", "-f", "json", *targets]
    if tool == "tsc":
        return ["--noEmit", "--pretty", "false"]
    if tool == "cargo":
        return ["check", "--quiet", "--message-format=short"]
    return None


def _spawn(executable: str, argv: list[str]) -> list[str]:
    """Wrap ``.cmd``/``.bat`` shims for ``shell=False`` on Windows."""
    if os.name == "nt" and Path(executable).suffix.lower() in _CMD_SUFFIXES:
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return [comspec, "/c", executable, *argv]
    return [executable, *argv]


def _parse_output(tool: str, text: str, root: Path) -> list[Finding]:
    if tool == "ruff":
        return _parse_ruff(text, root)
    if tool == "mypy":
        return _parse_mypy(text, root)
    if tool == "bandit":
        return _parse_bandit(text, root)
    if tool == "tsc":
        return _parse_tsc(text, root)
    if tool == "cargo":
        return _parse_cargo(text, root)
    return []


def _parse_ruff(text: str, root: Path) -> list[Finding]:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    findings: list[Finding] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "")
        location = item.get("location") or {}
        findings.append(
            Finding(
                rule=f"ruff:{code}" if code else "ruff",
                severity=_ruff_severity(code),
                file=_repo_relative(root, str(item.get("filename") or "")),
                line=_as_int(location.get("row"), 1),
                message=str(item.get("message") or ""),
            )
        )
    return findings


def _ruff_severity(code: str) -> str:
    if code.startswith(("F", "E9")):
        return "P1"
    return "P2"


def _parse_mypy(text: str, root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for raw_line in text.splitlines():
        match = _MYPY_RE.match(raw_line.strip())
        if match is None:
            continue
        message = match.group("msg")
        code_match = _MYPY_CODE_RE.search(message)
        code = code_match.group("code") if code_match else ""
        severity = {"error": "P1", "warning": "P2", "note": "P3"}[match.group("sev")]
        findings.append(
            Finding(
                rule=f"mypy:{code}" if code else "mypy",
                severity=severity,
                file=_repo_relative(root, match.group("file")),
                line=_as_int(match.group("line"), 1),
                message=message,
            )
        )
    return findings


def _parse_bandit(text: str, root: Path) -> list[Finding]:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    findings: list[Finding] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        test_id = str(item.get("test_id") or "")
        severity = {"HIGH": "P1", "MEDIUM": "P2", "LOW": "P3"}.get(
            str(item.get("issue_severity") or "").upper(), "P3"
        )
        findings.append(
            Finding(
                rule=f"bandit:{test_id}" if test_id else "bandit",
                severity=severity,
                file=_repo_relative(root, str(item.get("filename") or "")),
                line=_as_int(item.get("line_number"), 1),
                message=str(item.get("issue_text") or "").strip(),
            )
        )
    return findings


def _parse_tsc(text: str, root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for raw_line in text.splitlines():
        match = _TSC_RE.match(raw_line.strip())
        if match is None:
            continue
        findings.append(
            Finding(
                rule=f"tsc:{match.group('code')}",
                severity="P1" if match.group("sev") == "error" else "P2",
                file=_repo_relative(root, match.group("file")),
                line=_as_int(match.group("line"), 1),
                message=match.group("msg"),
            )
        )
    return findings


def _parse_cargo(text: str, root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for raw_line in text.splitlines():
        match = _CARGO_RE.match(raw_line.strip())
        if match is None:
            continue
        code = match.group("code") or ""
        findings.append(
            Finding(
                rule=f"cargo:{code}" if code else "cargo",
                severity="P1" if match.group("sev") == "error" else "P2",
                file=_repo_relative(root, match.group("file")),
                line=_as_int(match.group("line"), 1),
                message=match.group("msg"),
            )
        )
    return findings


def _repo_relative(root: Path, filename: str) -> str:
    normalized = filename.replace("\\", "/")
    if not normalized:
        return ""
    try:
        return Path(filename).resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return normalized


def _as_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, str, int, str]] = set()
    unique: list[Finding] = []
    for finding in findings:
        key = (finding.rule, finding.file, finding.line, finding.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique
