"""Project configuration: which files to index, and how to run the project gate.

A ``.code-mcp.toml`` at the repository root overrides the defaults. Without one,
gate steps are detected from well-known marker files so a fresh project works
without any setup.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_NAME = ".code-mcp.toml"

DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    ".turbo",
    "coverage",
)


@dataclass(slots=True)
class GateStepSpec:
    """A command the gate should run, declared either in config or detected."""

    name: str
    command: str
    timeout: int = 600


@dataclass(slots=True)
class Config:
    root: Path
    excludes: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    gate_steps: list[GateStepSpec] = field(default_factory=list)
    detected: bool = False

    def is_excluded(self, relative_path: str) -> bool:
        parts = relative_path.replace("\\", "/").split("/")
        return any(exclude in parts for exclude in self.excludes)


def _detect_gate(root: Path) -> list[GateStepSpec]:
    """Best-effort gate detection from marker files, ordered cheapest first."""
    steps: list[GateStepSpec] = []

    if (root / "run_tests.py").is_file():
        steps.append(GateStepSpec("tests", "python run_tests.py"))
    elif (root / "pyproject.toml").is_file() or (root / "pytest.ini").is_file():
        steps.append(GateStepSpec("tests", "python -m pytest -q"))

    if (root / "Cargo.toml").is_file():
        steps.append(GateStepSpec("rust-test", "cargo test --quiet"))

    if (root / "package.json").is_file():
        steps.append(GateStepSpec("js-test", "npm test"))

    return steps


def load_config(root: Path) -> Config:
    """Load ``.code-mcp.toml`` if present, else detect defaults from marker files."""
    root = root.resolve()
    config = Config(root=root, excludes=list(DEFAULT_EXCLUDES))

    path = root / CONFIG_NAME
    if not path.is_file():
        config.gate_steps = _detect_gate(root)
        config.detected = True
        return config

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    project = raw.get("project", {})
    excludes = project.get("exclude", [])
    config.excludes = list(DEFAULT_EXCLUDES) + [entry for entry in excludes if entry not in DEFAULT_EXCLUDES]
    config.languages = list(project.get("languages", []))

    for index, step in enumerate(raw.get("gate", {}).get("steps", [])):
        command = step.get("command")
        if not command:
            continue
        config.gate_steps.append(
            GateStepSpec(
                name=step.get("name", f"step{index + 1}"),
                command=command,
                timeout=int(step.get("timeout", 600)),
            )
        )

    if not config.gate_steps:
        config.gate_steps = _detect_gate(root)
        config.detected = True

    return config
