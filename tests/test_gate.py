"""Project gate execution: isolation of failures, timeouts and truncation."""

from __future__ import annotations

import sys
from pathlib import Path

from code_mcp.config import Config, GateStepSpec
from code_mcp.review.gate import run_gate


def _python(expression: str) -> str:
    return f'"{sys.executable}" -c "{expression}"'


def _config(root: Path, steps: list[GateStepSpec]) -> Config:
    return Config(root=root, gate_steps=steps)


def test_runs_every_step_after_a_failure(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        [
            GateStepSpec("first", _python("print('one')")),
            GateStepSpec("second", _python("import sys; sys.exit(3)")),
            GateStepSpec("third", _python("print('three')")),
        ],
    )

    result = run_gate(config)

    assert [step.name for step in result.steps] == ["first", "second", "third"]
    assert result.steps[0].passed is True
    assert result.steps[1].exit_code == 3
    assert result.steps[1].passed is False
    assert result.steps[2].passed is True
    assert result.passed is False


def test_all_steps_pass(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        [
            GateStepSpec("a", _python("print(1)")),
            GateStepSpec("b", _python("print(2)")),
        ],
    )

    result = run_gate(config)

    assert result.passed is True
    assert len(result.steps) == 2
    assert all(step.duration_s >= 0 for step in result.steps)


def test_only_filter(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        [
            GateStepSpec("keep", _python("print('keep')")),
            GateStepSpec("skip", _python("print('skip')")),
        ],
    )

    result = run_gate(config, only=["keep"])

    assert [step.name for step in result.steps] == ["keep"]
    assert result.passed is True


def test_output_is_truncated(tmp_path: Path) -> None:
    config = _config(tmp_path, [GateStepSpec("noisy", _python("print('x' * 5000)"))])

    result = run_gate(config)

    assert result.steps[0].passed is True
    assert len(result.steps[0].output) == 4000


def test_timeout_is_isolated(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        [
            GateStepSpec("slow", _python("import time; time.sleep(5)"), timeout=1),
            GateStepSpec("after", _python("print('after')")),
        ],
    )

    result = run_gate(config)

    assert len(result.steps) == 2
    slow = result.steps[0]
    assert slow.passed is False
    assert slow.exit_code != 0
    assert "timed out" in slow.output.lower()
    assert slow.duration_s < 4
    assert result.steps[1].passed is True


def test_missing_command_is_reported(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        [GateStepSpec("missing", "definitely-not-a-real-binary-xyz --flag")],
    )

    result = run_gate(config)

    step = result.steps[0]
    assert step.passed is False
    assert step.exit_code == 127
    assert "failed to start" in step.output
    assert result.passed is False


def test_stderr_is_captured(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        [GateStepSpec("stderr", _python("import sys; sys.stderr.write('boom')"))],
    )

    result = run_gate(config)

    assert result.steps[0].passed is True
    assert "boom" in result.steps[0].output


def test_no_steps_means_not_passed(tmp_path: Path) -> None:
    result = run_gate(_config(tmp_path, []))

    assert result.steps == []
    assert result.passed is False
