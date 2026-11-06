"""Project gate execution: isolation of failures, timeouts and truncation."""

from __future__ import annotations

import sys
from pathlib import Path

from code_mcp.config import Config, GateStepSpec
from code_mcp.review.gate import MAX_OUTPUT_CHARS, _truncate, run_gate
from code_mcp.serialize import gate_dict
from code_mcp.types import GateResult, GateStep


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


def test_output_is_truncated_keeping_both_ends(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        [GateStepSpec("noisy", _python("print('HEAD' + 'x' * 20000 + 'TAIL')"))],
    )

    result = run_gate(config)

    output = result.steps[0].output
    assert result.steps[0].passed is True
    assert len(output) == MAX_OUTPUT_CHARS
    # The failure summary lives at the end, so the tail must survive.
    assert output.startswith("HEAD")
    assert output.rstrip().endswith("TAIL")
    assert "...[truncated]..." in output


def test_truncate_uses_one_third_head_and_rest_tail() -> None:
    text = "a" * 1000 + "b" * 1000

    output = _truncate(text, limit=300)

    head, marker, tail = output.partition("\n...[truncated]...\n")
    assert marker
    assert len(output) == 300
    assert head == "a" * 100
    assert tail == "b" * (300 - 100 - len(marker))


def test_short_output_is_untouched() -> None:
    assert _truncate("small", limit=300) == "small"


def test_gate_dict_matches_the_mcp_shape() -> None:
    result = GateResult(
        steps=[
            GateStep(
                name="tests",
                command="python -m pytest -q",
                exit_code=1,
                duration_s=1.234,
                output="boom",
                passed=False,
            )
        ]
    )

    payload = gate_dict(result, detected_config=True)

    assert payload == {
        "passed": False,
        "detected_config": True,
        "steps": [
            {
                "name": "tests",
                "command": "python -m pytest -q",
                "passed": False,
                "exit_code": 1,
                "duration_s": 1.23,
                "output": "boom",
            }
        ],
    }


def test_gate_dict_leaves_unknown_detection_as_none() -> None:
    """``detected_config`` lives on Config, not GateResult: never guess it."""
    assert gate_dict(GateResult()) == {
        "passed": False,
        "detected_config": None,
        "steps": [],
    }


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
