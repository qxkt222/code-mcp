"""Project gate execution: isolation of failures, timeouts and truncation."""

from __future__ import annotations

import sys
from pathlib import Path

from code_mcp.config import Config, GateStepSpec
from code_mcp.review.gate import (
    MAX_OUTPUT_CHARS,
    SUCCESS_OUTPUT_LINES,
    _tail_lines,
    _truncate,
    output_truncated,
    run_gate,
)
from code_mcp.serialize import gate_dict
from code_mcp.types import GateResult, GateStep


def _python(expression: str) -> str:
    return f'"{sys.executable}" -c "{expression}"'


def _script(tmp_path: Path, source: str) -> str:
    path = tmp_path / "gate_script.py"
    path.write_text(source, encoding="utf-8")
    return f'"{sys.executable}" "{path}"'


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


def test_failed_output_is_truncated_keeping_both_ends(tmp_path: Path) -> None:
    command = _python("print('HEAD' + 'x' * 20000 + 'TAIL'); raise SystemExit(1)")
    config = _config(tmp_path, [GateStepSpec("noisy", command)])

    result = run_gate(config)

    output = result.steps[0].output
    assert result.steps[0].passed is False
    assert len(output) == MAX_OUTPUT_CHARS
    # The failure summary lives at the end, so the tail must survive.
    assert output.startswith("HEAD")
    assert output.rstrip().endswith("TAIL")
    assert "...[truncated]..." in output
    assert output_truncated(result.steps[0]) is True


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


def test_tail_lines_keeps_only_the_last_ten() -> None:
    text = "\n".join(f"line{i}" for i in range(1, 21))

    rendered, truncated = _tail_lines(text)

    assert truncated is True
    assert rendered == "\n".join(f"line{i}" for i in range(11, 21))


def test_tail_lines_leaves_short_output_alone() -> None:
    text = "\n".join(f"line{i}" for i in range(1, SUCCESS_OUTPUT_LINES + 1))

    assert _tail_lines(text) == (text, False)


def test_tail_lines_normalises_crlf() -> None:
    text = "\r\n".join(f"line{i}" for i in range(1, 21)) + "\r\n"

    rendered, truncated = _tail_lines(text)

    assert truncated is True
    assert "\r" not in rendered
    assert rendered == "\n".join(f"line{i}" for i in range(11, 21))


def test_passing_step_keeps_only_the_last_ten_lines(tmp_path: Path) -> None:
    command = _script(tmp_path, "for i in range(1, 21):\n    print(f'line{i}')\n")
    config = _config(tmp_path, [GateStepSpec("chatty", command)])

    result = run_gate(config)

    step = result.steps[0]
    assert step.passed is True
    assert step.output.splitlines() == [f"line{i}" for i in range(11, 21)]
    assert output_truncated(step) is True


def test_passing_step_with_short_output_is_untouched(tmp_path: Path) -> None:
    command = _script(tmp_path, "print('one')\nprint('two')\n")
    config = _config(tmp_path, [GateStepSpec("quiet", command)])

    result = run_gate(config)

    step = result.steps[0]
    assert step.passed is True
    assert step.output.splitlines() == ["one", "two"]
    assert output_truncated(step) is False


def test_failed_step_short_output_is_untouched(tmp_path: Path) -> None:
    command = _python("print('boom'); raise SystemExit(2)")
    config = _config(tmp_path, [GateStepSpec("short-fail", command)])

    result = run_gate(config)

    step = result.steps[0]
    assert step.passed is False
    assert step.output.splitlines() == ["boom"]
    assert output_truncated(step) is False


def test_output_truncated_defaults_to_false_for_plain_steps() -> None:
    assert output_truncated(GateStep(name="plain", command="true", exit_code=0)) is False


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

    assert payload["passed"] is False
    assert payload["detected_config"] is True
    (step,) = payload["steps"]
    # ``output_truncated`` is owned by serialize.gate_dict; accept it when
    # present, but the six stable keys must not change.
    assert {key: step[key] for key in (
        "name",
        "command",
        "passed",
        "exit_code",
        "duration_s",
        "output",
    )} == {
        "name": "tests",
        "command": "python -m pytest -q",
        "passed": False,
        "exit_code": 1,
        "duration_s": 1.23,
        "output": "boom",
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
