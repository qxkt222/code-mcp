"""Behavioural tests for segmenting oversized symbols."""

from __future__ import annotations

from code_mcp.index.chunks import (
    SEGMENT_MAX_TOKENS,
    SEGMENT_MIN_TOKENS,
    SEGMENT_THRESHOLD_TOKENS,
    segment_symbol,
)


def _python_symbol(statements: int) -> bytes:
    lines = ['def big():', '    """Doc."""', "    total = 0"]
    lines += [f"    total += {index}" for index in range(statements)]
    lines += ["    return total"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _assert_partition(segments, first_line: int, last_line: int) -> None:
    assert segments
    assert segments[0].start_line == first_line
    assert segments[-1].end_line == last_line
    for position, segment in enumerate(segments):
        assert segment.index == position + 1
        assert 1 <= segment.start_line <= segment.end_line <= last_line
        assert segment.tokens > 0
        if position:
            assert segments[position - 1].end_line + 1 == segment.start_line


def test_segments_are_deterministic():
    source = _python_symbol(400)
    end = len(source.rstrip())

    first = segment_symbol(source, "python", 0, end)
    second = segment_symbol(source, "python", 0, end)

    assert first == second
    assert len(first) > 1


def test_segments_form_a_partition_covering_the_symbol():
    source = _python_symbol(400)
    line_count = len(source.decode("utf-8").splitlines())

    segments = segment_symbol(source, "python", 0, len(source.rstrip()))

    _assert_partition(segments, 1, line_count)


def test_small_symbol_stays_one_segment():
    source = b"def tiny():\n    return 1\n"

    segments = segment_symbol(source, "python", 0, len(source.rstrip()))

    assert len(segments) == 1
    assert (segments[0].start_line, segments[0].end_line) == (1, 2)
    assert segments[0].label == "def tiny"
    assert segments[0].tokens < SEGMENT_THRESHOLD_TOKENS


def test_max_tokens_controls_block_size():
    source = _python_symbol(400)

    segments = segment_symbol(source, "python", 0, len(source.rstrip()), max_tokens=400)

    assert len(segments) > 1
    # A boundary line may be pulled into a block, but no block doubles the
    # requested budget.
    assert max(segment.tokens for segment in segments) < 500


def test_jsx_block_is_labelled_by_tag():
    source = (
        b'export function View() {\n  return (\n    <div className="root">\n'
        + b"".join(b"      <span key={%d}>item %d</span>\n" % (i, i) for i in range(300))
        + b"    </div>\n  );\n}\n"
    )

    segments = segment_symbol(source, "tsx", 0, len(source.rstrip()))

    assert len(segments) > 1
    assert any(segment.label.startswith("<") for segment in segments)


def test_unknown_language_uses_line_window_fallback():
    source = ("\n".join(f"line {index}" for index in range(200)) + "\n").encode()
    line_count = len(source.decode("utf-8").splitlines())

    segments = segment_symbol(source, "cobol", 0, len(source.rstrip()))

    assert len(segments) == 4
    assert segments[0].start_line == 1
    assert segments[0].end_line == 60
    assert segments[1].start_line == 61
    assert segments[-1].end_line == line_count


def test_fallback_still_respects_a_byte_subrange():
    source = ("\n".join(f"line {index}" for index in range(200)) + "\n").encode()
    lines = source.splitlines(keepends=True)
    start = sum(len(line) for line in lines[:119])
    end = sum(len(line) for line in lines[:180])

    segments = segment_symbol(source, "cobol", start, end)

    assert segments[0].start_line == 120
    assert segments[-1].end_line == 180
    assert len(segments) == 2


def test_segment_symbol_never_raises():
    assert isinstance(segment_symbol(b"", "python", 0, 0), list)
    assert isinstance(segment_symbol(b"abc", "python", -10, 10**9), list)
    assert isinstance(segment_symbol(b"\xff\xfe" * 200, "tsx", 0, 300), list)
    assert isinstance(segment_symbol(b"not source" * 100, "rust", 5, 6), list)


def test_min_token_merge_does_not_leave_tiny_blocks():
    source = _python_symbol(400)

    segments = segment_symbol(
        source, "python", 0, len(source.rstrip()), max_tokens=SEGMENT_MAX_TOKENS
    )

    # Only the final remainder may be small; everything else is addressable.
    assert sum(1 for segment in segments if segment.tokens < SEGMENT_MIN_TOKENS) <= 1
