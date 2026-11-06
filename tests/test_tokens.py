"""Unit tests for the token estimator and the MCP response envelope model."""

from __future__ import annotations

from code_mcp import tokens


def test_estimate_is_zero_for_empty_text():
    assert tokens.estimate("") == 0


def test_estimate_never_returns_zero_for_text():
    assert tokens.estimate("a") == 1


def test_estimate_charges_non_ascii_more_than_ascii():
    assert tokens.estimate("中文" * 20) > tokens.estimate("ab" * 20)


def test_estimate_scales_with_length():
    assert tokens.estimate("x" * 380) > tokens.estimate("x" * 190)
    assert tokens.estimate("x" * 38) == 10


def test_envelope_is_free_without_slices():
    assert tokens.envelope_tokens([]) == 0


def test_envelope_grows_with_each_slice():
    one = tokens.envelope_tokens([("app/service.py", "load")])
    two = tokens.envelope_tokens(
        [("app/service.py", "load"), ("app/util.py", "parse")]
    )
    assert one >= tokens.ENVELOPE_BASE + tokens.ENVELOPE_PER_SLICE
    assert two > one


def test_envelope_charges_metadata_names():
    short = tokens.envelope_tokens([("a.py", "f")])
    long = tokens.envelope_tokens(
        [("src/components/tavern/TavernView.tsx", "TavernView")]
    )
    assert long > short


def test_envelope_matches_the_fitted_formula():
    pairs = [("app/service.py", "load"), ("app/util.py", "parse")]
    meta = sum(
        tokens.estimate(file) + tokens.estimate(symbol) for file, symbol in pairs
    )
    expected = int(
        tokens.ENVELOPE_BASE
        + tokens.ENVELOPE_PER_SLICE * len(pairs)
        + meta * tokens.ENVELOPE_META_RATIO
    )
    assert tokens.envelope_tokens(pairs) == expected


def test_savings_pct_reports_negative_growth():
    assert tokens.savings_pct(120, 100) == -20.0
    assert tokens.savings_pct(50, 100) == 50.0
    assert tokens.savings_pct(0, 0) == 0.0


def test_worth_using_enforces_the_margin_and_non_empty_pack():
    assert tokens.worth_using(70, 100) is True
    assert tokens.worth_using(80, 100) is False
    assert tokens.worth_using(0, 100) is False
    assert tokens.worth_using(10, 0) is False
