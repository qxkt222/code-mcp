"""Token estimation used to prove whether packing actually saves anything.

No tokeniser dependency: a character-class heuristic is accurate enough for the
savings ratio, which is all the caller needs to decide between packed context
and a plain file read.
"""

from __future__ import annotations

ASCII_CHARS_PER_TOKEN = 3.8
NON_ASCII_CHARS_PER_TOKEN = 1.4


def estimate(text: str) -> int:
    """Estimate the token count of ``text`` for mixed English/CJK source code."""
    if not text:
        return 0
    non_ascii = sum(1 for char in text if ord(char) > 127)
    ascii_count = len(text) - non_ascii
    tokens = ascii_count / ASCII_CHARS_PER_TOKEN + non_ascii / NON_ASCII_CHARS_PER_TOKEN
    return max(1, int(tokens))


def savings_pct(packed: int, baseline: int) -> float:
    """Percentage saved by packing. Negative means packing was more expensive."""
    if baseline <= 0:
        return 0.0
    return round((baseline - packed) / baseline * 100.0, 1)


def worth_using(packed: int, baseline: int, min_saving_pct: float = 25.0) -> bool:
    """Only recommend the packed result when it beats a plain read by a margin."""
    return packed > 0 and baseline > 0 and savings_pct(packed, baseline) >= min_saving_pct
