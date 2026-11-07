"""Frozen data model shared by the index, graph, pack, review and edit layers.

Everything below is a plain dataclass so it can cross module boundaries and be
serialised to MCP tool responses without adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Supported parser languages. Mapping to tree-sitter grammars lives in
# index/parsers.py so this module stays dependency-free.
LANGUAGES: tuple[str, ...] = ("python", "typescript", "tsx", "javascript", "rust")

FILE_KIND = "file"
SYMBOL_KINDS: tuple[str, ...] = (
    "function",
    "method",
    "class",
    "interface",
    "struct",
    "enum",
    "trait",
    "type",
    "constant",
    "module",
)

# Reference kinds, ordered from most to least certain.
REF_CALL = "call"
REF_IMPORT = "import"
REF_INHERIT = "inherit"
REF_TYPE = "type_ref"
REF_NAME = "name_match"

CONF_EXACT = "exact"
CONF_NAME = "name_match"
CONF_BRIDGE = "bridge"

SEVERITY_ORDER: tuple[str, ...] = ("P0", "P1", "P2", "P3")


@dataclass(slots=True)
class FileRecord:
    """One indexed source file. ``path`` is repo-relative and forward-slashed."""

    id: int = 0
    path: str = ""
    lang: str = ""
    mtime: float = 0.0
    hash: str = ""
    loc: int = 0


@dataclass(slots=True)
class Symbol:
    """A definition extracted from a file."""

    id: int = 0
    file_id: int = 0
    file: str = ""
    name: str = ""
    qualname: str = ""
    kind: str = ""
    start_line: int = 0
    end_line: int = 0
    start_byte: int = 0
    end_byte: int = 0
    signature: str = ""
    docstring: str = ""


@dataclass(slots=True)
class Ref:
    """A directed edge between two symbols, or from a symbol to an unresolved name."""

    from_symbol_id: int = 0
    to_symbol_id: int | None = None
    to_name: str = ""
    kind: str = REF_CALL
    confidence: str = CONF_NAME


@dataclass(slots=True)
class Segment:
    """One addressable block inside an oversized symbol.

    ``index`` is 1-based and stable for a given source revision, so a caller can
    ask for the same block again by index or by its line range.
    """

    index: int
    start_line: int
    end_line: int
    tokens: int
    label: str


@dataclass(slots=True)
class Slice:
    """A bounded piece of source text handed to the caller."""

    file: str
    start_line: int
    end_line: int
    text: str = ""
    symbol: str = ""
    kind: str = SYMBOL_KINDS[0]
    tokens: int = 0
    # Set only when the symbol was too large to return whole: ``text`` then
    # holds the signature and ``segments`` describes what was left out.
    segments: list[Segment] = field(default_factory=list)
    truncated: bool = False
    total_tokens: int = 0

    def header(self) -> str:
        if self.truncated:
            return (
                f"{self.file}:{self.start_line}-{self.end_line} [{self.kind}] "
                f"(outline, {len(self.segments)} segments, {self.total_tokens} tokens)"
            )
        return f"{self.file}:{self.start_line}-{self.end_line} [{self.kind}]"


@dataclass(slots=True)
class Pack:
    """Result of packing the causal neighbourhood of a target.

    ``token_baseline`` is what reading the whole containing file(s) would cost;
    ``worth_using`` is False when packing saved nothing, so callers can fall
    back to a plain read instead of trusting the tool blindly. ``token_total``
    counts slice bodies only; ``response_tokens`` adds the serialised JSON
    envelope and is what ``worth_using`` is judged on.
    """

    target: Slice | None = None
    callers: list[Slice] = field(default_factory=list)
    callees: list[Slice] = field(default_factory=list)
    imports: list[Slice] = field(default_factory=list)
    tests: list[Slice] = field(default_factory=list)
    token_total: int = 0
    response_tokens: int = 0
    token_baseline: int = 0
    savings_pct: float = 0.0
    worth_using: bool = True
    note: str = ""

    def sections(self) -> dict[str, list[Slice]]:
        return {
            "target": [self.target] if self.target else [],
            "callers": self.callers,
            "callees": self.callees,
            "imports": self.imports,
            "tests": self.tests,
        }


@dataclass(slots=True)
class Finding:
    """One review result, whether from an external linter or a structural rule."""

    rule: str
    severity: str
    file: str
    line: int
    message: str
    symbol: str = ""


@dataclass(slots=True)
class ReviewReport:
    findings: list[Finding] = field(default_factory=list)
    external_ran: list[str] = field(default_factory=list)
    external_missing: list[str] = field(default_factory=list)
    score: float = 0.0

    def by_severity(self) -> dict[str, int]:
        counts = dict.fromkeys(SEVERITY_ORDER, 0)
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return counts


@dataclass(slots=True)
class GateStep:
    """One command executed as part of a project gate."""

    name: str
    command: str
    exit_code: int
    duration_s: float = 0.0
    output: str = ""
    passed: bool = False


@dataclass(slots=True)
class GateResult:
    steps: list[GateStep] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.steps) and all(step.passed for step in self.steps)


@dataclass(slots=True)
class EditResult:
    """Outcome of a surgical edit. ``applied`` is False whenever a guard tripped."""

    file: str
    symbol: str
    start_line: int = 0
    end_line: int = 0
    old_hash: str = ""
    new_hash: str = ""
    lines_added: int = 0
    lines_removed: int = 0
    applied: bool = False
    reason: str = ""
    diff: str = ""
