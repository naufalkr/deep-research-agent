from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Mode = Literal["facts", "debate"]
Stance = Literal["neutral", "pro", "con"]
Tool = Literal["web", "news", "papers", "database"]
Verdict = Literal["unchecked", "supported", "unsupported", "contradicted"]

SIDE_LABEL = {"pro": "for", "con": "against"}


@dataclass(frozen=True)
class SubTask:
    """One slice of the research, handed to one child agent."""

    id: str
    objective: str
    stance: Stance = "neutral"
    boundaries: str = ""

    def brief(self) -> str:
        lines = [f"Objective: {self.objective}"]
        if self.stance == "pro":
            lines.append("Your side: FOR. Find the strongest evidence that supports this position.")
        elif self.stance == "con":
            lines.append(
                "Your side: AGAINST. Find the strongest evidence that this position is "
                "wrong, risky or overstated."
            )
        if self.boundaries:
            lines.append(f"Not your job: {self.boundaries}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Plan:
    mode: Mode
    tasks: list[SubTask]
    complexity: str = "moderate"
    motion: str = ""  # the position argued over, debate mode only


@dataclass(frozen=True)
class Finding:
    claim: str
    evidence: str
    source_url: str
    confidence: Literal["high", "medium", "low"] = "medium"
    subtask_id: str = ""
    stance: Stance = "neutral"
    rebuttal: bool = False
    verdict: Verdict = "unchecked"

    def render(self, n: int | None = None) -> str:
        tag = f"[{n}] " if n else ""
        line = f"{tag}{self.claim}\n    evidence: {self.evidence}\n    confidence: {self.confidence}"
        if self.stance != "neutral":
            side = SIDE_LABEL[self.stance] + (" (rebuttal)" if self.rebuttal else "")
            line += f"\n    side: {side}"
        if self.verdict != "unchecked":
            line += f"\n    verdict: {self.verdict}"
        return line


@dataclass(frozen=True)
class Contradiction:
    """Two findings that cannot both be right. Reported, never silently resolved."""

    claims: list[str]
    note: str

    def render(self) -> str:
        return f"{self.note}\n" + "\n".join(f"    - {c}" for c in self.claims)


def database_url(database: str) -> str:
    """A database has no link but still needs its own citation number."""
    return f"db://{database}"


@dataclass(frozen=True)
class Source:
    n: int
    title: str
    url: str

    def __str__(self) -> str:
        if not self.url or self.url.startswith("db://"):
            return f"[{self.n}] {self.title}"
        return f"[{self.n}] {self.title} - {self.url}"


@dataclass
class SourceIndex:
    """Assigns each URL one citation number, shared across all child agents."""

    sources: list[Source] = field(default_factory=list)

    def add(self, title: str, url: str) -> Source:
        for existing in self.sources:
            if url and existing.url == url:
                return existing
        source = Source(len(self.sources) + 1, title, url)
        self.sources.append(source)
        return source

    def number_for(self, url: str) -> int | None:
        return next((s.n for s in self.sources if url and s.url == url), None)


def render_findings(findings: list[Finding], index: SourceIndex) -> str:
    if not findings:
        return "No findings were gathered."
    return "\n\n".join(f.render(index.number_for(f.source_url)) for f in findings)


_MARKERS = re.compile(r"(?:\[\d+\])+")


def compact_citations(report: str, sources: list[Source]) -> tuple[str, list[Source]]:
    """Keep only the sources the report cites, renumbered in reading order."""
    by_n = {s.n: s for s in sources}
    order: dict[int, int] = {}

    def renumber(match: re.Match) -> str:
        old = [int(n) for n in re.findall(r"\d+", match.group(0))]
        if not all(n in by_n for n in old):
            return match.group(0)
        for n in old:
            order.setdefault(n, len(order) + 1)
        return "".join(f"[{order[n]}]" for n in sorted(set(old), key=order.get))

    text = _MARKERS.sub(renumber, report)
    kept = [Source(new, by_n[old].title, by_n[old].url) for old, new in order.items()]
    return text, kept
