from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

Tool = Literal["web", "database"]


@dataclass(frozen=True)
class SubTask:
    """One slice of the research, handed to one subagent."""

    id: str
    objective: str
    tool: Tool
    boundaries: str = ""

    def brief(self) -> str:
        lines = [f"Objective: {self.objective}"]
        if self.boundaries:
            lines.append(f"Not your job: {self.boundaries}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Finding:
    claim: str
    evidence: str
    source_url: str
    confidence: Literal["high", "medium", "low"] = "medium"
    subtask_id: str = ""

    def render(self, n: int | None = None) -> str:
        tag = f"[{n}] " if n else ""
        return f"{tag}{self.claim}\n    evidence: {self.evidence}\n    confidence: {self.confidence}"


@dataclass(frozen=True)
class Source:
    n: int
    title: str
    url: str

    def __str__(self) -> str:
        return f"[{self.n}] {self.title} - {self.url}" if self.url else f"[{self.n}] {self.title}"


@dataclass
class SourceIndex:
    """Assigns each URL one citation number, shared across all subagents."""

    sources: list[Source] = field(default_factory=list)

    def add(self, title: str, url: str) -> Source:
        for existing in self.sources:
            if url and existing.url == url:
                return existing
        source = Source(len(self.sources) + 1, title, url)
        self.sources.append(source)
        return source

    def number_for(self, url: str) -> int | None:
        return next((s.n for s in self.sources if s.url == url), None)


class ResearchState(TypedDict, total=False):
    """Graph state. `findings` and `sources` use add reducers because parallel
    subagents write to them at the same time; the rest is written by one node.
    """

    question: str
    complexity: str
    plan: list[SubTask]
    findings: Annotated[list[Finding], operator.add]
    sources: Annotated[list[Source], operator.add]
    round: int
    answer: str
    ok: bool


def render_findings(findings: list[Finding], index: SourceIndex) -> str:
    """Lay out findings for the lead to synthesise from."""
    if not findings:
        return "No findings were gathered."
    lines = []
    for f in findings:
        lines.append(f.render(index.number_for(f.source_url)))
    return "\n\n".join(lines)
