from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from pathlib import Path

from ..config import Settings, settings as default_settings
from ..llm import LLMClient, parse_json
from ..state import Contradiction, Finding, SourceIndex

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
CRITIC_PROMPT = (PROMPTS / "critic.md").read_text(encoding="utf-8")
CITER_PROMPT = (PROMPTS / "citer.md").read_text(encoding="utf-8")

BATCH = 20


class Critic:
    """Checks each finding against its own evidence, on a clean context - an
    agent asked to review its own work talks itself into keeping it.

    Findings are checked in parallel batches: one call over 80 findings took
    13 minutes. A child's findings sit next to each other, so claims about the
    same thing mostly share a batch and contradictions are still caught.
    """

    def __init__(
        self, settings: Settings | None = None, *, llm: LLMClient | None = None
    ) -> None:
        self.settings = settings or default_settings
        self.llm = llm or LLMClient(settings=self.settings)

    async def run(
        self, findings: list[Finding], index: SourceIndex
    ) -> tuple[list[Finding], list[Contradiction]]:
        batches = [findings[i : i + BATCH] for i in range(0, len(findings), BATCH)]
        results = await asyncio.gather(*(self._check(b, index) for b in batches))
        checked = [f for fs, _ in results for f in fs]
        return checked, [c for _, cs in results for c in cs]

    async def _check(
        self, findings: list[Finding], index: SourceIndex
    ) -> tuple[list[Finding], list[Contradiction]]:
        numbered = "\n\n".join(
            f"{i}. {f.render(index.number_for(f.source_url))}"
            for i, f in enumerate(findings, 1)
        )
        reply = await self.llm.acomplete(
            f"Findings:\n\n{numbered}",
            role="critic",
            system=CRITIC_PROMPT,
            agent="critic",
            step="verify",
            max_tokens=12000,
        )
        try:
            payload = parse_json(reply)
        except (ValueError, json.JSONDecodeError):
            payload = {}
        return self._apply(findings, payload), self._contradictions(findings, payload)

    def _apply(self, findings: list[Finding], payload: dict) -> list[Finding]:
        by_index: dict[int, dict] = {}
        for raw in payload.get("verdicts", []):
            try:
                by_index[int(raw["n"])] = raw
            except (KeyError, TypeError, ValueError):
                continue

        out = []
        for i, finding in enumerate(findings, 1):
            raw = by_index.get(i, {})
            verdict = raw.get("verdict")
            if verdict not in ("supported", "unsupported", "contradicted"):
                out.append(finding)
                continue
            # kept but demoted, so the report hedges it instead of losing it
            confidence = finding.confidence if verdict == "supported" else "low"
            note = raw.get("note", "")
            evidence = f"{finding.evidence} [critic: {note}]" if note else finding.evidence
            out.append(replace(finding, verdict=verdict, confidence=confidence, evidence=evidence))
        return out

    def _contradictions(self, findings: list[Finding], payload: dict) -> list[Contradiction]:
        out = []
        for raw in payload.get("contradictions", []):
            numbers = raw.get("n") or []
            claims = [
                findings[n - 1].claim
                for n in numbers
                if isinstance(n, int) and 1 <= n <= len(findings)
            ]
            if len(claims) >= 2:
                out.append(Contradiction(claims=claims, note=str(raw.get("note", ""))))
        return out


# A sentence ends at .!? followed by space and a capital or quote. Decimals
# ("4.62%") and abbreviations ("U.S. Bank", "e.g. Exa") do not end one.
_SENTENCE_END = re.compile(
    r'(?<=[.!?])(?<!\b[A-Z]\.[A-Z]\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)'
    r'(?<!\bMr\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bNo\.)\s+(?=["“(]?[A-Z0-9])'
)


def split_sentences(report: str) -> list[list[str]]:
    """The report as lines, each split into sentences, so headings and bullets
    survive the round trip."""
    return [
        _SENTENCE_END.split(line) if line.strip() else [line]
        for line in report.split("\n")
    ]


class Citer:
    """Attaches citations to a finished report.

    The model returns a sentence-to-finding map, not a rewritten report, so it
    cannot change the prose. Findings are numbered by position; the marker
    written is the finding's source number.
    """

    def __init__(
        self, settings: Settings | None = None, *, llm: LLMClient | None = None
    ) -> None:
        self.settings = settings or default_settings
        self.llm = llm or LLMClient(settings=self.settings)

    async def run(self, report: str, findings: list[Finding], index: SourceIndex) -> str:
        if not report.strip() or not findings:
            return report

        lines = split_sentences(report)
        flat = [s for line in lines for s in line]
        sentences = "\n".join(f"{i}. {s}" for i, s in enumerate(flat, 1) if s.strip())
        listed = "\n\n".join(f"{i}. {f.render()}" for i, f in enumerate(findings, 1))
        reply = await self.llm.acomplete(
            f"Sentences:\n\n{sentences}\n\nFindings:\n\n{listed}",
            role="citation",
            system=CITER_PROMPT,
            agent="citer",
            step="cite",
            max_tokens=8000,
        )
        try:
            payload = parse_json(reply)
        except (ValueError, json.JSONDecodeError):
            return report

        source_of = [index.number_for(f.source_url) for f in findings]
        marks = self._marks(payload, len(flat), source_of)
        out, i = [], 0
        for line in lines:
            cited = []
            for sentence in line:
                i += 1
                if marks.get(i) and sentence.strip():
                    sentence = sentence.rstrip() + " " + "".join(f"[{n}]" for n in marks[i])
                cited.append(sentence)
            out.append(" ".join(cited))
        return "\n".join(out)

    @staticmethod
    def _marks(
        payload: dict, n_sentences: int, source_of: list[int | None]
    ) -> dict[int, list[int]]:
        marks: dict[int, list[int]] = {}
        for raw in payload.get("citations", []):
            try:
                s = int(raw["s"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 1 <= s <= n_sentences:
                continue
            sources = {
                source_of[n - 1]
                for n in raw.get("n", [])
                if isinstance(n, int) and 1 <= n <= len(source_of) and source_of[n - 1]
            }
            if sources:
                marks[s] = sorted(sources)
        return marks
