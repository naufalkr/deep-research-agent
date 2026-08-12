from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from pathlib import Path

from ..config import Settings, settings as default_settings
from ..llm import LLMClient
from ..state import Contradiction, Finding, SourceIndex, render_findings
from ..tools.web import WebSearch
from .subagents import parse_json

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
CRITIC_PROMPT = (PROMPTS / "critic.md").read_text(encoding="utf-8")
CITER_PROMPT = (PROMPTS / "citer.md").read_text(encoding="utf-8")

MAX_RECHECKS = 3
RECHECK_RESULTS = 3


class Critic:
    """Checks findings before they become a report.

    Runs on a clean context with no memory of gathering them - an agent asked
    to review its own work talks itself into keeping it.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        web: WebSearch | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.llm = llm or LLMClient(settings=self.settings)
        self.web = web

    async def run(
        self, findings: list[Finding], index: SourceIndex
    ) -> tuple[list[Finding], list[Contradiction]]:
        if not findings:
            return findings, []

        payload = await self._judge(findings, index)
        checked = self._apply(findings, payload)
        contradictions = self._contradictions(findings, payload)
        await self._recheck(payload)
        return checked, contradictions

    async def _judge(self, findings: list[Finding], index: SourceIndex) -> dict:
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
            return parse_json(reply)
        except (ValueError, json.JSONDecodeError):
            return {}

    def _apply(self, findings: list[Finding], payload: dict) -> list[Finding]:
        by_index: dict[int, dict] = {}
        for raw in payload.get("verdicts", []):
            try:
                by_index[int(raw["n"])] = raw
            except (KeyError, TypeError, ValueError):
                continue

        out = []
        for i, finding in enumerate(findings, 1):
            raw = by_index.get(i)
            verdict = (raw or {}).get("verdict")
            if verdict not in ("supported", "unsupported", "contradicted"):
                out.append(finding)
                continue
            # An unsupported claim is kept but demoted, so the report hedges it
            # instead of dropping a fact the reader may still need.
            confidence = "low" if verdict != "supported" else finding.confidence
            note = (raw or {}).get("note", "")
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

    async def _recheck(self, payload: dict) -> None:
        """Warm the cache for claims the critic flagged, so the next round's
        subagents find corroboration already fetched."""
        if self.web is None:
            return
        queries = [q for q in payload.get("recheck", []) if isinstance(q, str) and q.strip()]
        if not queries:
            return
        await asyncio.gather(
            *(
                asyncio.to_thread(self.web.search, q, num_results=RECHECK_RESULTS)
                for q in queries[:MAX_RECHECKS]
            ),
            return_exceptions=True,
        )


# A sentence ends at .!? followed by space and a capital or quote. Keeps
# "4.62%" and "U.S." in one piece, which splitting on ". " alone does not.
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=["“(]?[A-Z0-9])')


def split_sentences(report: str) -> list[list[str]]:
    """The report as lines, each split into sentences.

    Keeping the line structure means the report rejoins exactly as written -
    headings, blank lines and bullets survive the round trip.
    """
    return [
        _SENTENCE_END.split(line) if line.strip() else [line]
        for line in report.split("\n")
    ]


class Citer:
    """Attaches citations to a finished report.

    Returns a sentence-to-finding map rather than a rewritten report. Asking a
    reasoning model to reproduce the whole report while deciding every citation
    burned its entire token budget on thinking and returned nothing; a map is
    small, and it also makes "do not touch the prose" structural instead of a
    request the model may ignore.
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
        numbered = "\n".join(
            f"{i}. {s}" for i, s in enumerate(flat, 1) if s.strip()
        )
        reply = await self.llm.acomplete(
            f"Sentences:\n\n{numbered}\n\nFindings:\n\n{render_findings(findings, index)}",
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
        return self._apply(lines, payload, len(flat), len(findings))

    def _apply(
        self, lines: list[list[str]], payload: dict, n_sentences: int, n_findings: int
    ) -> str:
        marks = self._marks(payload, n_sentences, n_findings)
        out, i = [], 0
        for line in lines:
            cited = []
            for sentence in line:
                i += 1
                tags = marks.get(i)
                if tags and sentence.strip():
                    sentence = sentence.rstrip() + " " + "".join(f"[{n}]" for n in tags)
                cited.append(sentence)
            out.append(" ".join(cited))
        return "\n".join(out)

    @staticmethod
    def _marks(payload: dict, n_sentences: int, n_findings: int) -> dict[int, list[int]]:
        marks: dict[int, list[int]] = {}
        for raw in payload.get("citations", []):
            try:
                s = int(raw["s"])
            except (KeyError, TypeError, ValueError):
                continue
            numbers = [
                n for n in raw.get("n", []) if isinstance(n, int) and 1 <= n <= n_findings
            ]
            if numbers and 1 <= s <= n_sentences:
                marks[s] = sorted(dict.fromkeys(numbers))
        return marks
