from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings, settings as default_settings
from ..llm import LLMClient, parse_json
from ..state import Finding, SourceIndex, SubTask, database_url
from ..tools.database import DatabaseTool, DatabaseUnavailable
from ..tools.web import CATEGORIES, WebSearch

PROMPT = (Path(__file__).resolve().parent.parent / "prompts" / "child.md").read_text(
    encoding="utf-8"
)

MAX_CALLS = 3
RESULTS_PER_SEARCH = 5
MAX_OPPONENT_CLAIMS = 8
# Reasoning models bill thinking against max_tokens before writing anything.
PLAN_TOKENS = 6000
EXTRACT_TOKENS = 12000


@dataclass(frozen=True)
class Evidence:
    title: str
    url: str
    text: str


class Child:
    """One research agent. Picks its own tools, runs them in parallel, and
    returns findings tied to the sources it was actually shown.

    The lead gives it a side - neutral, pro or con. In a rebuttal round it
    also gets the other side's findings to answer.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        web: WebSearch | None = None,
        database: DatabaseTool | None = None,
        index: SourceIndex | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.llm = llm or LLMClient(settings=self.settings)
        self.web = web or WebSearch(self.settings)
        self.database = database or DatabaseTool(self.settings)
        self.index = index or SourceIndex()

    async def run(self, task: SubTask, opponent: list[Finding] | None = None) -> list[Finding]:
        brief = self._brief(task, opponent)
        calls = await self._plan(task, brief)
        evidence = await self._gather(calls)
        if not evidence:
            return []
        return await self._extract(task, brief, evidence, rebuttal=opponent is not None)

    def _brief(self, task: SubTask, opponent: list[Finding] | None) -> str:
        brief = task.brief()
        if opponent:
            claims = "\n".join(f"- {f.claim}" for f in opponent[:MAX_OPPONENT_CLAIMS])
            brief += f"\n\nRebuttal round. The other side found:\n{claims}"
        return brief

    async def _plan(self, task: SubTask, brief: str) -> list[dict]:
        reply = await self.llm.acomplete(
            brief,
            role="subagent",
            system=PROMPT,
            agent=f"child:{task.id}",
            step="plan_calls",
            max_tokens=PLAN_TOKENS,
        )
        try:
            raw = parse_json(reply).get("calls", [])
        except (ValueError, json.JSONDecodeError):
            raw = []
        calls = [
            c
            for c in raw
            if isinstance(c, dict)
            and c.get("tool") in (*CATEGORIES, "database")
            and str(c.get("query", "")).strip()
        ]
        return calls[:MAX_CALLS] or [{"tool": "web", "query": task.objective}]

    async def _gather(self, calls: list[dict]) -> list[Evidence]:
        batches = await asyncio.gather(*(self._call(c) for c in calls))
        return [e for batch in batches for e in batch]

    async def _call(self, call: dict) -> list[Evidence]:
        query = call["query"].strip()
        if call["tool"] == "database":
            name = str(call.get("database") or "chinook")
            try:
                answer = await self.database.query(query, name)
            except DatabaseUnavailable:
                return []
            return [Evidence(f"internal database ({name})", database_url(name), f"Asked: {query}\n{answer}")]

        results = await asyncio.to_thread(
            self.web.search, query, num_results=RESULTS_PER_SEARCH, tool=call["tool"]
        )
        return [Evidence(r.title, r.url, r.snippet) for r in results]

    async def _extract(
        self, task: SubTask, brief: str, evidence: list[Evidence], *, rebuttal: bool
    ) -> list[Finding]:
        for e in evidence:
            self.index.add(e.title, e.url)
        body = "\n\n".join(f"URL: {e.url}\nTITLE: {e.title}\n{e.text}" for e in evidence)
        raw = None
        for _ in range(2):  # one retry: a malformed reply would otherwise lose the whole search
            reply = await self.llm.acomplete(
                f"{brief}\n\nResults:\n\n{body}",
                role="subagent",
                system=PROMPT,
                agent=f"child:{task.id}",
                step="extract",
                max_tokens=EXTRACT_TOKENS,
            )
            try:
                raw = parse_json(reply).get("findings", [])
                break
            except (ValueError, json.JSONDecodeError):
                continue

        if raw is None:
            # a real database figure is worth keeping even if the write-up failed
            return [
                Finding(e.text.split("\n")[1], e.text, e.url, "high", task.id, task.stance, rebuttal)
                for e in evidence
                if e.url.startswith("db://")
            ]

        allowed = {e.url for e in evidence}
        out = []
        for f in raw:
            if not isinstance(f, dict):
                continue
            claim = str(f.get("claim") or "").strip()
            url = str(f.get("source_url") or "").strip()
            if not claim or url not in allowed:
                continue  # cited something it was never shown
            confidence = f.get("confidence")
            out.append(
                Finding(
                    claim=claim,
                    evidence=str(f.get("evidence") or "").strip(),
                    source_url=url,
                    confidence=confidence if confidence in ("high", "medium", "low") else "medium",
                    subtask_id=task.id,
                    stance=task.stance,
                    rebuttal=rebuttal,
                )
            )
        return out
