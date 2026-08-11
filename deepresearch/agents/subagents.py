from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from ..config import Settings, settings as default_settings
from ..llm import LLMClient
from ..state import Finding, SourceIndex, SubTask
from ..tools.database import DatabaseTool, DatabaseUnavailable
from ..tools.web import WebSearch

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
RESEARCHER_PROMPT = (PROMPTS / "researcher.md").read_text(encoding="utf-8")
ANALYST_PROMPT = (PROMPTS / "analyst.md").read_text(encoding="utf-8")

MAX_QUERIES = 3
RESULTS_PER_QUERY = 5
# Reasoning models bill thinking against max_tokens before writing a character,
# so even the short planning replies need far more headroom than the JSON itself.
PLAN_TOKENS = 6000
EXTRACT_TOKENS = 12000


def parse_json(text: str) -> dict:
    """Pull one JSON object out of a reply, tolerating fences and stray prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in reply: {text[:200]}")
    return json.loads(text[start : end + 1])


def _findings_from(payload: dict, task: SubTask, allowed_urls: set[str]) -> list[Finding]:
    """Keep only findings whose source was actually shown to the subagent."""
    out = []
    for raw in payload.get("findings", []):
        url = (raw.get("source_url") or "").strip()
        if url and url not in allowed_urls:
            continue  # cited something it never saw
        claim = (raw.get("claim") or "").strip()
        if not claim:
            continue
        confidence = raw.get("confidence", "medium")
        out.append(
            Finding(
                claim=claim,
                evidence=(raw.get("evidence") or "").strip(),
                source_url=url,
                confidence=confidence if confidence in ("high", "medium", "low") else "medium",
                subtask_id=task.id,
            )
        )
    return out


class Researcher:
    """Web subagent. Two LLM calls: plan queries, then extract findings.

    A fixed shape rather than a loop keeps cost per subagent predictable; when
    findings come back thin, the lead spawns another round instead.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        web: WebSearch | None = None,
        index: SourceIndex | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.llm = llm or LLMClient(settings=self.settings)
        self.web = web or WebSearch(self.settings)
        self.index = index or SourceIndex()

    async def run(self, task: SubTask) -> list[Finding]:
        queries = await self._plan(task)
        results = await self._search_all(queries)
        if not results:
            return []
        return await self._extract(task, results)

    async def _plan(self, task: SubTask) -> list[str]:
        reply = await self.llm.acomplete(
            task.brief(),
            role="subagent",
            system=RESEARCHER_PROMPT,
            agent=f"researcher:{task.id}",
            step="plan_queries",
            max_tokens=PLAN_TOKENS,
        )
        try:
            queries = parse_json(reply).get("queries", [])
        except (ValueError, json.JSONDecodeError):
            queries = [task.objective]
        return [q for q in queries if isinstance(q, str) and q.strip()][:MAX_QUERIES] or [
            task.objective
        ]

    async def _search_all(self, queries: list[str]) -> list:
        # Searches inside one subagent run concurrently too, not just across them.
        batches = await asyncio.gather(
            *(
                asyncio.to_thread(self.web.search, q, num_results=RESULTS_PER_QUERY)
                for q in queries
            )
        )
        return [r for batch in batches for r in batch]

    async def _extract(self, task: SubTask, results: list) -> list[Finding]:
        for r in results:
            self.index.add(r.title, r.url)
        body = "\n\n".join(f"URL: {r.url}\nTITLE: {r.title}\n{r.snippet}" for r in results)
        reply = await self.llm.acomplete(
            f"{task.brief()}\n\nSearch results:\n\n{body}",
            role="subagent",
            system=RESEARCHER_PROMPT,
            agent=f"researcher:{task.id}",
            step="extract",
            max_tokens=EXTRACT_TOKENS,
        )
        try:
            payload = parse_json(reply)
        except (ValueError, json.JSONDecodeError):
            return []
        return _findings_from(payload, task, {r.url for r in results})


class Analyst:
    """Database subagent. Two LLM calls: decide the query, then record it."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        database: DatabaseTool | None = None,
        index: SourceIndex | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.llm = llm or LLMClient(settings=self.settings)
        self.database = database or DatabaseTool(self.settings)
        self.index = index or SourceIndex()

    async def run(self, task: SubTask) -> list[Finding]:
        request = await self._decide(task)
        try:
            answer = (
                await self.database.query(request["question"], request["database"])
            ).answer
        except DatabaseUnavailable:
            return []

        self.index.add(f"internal database ({request['database']})", "")
        return await self._record(task, request, answer)

    async def _decide(self, task: SubTask) -> dict:
        reply = await self.llm.acomplete(
            task.brief(),
            role="subagent",
            system=ANALYST_PROMPT,
            agent=f"analyst:{task.id}",
            step="plan_query",
            max_tokens=PLAN_TOKENS,
        )
        try:
            payload = parse_json(reply)
        except (ValueError, json.JSONDecodeError):
            payload = {}
        return {
            "question": payload.get("question") or task.objective,
            "database": payload.get("database") or "chinook",
        }

    async def _record(self, task: SubTask, request: dict, answer: str) -> list[Finding]:
        reply = await self.llm.acomplete(
            f"{task.brief()}\n\nYou asked: {request['question']}\n\nThe database replied:\n{answer}",
            role="subagent",
            system=ANALYST_PROMPT,
            agent=f"analyst:{task.id}",
            step="record",
            max_tokens=EXTRACT_TOKENS,
        )
        try:
            payload = parse_json(reply)
        except (ValueError, json.JSONDecodeError):
            return [
                Finding(
                    claim=answer.split("\n")[0],
                    evidence=answer,
                    source_url="",
                    confidence="high",
                    subtask_id=task.id,
                )
            ]
        return _findings_from(payload, task, set())
