from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, settings as default_settings
from ..llm import LLMClient
from ..tools.database import DatabaseTool, DatabaseUnavailable
from ..tools.web import SearchResult, WebSearch

PROMPT = (Path(__file__).resolve().parent.parent / "prompts" / "baseline.md").read_text(
    encoding="utf-8"
)


@dataclass(frozen=True)
class Source:
    n: int
    title: str
    url: str

    def __str__(self) -> str:
        return f"[{self.n}] {self.title} - {self.url}"


@dataclass
class Step:
    action: str
    detail: str
    result: str


@dataclass
class ResearchResult:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    ok: bool = True

    def report(self) -> str:
        out = [self.answer]
        if self.sources:
            out += ["", "Sources:"] + [str(s) for s in self.sources]
        return "\n".join(out)


def _parse_action(text: str) -> dict:
    """Pull the JSON action out of a reply, tolerating fences and stray prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in reply: {text[:200]}")
    return json.loads(text[start : end + 1])


class BaselineAgent:
    """One agent, one loop, two tools - the control group for v0.1.

    Deliberately has no planner, no subagents, no separate critic or citation
    pass. Later phases are measured against what this scores.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        web: WebSearch | None = None,
        database: DatabaseTool | None = None,
        max_steps: int = 8,
    ) -> None:
        self.settings = settings or default_settings
        self.llm = llm or LLMClient(settings=self.settings)
        self.web = web or WebSearch(self.settings)
        self.database = database or DatabaseTool(self.settings)
        self.max_steps = max_steps

    async def run(self, question: str) -> ResearchResult:
        result = ResearchResult(question=question, answer="")
        observations: list[str] = []

        for _ in range(self.max_steps):
            try:
                action = _parse_action(await self._think(question, observations))
            except (ValueError, json.JSONDecodeError) as exc:
                observations.append(f"Your last reply was not valid JSON ({exc}).")
                continue

            kind = action.get("action")
            if kind == "answer":
                result.answer = action.get("text", "").strip()
                return result
            if kind == "search":
                observations.append(self._do_search(action, result))
            elif kind == "query_database":
                observations.append(await self._do_query(action, result))
            else:
                observations.append(f"Unknown action {kind!r}.")

        result.answer = await self._force_answer(question, observations)
        result.ok = False  # ran out of steps rather than deciding it was done
        return result

    async def _think(self, question: str, observations: list[str]) -> str:
        return await self.llm.acomplete(
            self._context(question, observations),
            role="lead",
            system=PROMPT,
            agent="baseline",
            step="think",
        )

    async def _force_answer(self, question: str, observations: list[str]) -> str:
        return await self.llm.acomplete(
            self._context(question, observations)
            + "\n\nStep budget reached. Answer now with what you have, and say "
            "which parts remain unsupported.",
            role="lead",
            system=PROMPT,
            agent="baseline",
            step="force_answer",
        )

    def _context(self, question: str, observations: list[str]) -> str:
        parts = [f"Question: {question}"]
        if observations:
            parts.append("\nWhat you have gathered so far:\n" + "\n\n".join(observations))
        return "\n".join(parts)

    def _do_search(self, action: dict, result: ResearchResult) -> str:
        query = action.get("query", "")
        results = self.web.search(query, num_results=5)
        if not results:
            result.steps.append(Step("search", query, "no new results"))
            return f"Search {query!r} returned nothing new - every hit was already read."

        lines = []
        for r in results:
            source = self._add_source(result, r.title, r.url)
            lines.append(f"[{source.n}] {r.title}\n{r.snippet}")
        observation = f"Search {query!r} returned:\n" + "\n\n".join(lines)
        result.steps.append(Step("search", query, f"{len(results)} results"))
        return observation

    async def _do_query(self, action: dict, result: ResearchResult) -> str:
        question = action.get("question", "")
        database = action.get("database", "chinook")
        try:
            answer = (await self.database.query(question, database)).answer
        except DatabaseUnavailable as exc:
            result.steps.append(Step("query_database", question, f"unavailable: {exc}"))
            return f"The database is unavailable ({exc}). Continue with web sources."

        source = self._add_source(result, f"internal database ({database})", "")
        result.steps.append(Step("query_database", question, "ok"))
        return f"[{source.n}] Database answer to {question!r}:\n{answer}"

    def _add_source(self, result: ResearchResult, title: str, url: str) -> Source:
        for existing in result.sources:
            if url and existing.url == url:
                return existing
        source = Source(len(result.sources) + 1, title, url)
        result.sources.append(source)
        return source


async def ask(question: str, **kwargs) -> ResearchResult:
    return await BaselineAgent(**kwargs).run(question)
