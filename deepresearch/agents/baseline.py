from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, settings as default_settings
from ..llm import LLMClient, parse_json
from ..state import Source, SourceIndex, database_url
from ..tools.database import DatabaseTool, DatabaseUnavailable
from ..tools.web import CATEGORIES, WebSearch

PROMPT = (Path(__file__).resolve().parent.parent / "prompts" / "baseline.md").read_text(
    encoding="utf-8"
)


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


class BaselineAgent:
    """One agent, one loop, the same four tools the children have.

    The control group: no lead, no parallel children, no checking or citation
    pass. The multi-agent system is measured against this.
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
        index = SourceIndex()
        observations: list[str] = []

        for _ in range(self.max_steps):
            try:
                action = parse_json(await self._think(question, observations))
            except (ValueError, json.JSONDecodeError) as exc:
                observations.append(f"Your last reply was not valid JSON ({exc}).")
                continue

            kind = action.get("action")
            if kind == "answer":
                result.answer = str(action.get("text", "")).strip()
                break
            if kind == "search":
                observations.append(self._search(action, result, index))
            elif kind == "query_database":
                observations.append(await self._query(action, result, index))
            else:
                observations.append(f"Unknown action {kind!r}.")
        else:
            result.answer = await self._think(
                question,
                observations,
                nudge="Step budget reached. Answer now with what you have, and say "
                "which parts remain unsupported.",
            )
            result.ok = False  # ran out of steps rather than deciding it was done

        result.sources = index.sources
        return result

    async def _think(self, question: str, observations: list[str], nudge: str = "") -> str:
        prompt = f"Question: {question}"
        if observations:
            prompt += "\n\nWhat you have gathered so far:\n" + "\n\n".join(observations)
        if nudge:
            prompt += f"\n\n{nudge}"
        return await self.llm.acomplete(
            prompt,
            role="lead",
            system=PROMPT,
            agent="baseline",
            step="force_answer" if nudge else "think",
        )

    def _search(self, action: dict, result: ResearchResult, index: SourceIndex) -> str:
        query = str(action.get("query", ""))
        tool = action.get("tool", "web")
        if tool not in CATEGORIES:
            tool = "web"
        results = self.web.search(query, num_results=5, tool=tool)
        result.steps.append(Step(tool, query, f"{len(results)} results"))
        if not results:
            return f"Search {query!r} returned nothing new - every hit was already read."
        lines = [f"[{index.add(r.title, r.url).n}] {r.title}\n{r.snippet}" for r in results]
        return f"{tool.capitalize()} search {query!r} returned:\n" + "\n\n".join(lines)

    async def _query(self, action: dict, result: ResearchResult, index: SourceIndex) -> str:
        question = str(action.get("question", ""))
        database = str(action.get("database") or "chinook")
        try:
            answer = await self.database.query(question, database)
        except DatabaseUnavailable as exc:
            result.steps.append(Step("database", question, f"unavailable: {exc}"))
            return f"The database is unavailable ({exc}). Continue with the other tools."
        n = index.add(f"internal database ({database})", database_url(database)).n
        result.steps.append(Step("database", question, "ok"))
        return f"[{n}] Database answer to {question!r}:\n{answer}"
