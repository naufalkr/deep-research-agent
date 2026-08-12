from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .agents.lead import Lead
from .agents.verify import Citer, Critic
from .agents.subagents import Analyst, Researcher
from .config import Settings, settings as default_settings
from .llm import LLMClient
from .state import Contradiction, Finding, Source, SourceIndex, SubTask
from .tools.database import DatabaseTool
from .tools.web import WebSearch


@dataclass
class MultiAgentResult:
    question: str
    answer: str
    complexity: str = "moderate"
    plan: list[SubTask] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    rounds: int = 1
    ok: bool = True

    def report(self) -> str:
        out = [self.answer]
        if self.sources:
            out += ["", "Sources:"] + [str(s) for s in self.sources]
        return "\n".join(out)


class ResearchGraph:
    """plan -> fan out to subagents in parallel -> gather -> synthesise.

    Subagents share one SourceIndex, so a page cited by two of them gets one
    number. They do *not* share URL dedup: sibling subagents often cover
    adjacent ground, and a shared set lets whichever searches first take every
    result and leave the others with nothing to extract.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        web: WebSearch | None = None,
        database: DatabaseTool | None = None,
        max_subagents: int | None = None,
        verify: bool = True,
    ) -> None:
        self.settings = settings or default_settings
        self.llm = llm or LLMClient(settings=self.settings)
        self.web = web  # injected for tests; otherwise one per subagent
        self.database = database or DatabaseTool(self.settings)
        self.lead = Lead(self.settings, llm=self.llm, max_subagents=max_subagents)
        # Off for the v0.2 comparison, so a phase-3 gain is not mistaken for
        # a phase-2 one.
        self.verify = verify
        self.critic = Critic(self.settings, llm=self.llm, web=self.web)
        self.citer = Citer(self.settings, llm=self.llm)

    async def run(self, question: str) -> MultiAgentResult:
        index = SourceIndex()
        failures: dict[str, str] = {}
        complexity, plan = await self.lead.plan(question)

        findings = await self._fan_out(plan, index, failures)

        # A plan made once and never checked leaves whatever it overlooked
        # missing, however many subagents ran. Each extra round is spent only
        # on parts of the question nothing has answered.
        rounds = 1
        while rounds < self.settings.max_research_rounds:
            gaps = await self.lead.review(question, findings, index, round_no=rounds + 1)
            if not gaps:
                break
            rounds += 1
            plan = plan + gaps
            findings = findings + await self._fan_out(gaps, index, failures)

        # Verification improves a report; losing it should not cost the report.
        # Both stages degrade to their unverified form, the way a dead subagent
        # narrows the research instead of ending it.
        contradictions: list[Contradiction] = []
        if self.verify:
            try:
                findings, contradictions = await self.critic.run(findings, index)
            except Exception as exc:  # noqa: BLE001 - any critic failure is survivable
                failures["critic"] = f"{type(exc).__name__}: {exc}"

        answer = await self.lead.synthesize(question, findings, index, contradictions)
        if self.verify:
            try:
                answer = await self.citer.run(answer, findings, index)
            except Exception as exc:  # noqa: BLE001 - ship it uncited rather than not at all
                failures["citer"] = f"{type(exc).__name__}: {exc}"

        return MultiAgentResult(
            question=question,
            answer=answer.strip(),
            complexity=complexity,
            plan=plan,
            findings=findings,
            sources=list(index.sources),
            contradictions=contradictions,
            failures=failures,
            rounds=rounds,
            ok=bool(findings) and not failures,
        )

    async def _fan_out(
        self, plan: list[SubTask], index: SourceIndex, failures: dict[str, str]
    ) -> list[Finding]:
        results = await asyncio.gather(
            *(self._run_task(task, index, failures) for task in plan),
            return_exceptions=True,
        )
        findings: list[Finding] = []
        for task, outcome in zip(plan, results):
            if isinstance(outcome, BaseException):
                failures[task.id] = f"{type(outcome).__name__}: {outcome}"
                continue
            findings.extend(outcome)
        return findings

    async def _run_task(
        self, task: SubTask, index: SourceIndex, failures: dict[str, str]
    ) -> list[Finding]:
        if task.tool == "database":
            agent = Analyst(
                self.settings, llm=self.llm, database=self.database, index=index
            )
        else:
            agent = Researcher(
                self.settings,
                llm=self.llm,
                web=self.web or WebSearch(self.settings),
                index=index,
            )

        try:
            return await asyncio.wait_for(
                agent.run(task), timeout=self.settings.subagent_timeout_s
            )
        except asyncio.TimeoutError:
            failures[task.id] = f"timed out after {self.settings.subagent_timeout_s}s"
            return []


async def research(question: str, **kwargs) -> MultiAgentResult:
    return await ResearchGraph(**kwargs).run(question)
