from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .agents.child import Child
from .agents.lead import Lead
from .agents.verify import Citer, Critic
from .config import Settings, settings as default_settings
from .llm import LLMClient
from .state import (
    Contradiction,
    Finding,
    Mode,
    Source,
    SourceIndex,
    SubTask,
    compact_citations,
)
from .tools.database import DatabaseTool
from .tools.web import WebSearch

Listener = Callable[[str, dict], Awaitable[None]]


@dataclass
class MultiAgentResult:
    question: str
    answer: str
    mode: Mode = "facts"
    motion: str = ""
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
    """lead plans -> children research in parallel -> second round -> check -> write -> cite.

    The second round depends on the mode: in facts mode the lead looks for
    parts nothing answered; in debate mode each side rebuts the other.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        web: WebSearch | None = None,
        database: DatabaseTool | None = None,
        on_event: Listener | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.llm = llm or LLMClient(settings=self.settings)
        self.web = web  # injected for tests; otherwise one per child
        self.database = database or DatabaseTool(self.settings)
        self.lead = Lead(self.settings, llm=self.llm)
        self.critic = Critic(self.settings, llm=self.llm)
        self.citer = Citer(self.settings, llm=self.llm)
        self.on_event = on_event

    async def run(self, question: str) -> MultiAgentResult:
        index = SourceIndex()
        failures: dict[str, str] = {}
        await self._emit("start", question=question)

        plan = await self.lead.plan(question)
        await self._emit("planned", plan=plan)
        tasks = list(plan.tasks)
        findings = await self._fan_out(tasks, index, failures)

        rounds = 1
        if plan.mode == "debate":
            rebuttals = self._rebuttal_tasks(plan.tasks, findings)
            if rebuttals:
                rounds = 2
                await self._emit("rebuttal", tasks=[t for t, _ in rebuttals])
                tasks += [t for t, _ in rebuttals]
                findings += await self._fan_out(rebuttals, index, failures)
        else:
            while rounds < self.settings.max_research_rounds:
                gaps = await self.lead.review(question, findings, index, round_no=rounds + 1)
                await self._emit("reviewed", round=rounds, gaps=gaps)
                if not gaps:
                    break
                rounds += 1
                tasks += gaps
                findings += await self._fan_out(gaps, index, failures)

        # Checking and citing improve a report; losing them must not cost it.
        contradictions: list[Contradiction] = []
        try:
            findings, contradictions = await self.critic.run(findings, index)
            await self._emit("verified", findings=findings, contradictions=contradictions)
        except Exception as exc:  # noqa: BLE001
            failures["critic"] = f"{type(exc).__name__}: {exc}"

        await self._emit("writing")
        answer = await self.lead.synthesize(question, plan, findings, index, contradictions)
        sources = list(index.sources)
        try:
            answer = await self.citer.run(answer, findings, index)
            answer, sources = compact_citations(answer, sources)
        except Exception as exc:  # noqa: BLE001
            failures["citer"] = f"{type(exc).__name__}: {exc}"

        return MultiAgentResult(
            question=question,
            answer=answer.strip(),
            mode=plan.mode,
            motion=plan.motion,
            complexity=plan.complexity,
            plan=tasks,
            findings=findings,
            sources=sources,
            contradictions=contradictions,
            failures=failures,
            rounds=rounds,
            ok=bool(findings) and not failures,
        )

    @staticmethod
    def _rebuttal_tasks(
        sides: list[SubTask], findings: list[Finding]
    ) -> list[tuple[SubTask, list[Finding]]]:
        """Each side gets a second turn to answer what the other side found."""
        out = []
        for task in sides:
            opponent = [f for f in findings if f.stance not in (task.stance, "neutral")]
            if opponent:
                out.append((SubTask(f"{task.id}-2", task.objective, task.stance), opponent))
        return out

    async def _emit(self, kind: str, **data) -> None:
        if self.on_event is None:
            return
        try:
            await self.on_event(kind, data)
        except Exception:  # noqa: BLE001 - a UI hiccup must not lose the research
            pass

    async def _fan_out(self, work: list, index: SourceIndex, failures: dict[str, str]) -> list[Finding]:
        """Run children in parallel. `work` holds tasks, or (task, opponent) pairs."""
        jobs = [w if isinstance(w, tuple) else (w, None) for w in work]
        results = await asyncio.gather(
            *(self._run_child(task, opponent, index, failures) for task, opponent in jobs),
            return_exceptions=True,
        )
        findings: list[Finding] = []
        for (task, _), outcome in zip(jobs, results):
            if isinstance(outcome, BaseException):
                failures[task.id] = f"{type(outcome).__name__}: {outcome}"
            else:
                findings.extend(outcome)
        return findings

    async def _run_child(
        self,
        task: SubTask,
        opponent: list[Finding] | None,
        index: SourceIndex,
        failures: dict[str, str],
    ) -> list[Finding]:
        child = Child(
            self.settings,
            llm=self.llm,
            web=self.web or WebSearch(self.settings),
            database=self.database,
            index=index,
        )
        await self._emit("child_start", task=task)
        try:
            found = await asyncio.wait_for(
                child.run(task, opponent), timeout=self.settings.subagent_timeout_s
            )
        except asyncio.TimeoutError:
            reason = f"timed out after {self.settings.subagent_timeout_s:.0f}s"
            failures[task.id] = reason
            await self._emit("child_failed", task=task, reason=reason)
            return []
        except Exception as exc:
            await self._emit("child_failed", task=task, reason=str(exc))
            raise
        await self._emit("child_done", task=task, findings=found)
        return found
