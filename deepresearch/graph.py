from __future__ import annotations

import asyncio
import operator
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents.child import Child
from .agents.lead import Lead
from .agents.verify import Citer, Critic
from .config import Settings, settings as default_settings
from .llm import LLMClient
from .state import (
    Contradiction,
    Finding,
    Mode,
    Plan,
    Source,
    SourceIndex,
    SubTask,
    compact_citations,
)
from .tools.database import DatabaseTool
from .tools.web import WebSearch

Listener = Callable[[str, dict], Awaitable[None]]


def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


class Job(TypedDict):
    """What one child is sent: its task, and in a rebuttal the other side's findings."""

    task: SubTask
    opponent: list[Finding] | None
    index: SourceIndex


class State(TypedDict, total=False):
    question: str
    index: SourceIndex  # shared, so a page found by two children gets one number
    plan: Plan
    round: int
    pending: list[Job]  # children to send next; empty means move on to checking
    # children run in parallel and all write here, so these are merged, not replaced
    tasks: Annotated[list[SubTask], operator.add]
    findings: Annotated[list[Finding], operator.add]
    failures: Annotated[dict[str, str], _merge]
    checked: list[Finding]
    contradictions: list[Contradiction]
    answer: str
    sources: list[Source]


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
    """The research run as a LangGraph StateGraph:

        plan -> child x N (parallel) -> next_round -+-> child x N ...
                                                    +-> check -> write -> cite

    next_round depends on the mode: in facts mode the lead looks for parts
    nothing answered; in debate mode each side rebuts the other, once.
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
        self.graph = self._build()

    def _build(self):
        g = StateGraph(State)
        g.add_node("plan", self._plan)
        g.add_node("child", self._child)
        g.add_node("next_round", self._next_round)
        g.add_node("check", self._check)
        g.add_node("write", self._write)
        g.add_node("cite", self._cite)
        g.add_edge(START, "plan")
        g.add_conditional_edges("plan", self._dispatch, ["child", "check"])
        g.add_edge("child", "next_round")  # runs once every child of the round is done
        g.add_conditional_edges("next_round", self._dispatch, ["child", "check"])
        g.add_edge("check", "write")
        g.add_edge("write", "cite")
        g.add_edge("cite", END)
        return g.compile()

    async def run(self, question: str) -> MultiAgentResult:
        await self._emit("start", question=question)
        s = await self.graph.ainvoke(
            {"question": question, "index": SourceIndex(), "tasks": [], "findings": [], "failures": {}}
        )
        plan, findings = s["plan"], s["checked"]
        return MultiAgentResult(
            question=question,
            answer=s["answer"].strip(),
            mode=plan.mode,
            motion=plan.motion,
            complexity=plan.complexity,
            plan=s["tasks"],
            findings=findings,
            sources=s["sources"],
            contradictions=s["contradictions"],
            failures=s["failures"],
            rounds=s["round"],
            ok=bool(findings) and not s["failures"],
        )

    # --- nodes ---------------------------------------------------------------

    async def _plan(self, s: State) -> dict:
        plan = await self.lead.plan(s["question"])
        await self._emit("planned", plan=plan)
        return {
            "plan": plan,
            "round": 1,
            "tasks": plan.tasks,
            "pending": [self._job(s, t) for t in plan.tasks],
        }

    async def _child(self, job: Job) -> dict:
        task = job["task"]
        child = Child(
            self.settings,
            llm=self.llm,
            web=self.web or WebSearch(self.settings),
            database=self.database,
            index=job["index"],
        )
        await self._emit("child_start", task=task)
        try:
            found = await asyncio.wait_for(
                child.run(task, job["opponent"]), timeout=self.settings.subagent_timeout_s
            )
        except asyncio.TimeoutError:
            reason = f"timed out after {self.settings.subagent_timeout_s:.0f}s"
            await self._emit("child_failed", task=task, reason=reason)
            return {"failures": {task.id: reason}}
        except Exception as exc:  # noqa: BLE001 - one dead child narrows the research
            await self._emit("child_failed", task=task, reason=str(exc))
            return {"failures": {task.id: f"{type(exc).__name__}: {exc}"}}
        await self._emit("child_done", task=task, findings=found)
        return {"findings": found}

    async def _next_round(self, s: State) -> dict:
        plan, findings, round_no = s["plan"], s["findings"], s["round"]

        if plan.mode == "debate":
            if round_no > 1:
                return {"pending": []}
            rebuttals = []
            for task in plan.tasks:  # each side answers what the other side found
                opponent = [f for f in findings if f.stance not in (task.stance, "neutral")]
                if opponent:
                    rebuttals.append((SubTask(f"{task.id}-2", task.objective, task.stance), opponent))
            if not rebuttals:
                return {"pending": []}
            await self._emit("rebuttal", tasks=[t for t, _ in rebuttals])
            return {
                "round": 2,
                "tasks": [t for t, _ in rebuttals],
                "pending": [self._job(s, t, opp) for t, opp in rebuttals],
            }

        # A plan made once and never checked leaves whatever it overlooked missing.
        if round_no >= self.settings.max_research_rounds:
            return {"pending": []}
        gaps = await self.lead.review(s["question"], findings, s["index"], round_no=round_no + 1)
        await self._emit("reviewed", round=round_no, gaps=gaps)
        if not gaps:
            return {"pending": []}
        return {"round": round_no + 1, "tasks": gaps, "pending": [self._job(s, g) for g in gaps]}

    async def _check(self, s: State) -> dict:
        # Checking and citing improve a report; losing them must not cost it.
        try:
            checked, contradictions = await self.critic.run(s["findings"], s["index"])
        except Exception as exc:  # noqa: BLE001
            return {
                "checked": s["findings"],
                "contradictions": [],
                "failures": {"critic": f"{type(exc).__name__}: {exc}"},
            }
        await self._emit("verified", findings=checked, contradictions=contradictions)
        return {"checked": checked, "contradictions": contradictions}

    async def _write(self, s: State) -> dict:
        await self._emit("writing")
        answer = await self.lead.synthesize(
            s["question"], s["plan"], s["checked"], s["index"], s["contradictions"]
        )
        return {"answer": answer}

    async def _cite(self, s: State) -> dict:
        try:
            answer = await self.citer.run(s["answer"], s["checked"], s["index"])
        except Exception as exc:  # noqa: BLE001 - ship it uncited rather than not at all
            return {"sources": list(s["index"].sources), "failures": {"citer": f"{type(exc).__name__}: {exc}"}}
        answer, sources = compact_citations(answer, s["index"].sources)
        return {"answer": answer, "sources": sources}

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _dispatch(s: State):
        """Send every pending child in parallel, or move on to checking."""
        return [Send("child", job) for job in s.get("pending", [])] or "check"

    @staticmethod
    def _job(s: State, task: SubTask, opponent: list[Finding] | None = None) -> Job:
        return {"task": task, "opponent": opponent, "index": s["index"]}

    async def _emit(self, kind: str, **data) -> None:
        if self.on_event is None:
            return
        try:
            await self.on_event(kind, data)
        except Exception:  # noqa: BLE001 - a UI hiccup must not lose the research
            pass
