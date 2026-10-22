"""Chainlit front end for the research graph.

The graph reports progress through its `on_event` seam; this file is the only
place that knows what Chainlit is, so `deepresearch` stays UI-agnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path

# chainlit runs this file directly, so sys.path holds ui/ rather than the
# project root and `deepresearch` would not be importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chainlit as cl  # noqa: E402

from deepresearch import observability  # noqa: E402
from deepresearch.config import settings  # noqa: E402
from deepresearch.graph import ResearchGraph  # noqa: E402
from deepresearch.observability import BudgetExceeded  # noqa: E402
from deepresearch.tools.database import DatabaseUnavailable  # noqa: E402

TOOL_LABEL = {"web": "web", "database": "database"}


class StepTree:
    """Turns graph events into Chainlit steps.

    Subagents run concurrently, so their steps are opened and closed by id
    rather than nested by call stack.
    """

    def __init__(self) -> None:
        self.open: dict[str, cl.Step] = {}

    async def __call__(self, kind: str, data: dict) -> None:
        handler = getattr(self, f"_{kind}", None)
        if handler:
            await handler(data)

    async def _planned(self, data: dict) -> None:
        tasks = data["tasks"]
        async with cl.Step(name="lead · plan", type="llm") as step:
            step.output = f"complexity: {data['complexity']} → {len(tasks)} subagents"
            step.elements = [
                cl.Text(
                    name="tasks",
                    content="\n".join(
                        f"{t.id}  [{TOOL_LABEL.get(t.tool, t.tool)}]  {t.objective}"
                        for t in tasks
                    ),
                    display="inline",
                )
            ]

    async def _subagent_start(self, data: dict) -> None:
        task = data["task"]
        step = cl.Step(name=f"{TOOL_LABEL.get(task.tool, task.tool)} · {task.id}", type="tool")
        step.input = task.objective
        await step.send()
        self.open[task.id] = step

    async def _subagent_done(self, data: dict) -> None:
        task, found = data["task"], data["findings"]
        step = self.open.pop(task.id, None)
        if step:
            step.output = f"{len(found)} findings"
            await step.update()

    async def _subagent_failed(self, data: dict) -> None:
        task = data["task"]
        step = self.open.pop(task.id, None)
        if step:
            step.output = f"failed: {data['reason']}"
            step.is_error = True
            await step.update()

    async def _reviewed(self, data: dict) -> None:
        gaps = data["gaps"]
        async with cl.Step(name="lead · coverage review", type="llm") as step:
            step.input = f"{data['findings']} findings against the question"
            step.output = (
                "every part covered"
                if not gaps
                else f"{len(gaps)} gap(s) → round {data['round'] + 1}\n"
                + "\n".join(f"- {g.objective}" for g in gaps)
            )

    async def _verified(self, data: dict) -> None:
        findings, clashes = data["findings"], data["contradictions"]
        counts: dict[str, int] = {}
        for f in findings:
            counts[f.verdict] = counts.get(f.verdict, 0) + 1
        async with cl.Step(name="critic", type="llm") as step:
            step.input = f"{len(findings)} claims"
            lines = [f"{v}: {n}" for v, n in sorted(counts.items())]
            if clashes:
                lines.append("")
                lines += [f"conflict: {c.note}" for c in clashes]
            step.output = "\n".join(lines)

    async def _cited(self, data: dict) -> None:
        async with cl.Step(name="citer", type="llm") as step:
            step.output = f"{data['markers']} citation markers attached"


@cl.on_chat_start
async def start() -> None:
    gaps = settings.missing()
    if gaps:
        await cl.Message(
            content=f"⚠️ Missing config: {', '.join(gaps)} — add them to `.env`."
        ).send()
        return
    await cl.Message(
        content=(
            "Ask a research question. Anything needing hard numbers is sent to "
            "the SQL database; everything else is researched on the web.\n\n"
            f"*{settings.provider} · lead `{settings.model_for('lead')}`*"
        )
    ).send()


@cl.on_message
async def main(message: cl.Message) -> None:
    observability.tracker.reset(budget_usd=settings.max_cost_usd_per_run)
    graph = ResearchGraph(on_event=StepTree())

    try:
        async with graph.database.session():
            result = await graph.run(message.content)
    except DatabaseUnavailable:
        # A dead database narrows the research; it does not end it.
        result = await graph.run(message.content)
    except BudgetExceeded as exc:
        await cl.Message(content=f"⚠️ Stopped: {exc}").send()
        return

    tracker = observability.tracker
    elements = [
        cl.Text(
            name="run report",
            content=(
                f"rounds     {result.rounds}\n"
                f"findings   {len(result.findings)}\n"
                f"sources    {len(result.sources)}\n"
                f"cost       ${tracker.total_cost:.4f}\n"
                f"elapsed    {tracker.elapsed_s:.0f}s "
                f"({tracker.call_seconds - tracker.elapsed_s:.0f}s saved in parallel)"
            ),
            display="inline",
        ),
        cl.Text(
            name="sources",
            content="\n".join(str(s) for s in result.sources) or "none",
            display="inline",
        ),
    ]
    if result.failures:
        elements.append(
            cl.Text(
                name="failures",
                content="\n".join(f"{k}: {v}" for k, v in result.failures.items()),
                display="inline",
            )
        )

    await cl.Message(content=result.answer, elements=elements).send()
    observability.tracker.dump("ui")
