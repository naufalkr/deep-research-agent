"""Chainlit front end. The only file that knows what Chainlit is."""
from __future__ import annotations

import sys
from pathlib import Path

# chainlit runs this file directly, so the project root is not on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chainlit as cl  # noqa: E402

from deepresearch import observability  # noqa: E402
from deepresearch.config import settings  # noqa: E402
from deepresearch.graph import MultiAgentResult, ResearchGraph  # noqa: E402
from deepresearch.observability import BudgetExceeded  # noqa: E402
from deepresearch.tools.database import DatabaseUnavailable  # noqa: E402

SIDE = {"pro": "For", "con": "Against"}


def child_name(task) -> str:
    if task.stance in SIDE:
        return SIDE[task.stance] + (" · rebuttal" if task.id.endswith("-2") else "")
    return f"Child {task.id}"


class StepTree:
    """Turns graph events into Chainlit steps. Children run at the same time,
    so their steps are tracked by task id."""

    def __init__(self) -> None:
        self.open: dict[str, cl.Step] = {}

    async def __call__(self, kind: str, data: dict) -> None:
        handler = getattr(self, f"_{kind}", None)
        if handler:
            await handler(data)

    async def _planned(self, data: dict) -> None:
        plan = data["plan"]
        async with cl.Step(name="Plan", type="llm", show_input=False) as step:
            if plan.mode == "debate":
                step.output = f"**Debate** · {plan.motion}"
            else:
                tasks = "\n".join(f"- {t.objective}" for t in plan.tasks)
                step.output = f"**Facts** · {len(plan.tasks)} children\n\n{tasks}"

    async def _child_start(self, data: dict) -> None:
        task = data["task"]
        step = cl.Step(name=child_name(task), type="tool", show_input=False)
        step.output = task.objective if task.stance == "neutral" else ""
        await step.send()
        self.open[task.id] = step

    async def _child_done(self, data: dict) -> None:
        step = self.open.pop(data["task"].id, None)
        if step:
            found = len(data["findings"])
            step.output = f"{step.output}\n\n**{found} findings**".strip()
            await step.update()

    async def _child_failed(self, data: dict) -> None:
        step = self.open.pop(data["task"].id, None)
        if step:
            step.output = f"Failed: {data['reason']}"
            step.is_error = True
            await step.update()

    async def _reviewed(self, data: dict) -> None:
        gaps = data["gaps"]
        async with cl.Step(name="Coverage check", type="llm", show_input=False) as step:
            step.output = (
                "Complete" if not gaps else "Missing:\n" + "\n".join(f"- {g.objective}" for g in gaps)
            )

    async def _verified(self, data: dict) -> None:
        counts: dict[str, int] = {}
        for f in data["findings"]:
            counts[f.verdict] = counts.get(f.verdict, 0) + 1
        parts = [f"{n} {v}" for v, n in sorted(counts.items())]
        if data["contradictions"]:
            parts.append(f"{len(data['contradictions'])} conflicts")
        async with cl.Step(name="Fact check", type="llm", show_input=False) as step:
            step.output = " · ".join(parts)


def render(result: MultiAgentResult) -> str:
    tracker = observability.tracker
    lines = [result.answer]
    if result.sources:
        lines += ["", "---", "**Sources**", ""]
        for s in result.sources:
            link = s.url if s.url.startswith("http") else ""
            lines.append(f"{s.n}. [{s.title}]({link})" if link else f"{s.n}. {s.title}")
    minutes, seconds = divmod(int(tracker.elapsed_s), 60)
    lines += [
        "",
        f"*{result.mode} · {len(result.findings)} findings · {result.rounds} rounds · "
        f"${tracker.total_cost:.3f} · {minutes}m {seconds:02d}s*",
    ]
    if result.failures:
        lines.append(f"*incomplete: {', '.join(result.failures)}*")
    return "\n".join(lines)


@cl.set_starters
async def starters() -> list[cl.Starter]:
    return [
        cl.Starter(
            label="Should banks use AI for loans?",
            message="Should banks use AI models to approve or reject consumer loans?",
        ),
        cl.Starter(
            label="Fintech defaults in Asia",
            message=(
                "Compare the rise in fintech lending defaults in Indonesia, the "
                "Philippines and Thailand: default figures, what regulators blamed, "
                "and what they did about it."
            ),
        ),
        cl.Starter(
            label="Chinook revenue vs real market",
            message=(
                "Find the top 3 countries by invoice revenue in the chinook database, "
                "and what the recorded-music market looks like in each today."
            ),
        ),
    ]


@cl.on_chat_start
async def start() -> None:
    missing = settings.missing()
    if missing:
        await cl.Message(content=f"Missing in `.env`: {', '.join(missing)}").send()


@cl.on_message
async def main(message: cl.Message) -> None:
    observability.tracker.reset(budget_usd=settings.max_cost_usd_per_run)
    graph = ResearchGraph(on_event=StepTree())
    try:
        try:
            async with graph.database.session():
                result = await graph.run(message.content)
        except DatabaseUnavailable:
            result = await graph.run(message.content)
    except BudgetExceeded as exc:
        await cl.Message(content=f"Stopped: {exc}").send()
        return

    await cl.Message(content=render(result)).send()
    observability.tracker.dump("ui")
