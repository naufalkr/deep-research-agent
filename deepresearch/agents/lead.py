from __future__ import annotations

import json
from pathlib import Path

from ..config import Settings, settings as default_settings
from ..llm import LLMClient
from ..state import Contradiction, Finding, SourceIndex, SubTask, render_findings
from .subagents import parse_json

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def _prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


# One prompt per step. A single file covering all three let the model answer a
# request to write the report with the review step's JSON instead.
PLAN_PROMPT = _prompt("lead_plan.md")
REVIEW_PROMPT = _prompt("lead_review.md")
REPORT_PROMPT = _prompt("lead_report.md")

TASKS_PER_COMPLEXITY = {"simple": 1, "moderate": 3, "complex": 6}


class Lead:
    """Plans the research, then writes the report. Never calls a tool itself."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        max_subagents: int | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.llm = llm or LLMClient(settings=self.settings)
        self.max_subagents = max_subagents or self.settings.max_subagents

    async def plan(self, question: str) -> tuple[str, list[SubTask]]:
        reply = await self.llm.acomplete(
            f"Question: {question}",
            role="lead",
            system=PLAN_PROMPT,
            agent="lead",
            step="plan",
            max_tokens=4000,
        )
        try:
            payload = parse_json(reply)
        except (ValueError, json.JSONDecodeError):
            payload = {}
        return self._tasks_from(payload, question)

    def _tasks_from(self, payload: dict, question: str) -> tuple[str, list[SubTask]]:
        complexity = payload.get("complexity", "moderate")
        if complexity not in TASKS_PER_COMPLEXITY:
            complexity = "moderate"

        tasks = []
        for i, raw in enumerate(payload.get("tasks", []), 1):
            objective = (raw.get("objective") or "").strip()
            if not objective:
                continue
            tool = raw.get("tool")
            tasks.append(
                SubTask(
                    id=str(raw.get("id") or f"t{i}"),
                    objective=objective,
                    tool=tool if tool in ("web", "database") else "web",
                    boundaries=(raw.get("boundaries") or "").strip(),
                )
            )

        if not tasks:
            # A failed plan still gets one subagent rather than no research.
            tasks = [SubTask("t1", question, "web")]

        cap = min(self.max_subagents, TASKS_PER_COMPLEXITY[complexity])
        return complexity, tasks[:cap]

    async def review(
        self,
        question: str,
        findings: list[Finding],
        index: SourceIndex,
        *,
        round_no: int,
    ) -> list[SubTask]:
        """Parts of the question nothing has answered yet.

        Without this the plan is made once and never checked, so a part the
        planner overlooked stays missing however many subagents ran.
        """
        if not findings:
            return []
        reply = await self.llm.acomplete(
            f"Question: {question}\n\nFindings so far:\n\n"
            f"{render_findings(findings, index)}",
            role="lead",
            system=REVIEW_PROMPT,
            agent="lead",
            step="review",
            max_tokens=4000,
        )
        try:
            gaps = parse_json(reply).get("gaps", [])
        except (ValueError, json.JSONDecodeError):
            return []

        tasks = []
        for i, raw in enumerate(gaps, 1):
            objective = (raw.get("objective") or "").strip()
            if not objective:
                continue
            tool = raw.get("tool")
            tasks.append(
                SubTask(
                    id=f"r{round_no}-{i}",
                    objective=objective,
                    tool=tool if tool in ("web", "database") else "web",
                    boundaries=(raw.get("boundaries") or "").strip(),
                )
            )
        return tasks[: self.max_subagents]

    async def synthesize(
        self,
        question: str,
        findings: list[Finding],
        index: SourceIndex,
        contradictions: list[Contradiction] | None = None,
    ) -> str:
        if not findings:
            return (
                "The research returned no usable findings, so this question is left "
                "unanswered rather than answered from guesswork."
            )
        prompt = f"Question: {question}\n\nFindings:\n\n{render_findings(findings, index)}"
        if contradictions:
            clashes = "\n\n".join(c.render() for c in contradictions)
            prompt += f"\n\nThe critic found these findings in conflict:\n\n{clashes}"
        return await self.llm.acomplete(
            prompt,
            role="lead",
            system=REPORT_PROMPT,
            agent="lead",
            step="synthesize",
            max_tokens=8000,
        )
