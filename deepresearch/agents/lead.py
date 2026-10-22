from __future__ import annotations

import json
from pathlib import Path

from ..config import Settings, settings as default_settings
from ..llm import LLMClient, parse_json
from ..state import Contradiction, Finding, Plan, SourceIndex, SubTask, render_findings

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def _prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


# One prompt per step: a shared one let the report step answer with review JSON.
PLAN_PROMPT = _prompt("lead_plan.md")
REVIEW_PROMPT = _prompt("lead_review.md")
REPORT_PROMPT = _prompt("lead_report.md")
VERDICT_PROMPT = _prompt("lead_verdict.md")

TASKS_PER_COMPLEXITY = {"simple": 1, "moderate": 3, "complex": 5}


class Lead:
    """Picks the mode, splits the work, checks coverage, writes the report.
    Never searches anything itself."""

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

    async def plan(self, question: str) -> Plan:
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

        motion = str(payload.get("motion") or "").strip()
        if payload.get("mode") == "debate" and motion:
            return Plan(
                mode="debate",
                motion=motion,
                complexity="debate",
                tasks=[SubTask("pro", motion, "pro"), SubTask("con", motion, "con")],
            )

        complexity = payload.get("complexity")
        if complexity not in TASKS_PER_COMPLEXITY:
            complexity = "moderate"
        tasks = self._tasks(payload.get("tasks", []), prefix="t")
        if not tasks:
            tasks = [SubTask("t1", question)]  # a failed plan still gets researched
        cap = min(self.max_subagents, TASKS_PER_COMPLEXITY[complexity])
        return Plan(mode="facts", complexity=complexity, tasks=tasks[:cap])

    async def review(
        self, question: str, findings: list[Finding], index: SourceIndex, *, round_no: int
    ) -> list[SubTask]:
        """Parts of the question nothing has answered yet."""
        if not findings:
            return []
        reply = await self.llm.acomplete(
            f"Question: {question}\n\nFindings so far:\n\n{render_findings(findings, index)}",
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
        return self._tasks(gaps, prefix=f"r{round_no}-")[: self.max_subagents]

    async def synthesize(
        self,
        question: str,
        plan: Plan,
        findings: list[Finding],
        index: SourceIndex,
        contradictions: list[Contradiction] | None = None,
    ) -> str:
        if not findings:
            return (
                "The research returned no usable findings, so this question is left "
                "unanswered rather than answered from guesswork."
            )
        prompt = f"Question: {question}\n\n"
        if plan.mode == "debate":
            prompt += f"Position debated: {plan.motion}\n\n"
        prompt += f"Findings:\n\n{render_findings(findings, index)}"
        if contradictions:
            clashes = "\n\n".join(c.render() for c in contradictions)
            prompt += f"\n\nThe critic found these findings in conflict:\n\n{clashes}"
        return await self.llm.acomplete(
            prompt,
            role="lead",
            system=VERDICT_PROMPT if plan.mode == "debate" else REPORT_PROMPT,
            agent="lead",
            step="synthesize",
            max_tokens=8000,
        )

    @staticmethod
    def _tasks(raw: list, *, prefix: str) -> list[SubTask]:
        tasks = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            objective = str(item.get("objective") or "").strip()
            if objective:
                tasks.append(
                    SubTask(
                        id=f"{prefix}{len(tasks) + 1}",
                        objective=objective,
                        boundaries=str(item.get("boundaries") or "").strip(),
                    )
                )
        return tasks
