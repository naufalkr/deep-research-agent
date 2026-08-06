"""Score the agent on a fixed question set.

    python -m evals.run_eval                  # all questions
    python -m evals.run_eval --only db-       # id prefix filter
    python -m evals.run_eval --tag v0.1       # label the saved results

Results land in runs/evals/<tag>.json so two versions can be compared later.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from deepresearch import observability
from deepresearch.agents.baseline import BaselineAgent
from deepresearch.config import settings
from deepresearch.llm import LLMClient
from deepresearch.observability import BudgetExceeded
from deepresearch.tools.database import DatabaseUnavailable

from .judge import grade

QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"
RESULTS_DIR = Path("runs") / "evals"


def load_cases(only: str | None) -> list[dict]:
    cases = [
        json.loads(line)
        for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [c for c in cases if not only or c["id"].startswith(only)]


async def run_case(case: dict, max_steps: int) -> tuple[str, float, float]:
    """Returns (answer, cost, seconds). Cost is measured per case, not per run."""
    observability.tracker.reset(budget_usd=settings.max_cost_usd_per_run)
    start = time.perf_counter()
    agent = BaselineAgent(max_steps=max_steps)
    try:
        async with agent.database.session():
            result = await agent.run(case["question"])
    except DatabaseUnavailable:
        result = await agent.run(case["question"])
    except BudgetExceeded as exc:
        return f"[budget exceeded: {exc}]", observability.tracker.total_cost, 0.0
    return (
        result.report(),
        observability.tracker.total_cost,
        time.perf_counter() - start,
    )


async def main() -> int:
    parser = argparse.ArgumentParser(prog="evals.run_eval")
    parser.add_argument("--only", help="run only ids starting with this")
    parser.add_argument("--tag", default="latest", help="name for the saved results")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    cases = load_cases(args.only)
    if not cases:
        print("no matching questions")
        return 1

    judge_llm = LLMClient()
    rows = []
    print(f"{len(cases)} questions, provider={settings.provider}\n")

    for case in cases:
        answer, cost, secs = await run_case(case, args.max_steps)
        verdict = grade(judge_llm, case, answer)
        rows.append(
            {
                "id": case["id"],
                "check": case["check"],
                "score": verdict.score,
                "reason": verdict.reason,
                "cost_usd": round(cost, 5),
                "seconds": round(secs, 1),
                "answer": answer,
            }
        )
        mark = "ok  " if verdict.passed else "FAIL"
        print(
            f"  {mark} {case['id']:<16} {verdict.score}/3  "
            f"${cost:.4f}  {secs:>5.1f}s  {verdict.reason[:60]}"
        )

    total_cost = sum(r["cost_usd"] for r in rows)
    passed = sum(1 for r in rows if r["score"] >= 2)
    mean = sum(r["score"] for r in rows) / len(rows)

    print(f"\n  passed   {passed}/{len(rows)}")
    print(f"  mean     {mean:.2f}/3")
    print(f"  cost     ${total_cost:.4f}  (${total_cost / len(rows):.4f} per question)")
    print(f"  time     {sum(r['seconds'] for r in rows):.0f}s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = RESULTS_DIR / f"{args.tag}.json"
    out.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "provider": settings.provider,
                "models": {r: settings.model_for(r) for r in ("lead", "critic")},
                "passed": passed,
                "of": len(rows),
                "mean_score": round(mean, 2),
                "total_cost_usd": round(total_cost, 5),
                "cases": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"  saved    {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
