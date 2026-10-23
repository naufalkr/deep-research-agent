"""Score an agent on the fixed question set.

    python -m evals.run_eval --agent multiagent --tag v1.0
    python -m evals.run_eval --agent baseline --tag v1.0-baseline
    python -m evals.run_eval --only debate-,sources-   # id prefixes, comma-separated

Results land in runs/evals/<tag>.json; compare two with evals.compare.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from deepresearch import observability
from deepresearch.agents.baseline import BaselineAgent
from deepresearch.config import settings
from deepresearch.graph import ResearchGraph
from deepresearch.llm import LLMClient
from deepresearch.observability import BudgetExceeded
from deepresearch.tools.database import DatabaseUnavailable

from .judge import grade

QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"
RESULTS_DIR = Path("runs") / "evals"
GROUPS = ("simple", "hard", "sources", "debate")


def load_cases(only: str | None) -> list[dict]:
    cases = [
        json.loads(line)
        for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prefixes = tuple(p.strip() for p in only.split(",")) if only else ("",)
    return [c for c in cases if c["id"].startswith(prefixes)]


async def run_case(case: dict, agent: str) -> dict:
    """One question, one run. Cost and time are measured per case."""
    observability.tracker.reset(budget_usd=settings.max_cost_usd_per_run)
    start = time.perf_counter()
    runner = BaselineAgent() if agent == "baseline" else ResearchGraph()
    try:
        try:
            async with runner.database.session():
                result = await runner.run(case["question"])
        except DatabaseUnavailable:
            result = await runner.run(case["question"])
    except BudgetExceeded as exc:
        answer, extra = f"[budget exceeded: {exc}]", {}
    except Exception as exc:  # noqa: BLE001 - one broken case must not end the sweep
        answer, extra = f"[error: {type(exc).__name__}: {exc}]", {}
    else:
        answer, extra = result.report(), {}
        if agent != "baseline":
            # without these a failed case can only be diagnosed by reading prose
            extra = {
                "mode": result.mode,
                "motion": result.motion,
                "plan": [{"id": t.id, "stance": t.stance, "objective": t.objective} for t in result.plan],
                "rounds": result.rounds,
                "findings": len(result.findings),
                "failures": result.failures,
            }
    return {
        "answer": answer,
        "cost": observability.tracker.total_cost,
        "seconds": time.perf_counter() - start,
        **extra,
    }


def summarise(rows: list[dict]) -> dict:
    def stats(rs: list[dict]) -> dict:
        return {
            "n": len(rs),
            "passed": sum(r["score"] >= 2 for r in rs),
            "mean": round(sum(r["score"] for r in rs) / len(rs), 2),
            "cost_per_q": round(sum(r["cost_usd"] for r in rs) / len(rs), 4),
            "secs_per_q": round(sum(r["seconds"] for r in rs) / len(rs)),
        }

    out = {"all": stats(rows)}
    for g in GROUPS:
        rs = [r for r in rows if r["group"] == g]
        if rs:
            out[g] = stats(rs)
    judged = [r for r in rows if r.get("expected_mode") and "mode" in r]
    if judged:
        right = sum(r["mode"] == r["expected_mode"] for r in judged)
        out["mode_choice"] = f"{right}/{len(judged)}"
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(prog="evals.run_eval")
    parser.add_argument("--agent", choices=("baseline", "multiagent"), default="multiagent")
    parser.add_argument("--tag", default="latest", help="name for the saved results")
    parser.add_argument("--only", help="run only ids starting with these (comma-separated)")
    args = parser.parse_args()

    cases = load_cases(args.only)
    if not cases:
        print("no matching questions")
        return 1

    judge = LLMClient()
    out_path = RESULTS_DIR / f"{args.tag}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    print(f"{len(cases)} questions, agent={args.agent}, provider={settings.provider}\n", flush=True)

    for case in cases:
        run = await run_case(case, args.agent)
        verdict = grade(judge, case, run["answer"])
        rows.append(
            {
                "id": case["id"],
                "group": case["group"],
                "expected_mode": case.get("mode"),
                "score": verdict.score,
                "reason": verdict.reason,
                "cost_usd": round(run.pop("cost"), 5),
                "seconds": round(run.pop("seconds"), 1),
                **run,
            }
        )
        note = verdict.reason[:60]
        if run.get("failures"):
            note = f"[{len(run['failures'])} failed] {note}"
        mode = f"{run['mode']:<6}" if "mode" in run else ""
        print(
            f"  {'ok  ' if verdict.passed else 'FAIL'} {case['id']:<22} {verdict.score}/3 "
            f"{mode} ${rows[-1]['cost_usd']:.4f} {rows[-1]['seconds']:>6.1f}s  {note}",
            flush=True,
        )
        # saved after every case, so a crash or a stall keeps what finished
        out_path.write_text(
            json.dumps(
                {
                    "tag": args.tag,
                    "agent": args.agent,
                    "provider": settings.provider,
                    "models": {r: settings.model_for(r) for r in ("lead", "subagent")},
                    "summary": summarise(rows),
                    "cases": rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    summary = summarise(rows)
    print()
    for name, s in summary.items():
        if isinstance(s, dict):
            print(f"  {name:<8} {s['passed']}/{s['n']} passed  mean {s['mean']:.2f}  "
                  f"${s['cost_per_q']:.4f}/q  {s['secs_per_q']}s/q")
    if "mode_choice" in summary:
        print(f"  mode chosen correctly: {summary['mode_choice']}")
    print(f"  saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
