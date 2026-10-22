from __future__ import annotations

import argparse
import asyncio

from . import observability
from .agents.baseline import BaselineAgent
from .config import settings
from .graph import ResearchGraph
from .llm import LLMClient
from .observability import BudgetExceeded
from .tools.database import DatabaseTool, DatabaseUnavailable
from .tools.web import CATEGORIES, WebSearch


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check config and make one live call."""
    provider = args.provider or settings.provider
    print(f"provider : {provider}   search : {settings.search_provider}")
    for role in ("lead", "critic", "subagent", "citation"):
        print(f"  {role:<9}: {settings.model_for(role)}")

    gaps = settings.missing()
    if gaps and provider != "mock":
        print(f"missing  : {', '.join(gaps)} - add to .env")

    observability.tracker.reset(budget_usd=settings.max_cost_usd_per_run)
    try:
        reply = LLMClient(provider=provider).complete(
            "Reply with the single word: ok", agent="doctor", step="smoke", max_tokens=16
        )
    except Exception as exc:  # noqa: BLE001 - report, don't crash the check
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(f"reply    : {reply.strip()!r}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Run one search and print what a child agent would receive."""
    try:
        results = WebSearch().search(args.query, num_results=args.n, tool=args.tool)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    if not results:
        print("no results")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.title}\n    {r.url}\n    {r.snippet[:300]}")
    return 0


def cmd_db(args: argparse.Namespace) -> int:
    """Ask the database through the nlquery-agent MCP server."""

    async def run() -> str:
        tool = DatabaseTool()
        async with tool.session():
            if args.question:
                return await tool.query(args.question, args.database)
            return f"databases: {await tool.databases()}\n\n{await tool.schema(args.database)}"

    try:
        print(asyncio.run(run()))
    except DatabaseUnavailable as exc:
        print(f"FAILED: {exc}")
        return 1
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Research one question end to end."""
    observability.tracker.reset(budget_usd=settings.max_cost_usd_per_run)
    runner = BaselineAgent() if args.baseline else ResearchGraph()

    async def run():
        try:
            async with runner.database.session():
                return await runner.run(args.question)
        except DatabaseUnavailable as exc:
            print(f"note: database unavailable ({exc}); continuing without it\n")
            return await runner.run(args.question)

    try:
        result = asyncio.run(run())
    except BudgetExceeded as exc:
        print(f"STOPPED: {exc}")
        print(observability.tracker.summary())
        return 1

    print(result.report())
    if args.trace and not args.baseline:
        print(f"\n--- {result.mode}" + (f": {result.motion}" if result.motion else ""))
        for task in result.plan:
            print(f"  {task.id:<6} {task.stance:<8} {task.objective[:70]}")
        print(f"  {len(result.findings)} findings, {result.rounds} rounds")
    if not result.ok:
        print("\n(the run was incomplete - see failures or step budget)")
    print(observability.tracker.summary())
    observability.tracker.dump("ask")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepresearch")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="check config and make one test call")
    p.add_argument("--provider", help="override LLM_PROVIDER for this check")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("search", help="run one search, no LLM involved")
    p.add_argument("query")
    p.add_argument("--tool", choices=list(CATEGORIES), default="web")
    p.add_argument("-n", type=int, default=3, help="number of results")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("db", help="query the database through nlquery-agent")
    p.add_argument("question", nargs="?", help="omit to print the schema")
    p.add_argument("--database", default="chinook")
    p.set_defaults(func=cmd_db)

    p = sub.add_parser("ask", help="research a question end to end")
    p.add_argument("question")
    p.add_argument("--baseline", action="store_true", help="single agent instead")
    p.add_argument("--trace", action="store_true", help="show the plan")
    p.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)
