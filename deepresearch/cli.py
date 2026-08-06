from __future__ import annotations

import argparse
import asyncio

from . import observability
from .agents.baseline import BaselineAgent
from .config import settings
from .llm import LLMClient
from .observability import BudgetExceeded
from .tools.database import DatabaseTool, DatabaseUnavailable
from .tools.web import WebSearch

ROLES = ("lead", "critic", "subagent", "citation")


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check config and make one live call."""
    provider = args.provider or settings.provider
    print(f"provider        : {provider}")
    print(f"search provider : {settings.search_provider}")
    print("models")
    for role in ROLES:
        print(f"  {role:<10} : {settings.model_for(role)}")

    gaps = settings.missing()
    if provider == "mock":
        print("\nrunning in mock mode - no API keys needed")
    elif gaps:
        print(f"\nmissing config  : {', '.join(gaps)}")
        print("fill these in .env before a real run")
    else:
        print("\nconfig          : all required keys present")

    print(f"\ncalling {provider} once...")
    observability.tracker.reset(budget_usd=settings.max_cost_usd_per_run)
    try:
        client = LLMClient(provider=provider)
        reply = client.complete(
            "Reply with the single word: ok",
            role="subagent",
            agent="doctor",
            step="smoke",
            max_tokens=16,
        )
    except Exception as exc:  # noqa: BLE001 - report, don't crash the check
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1

    print(f"reply           : {reply.strip()!r}")
    print(observability.tracker.summary())
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Run one search and print what a subagent would receive."""
    try:
        results = WebSearch().search(args.query, num_results=args.n)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the check
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1

    if not results:
        print("no results")
        return 0
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.title}")
        print(f"    {r.url}")
        print(f"    {r.snippet[:300]}")
    return 0


def cmd_db(args: argparse.Namespace) -> int:
    """Ask the database subagent's MCP bridge one question."""

    async def run() -> str:
        tool = DatabaseTool()
        async with tool.session():
            if args.question:
                return (await tool.query(args.question, args.database)).answer
            return f"databases: {await tool.databases()}\n\n{await tool.schema(args.database)}"

    try:
        print(asyncio.run(run()))
    except DatabaseUnavailable as exc:
        print(f"FAILED: {exc}")
        return 1
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Run one research question end to end."""
    observability.tracker.reset(budget_usd=settings.max_cost_usd_per_run)

    async def run():
        agent = BaselineAgent(max_steps=args.max_steps)
        try:
            async with agent.database.session():
                return await agent.run(args.question)
        except DatabaseUnavailable as exc:
            # A dead database narrows the research, it doesn't end it.
            print(f"note: database unavailable ({exc}); using web sources only\n")
            return await agent.run(args.question)

    try:
        result = asyncio.run(run())
    except BudgetExceeded as exc:
        print(f"STOPPED: {exc}")
        print(observability.tracker.summary())
        return 1

    print(result.report())
    if args.trace:
        print("\n--- steps " + "-" * 33)
        for i, step in enumerate(result.steps, 1):
            print(f"  {i}. {step.action:<15} {step.detail[:50]:<52} {step.result}")
    if not result.ok:
        print("\n(step budget reached before the agent decided it was done)")
    print(observability.tracker.summary())
    observability.tracker.dump("ask")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepresearch")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_doctor = sub.add_parser("doctor", help="check config and make one test call")
    p_doctor.add_argument("--provider", help="override LLM_PROVIDER for this check")
    p_doctor.set_defaults(func=cmd_doctor)

    p_search = sub.add_parser("search", help="run one search, no LLM involved")
    p_search.add_argument("query")
    p_search.add_argument("-n", type=int, default=3, help="number of results")
    p_search.set_defaults(func=cmd_search)

    p_db = sub.add_parser("db", help="check the nlquery-agent MCP bridge")
    p_db.add_argument("question", nargs="?", help="omit to print schema instead")
    p_db.add_argument("--database", default="chinook")
    p_db.set_defaults(func=cmd_db)

    p_ask = sub.add_parser("ask", help="research a question end to end")
    p_ask.add_argument("question")
    p_ask.add_argument("--max-steps", type=int, default=8)
    p_ask.add_argument("--trace", action="store_true", help="show the tool calls")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)
