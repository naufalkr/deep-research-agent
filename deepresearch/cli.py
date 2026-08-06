from __future__ import annotations

import argparse
import asyncio

from . import observability
from .config import settings
from .llm import LLMClient
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

    args = parser.parse_args(argv)
    return args.func(args)
