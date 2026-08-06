from __future__ import annotations

import argparse

from . import observability
from .config import settings
from .llm import LLMClient

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepresearch")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_doctor = sub.add_parser("doctor", help="check config and make one test call")
    p_doctor.add_argument("--provider", help="override LLM_PROVIDER for this check")
    p_doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return args.func(args)
