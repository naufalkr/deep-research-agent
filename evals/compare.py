"""Put two eval runs side by side as a markdown table.

    python -m evals.compare v1.0-baseline v1.0
"""
from __future__ import annotations

import argparse
import json

from .run_eval import GROUPS, RESULTS_DIR

LABELS = {
    "simple": "Simple lookups",
    "hard": "Multi-part",
    "sources": "News + papers",
    "debate": "Judgement (debate)",
    "all": "**All**",
}


def load(tag: str) -> dict:
    return json.loads((RESULTS_DIR / f"{tag}.json").read_text(encoding="utf-8"))


def table(a: dict, b: dict) -> str:
    rows = [
        f"| | n | {a['tag']} | {b['tag']} |",
        "|---|---|---|---|",
    ]
    for g in (*GROUPS, "all"):
        sa, sb = a["summary"].get(g), b["summary"].get(g)
        if not sa or not sb:
            continue
        rows.append(
            f"| {LABELS[g]} | {sa['n']} | {sa['mean']:.2f} ({sa['passed']}/{sa['n']}) "
            f"| {sb['mean']:.2f} ({sb['passed']}/{sb['n']}) |"
        )
    sa, sb = a["summary"]["all"], b["summary"]["all"]
    rows.append(f"| Cost / question | | ${sa['cost_per_q']:.4f} | ${sb['cost_per_q']:.4f} |")
    rows.append(f"| Time / question | | {sa['secs_per_q']}s | {sb['secs_per_q']}s |")
    return "\n".join(rows)


def changes(a: dict, b: dict) -> str:
    before = {c["id"]: c for c in a["cases"]}
    lines = []
    for c in b["cases"]:
        old = before.get(c["id"])
        if old and old["score"] != c["score"]:
            lines.append(f"- `{c['id']}` {old['score']} -> {c['score']}: {c['reason']}")
    return "\n".join(lines) or "- no score changed"


def main() -> int:
    parser = argparse.ArgumentParser(prog="evals.compare")
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    a, b = load(args.before), load(args.after)
    print(table(a, b))
    if "mode_choice" in b["summary"]:
        print(f"\nMode chosen correctly ({b['tag']}): {b['summary']['mode_choice']}")
    print(f"\nScore changes:\n{changes(a, b)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
