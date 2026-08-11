from __future__ import annotations

import json
import re
from dataclasses import dataclass

from deepresearch.llm import LLMClient

SYSTEM = """You grade research answers. Be strict about evidence and lenient
about style.

Reply with a single JSON object and nothing else:

    {"score": 0-3, "reason": "one sentence"}

    3  meets the criterion, and the claims that matter carry citations
    2  substantially right, with a gap or an uncited claim
    1  partially right, or right for the wrong reason
    0  wrong, invents facts, or reports nothing at all

Honesty about a gap earns credit only on top of substance. An answer that is
mostly right and admits one hole is a 2. An answer that gathered nothing and
says so is a 0 however gracefully it says it — the question was answerable and
it went unanswered."""


@dataclass(frozen=True)
class Verdict:
    score: int
    reason: str

    @property
    def passed(self) -> bool:
        return self.score >= 2


def check_exact(answer: str, expect: list[str]) -> Verdict:
    """Deterministic check: no model call, no cost."""
    text = answer.lower()
    missing = [e for e in expect if e.lower() not in text]
    if not missing:
        return Verdict(3, "all expected values present")
    found = len(expect) - len(missing)
    return Verdict(
        1 if found else 0, f"missing {', '.join(missing)}"
    )


def check_judge(llm: LLMClient, question: str, answer: str, expect: str) -> Verdict:
    reply = llm.complete(
        f"Question:\n{question}\n\nCriterion:\n{expect}\n\nAnswer:\n{answer}",
        role="critic",
        system=SYSTEM,
        agent="judge",
        step="grade",
        # generous: the reply is short, but reasoning models bill thinking
        # against this budget before writing a single character
        max_tokens=2000,
    )
    try:
        start, end = reply.find("{"), reply.rfind("}")
        data = json.loads(reply[start : end + 1])
        return Verdict(int(data["score"]), str(data.get("reason", "")))
    except (ValueError, KeyError, json.JSONDecodeError):
        return Verdict(0, f"judge reply unparseable: {reply[:120]}")


def grade(llm: LLMClient, case: dict, answer: str) -> Verdict:
    if not answer.strip():
        return Verdict(0, "empty answer")
    if case["check"] == "exact":
        return check_exact(answer, case["expect"])
    return check_judge(llm, case["question"], answer, case["expect"])
