from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

# USD per 1M tokens (input, output). Estimates for budgeting, not billed truth;
# DeepSeek cache hits land ~50x below these.
PRICING: dict[str, tuple[float, float]] = {
    "deepseek-v4-pro": (0.435, 0.87),
    "deepseek-v4-flash": (0.14, 0.28),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}


class BudgetExceeded(RuntimeError):
    """Raised when a run passes its configured spend cap."""


@dataclass
class Call:
    agent: str  # "lead", "researcher:2", "critic"
    step: str  # "plan", "extract", "verify"
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float

    @property
    def cost_usd(self) -> float:
        pin, pout = PRICING.get(self.model, (0.0, 0.0))
        return (self.prompt_tokens * pin + self.completion_tokens * pout) / 1_000_000

    @property
    def priced(self) -> bool:
        return self.model in PRICING


@dataclass
class Tracker:
    calls: list[Call] = field(default_factory=list)
    budget_usd: float = 0.0  # 0 disables the guard

    def record(self, call: Call) -> None:
        # Record before checking, so the call that blew the budget stays visible.
        self.calls.append(call)
        if self.budget_usd and self.total_cost > self.budget_usd:
            raise BudgetExceeded(
                f"run cost ${self.total_cost:.4f} exceeded cap ${self.budget_usd:.2f} "
                f"after {len(self.calls)} calls"
            )

    def reset(self, budget_usd: float | None = None) -> None:
        self.calls = []
        if budget_usd is not None:
            self.budget_usd = budget_usd

    @property
    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.calls)

    @property
    def unpriced_models(self) -> list[str]:
        return sorted({c.model for c in self.calls if not c.priced})

    def summary(self) -> str:
        if not self.calls:
            return "no LLM calls recorded"
        by_agent: dict[str, list[Call]] = {}
        for c in self.calls:
            by_agent.setdefault(c.agent, []).append(c)
        lines = ["", "--- run report " + "-" * 29]
        for agent, calls in by_agent.items():
            toks = sum(c.prompt_tokens + c.completion_tokens for c in calls)
            cost = sum(c.cost_usd for c in calls)
            secs = sum(c.latency_s for c in calls)
            lines.append(
                f"  {agent:<20} {len(calls):>3} calls  {toks:>7} tok  {secs:>6.1f}s  ${cost:.4f}"
            )
        lines.append("  " + "-" * 42)
        lines.append(
            f"  {'TOTAL':<20} {len(self.calls):>3} calls  {self.total_tokens:>7} tok  "
            f"{sum(c.latency_s for c in self.calls):>6.1f}s  ${self.total_cost:.4f}"
        )
        if self.unpriced_models:
            lines.append(f"  (no pricing data: {', '.join(self.unpriced_models)})")
        lines.append("-" * 44)
        return "\n".join(lines)

    def dump(self, name: str, runs_dir: str = "runs") -> None:
        os.makedirs(runs_dir, exist_ok=True)
        with open(os.path.join(runs_dir, f"{name}.jsonl"), "a", encoding="utf-8") as fh:
            for c in self.calls:
                fh.write(json.dumps({**asdict(c), "cost_usd": c.cost_usd}) + "\n")


tracker = Tracker()
