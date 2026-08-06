from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# absolute path so it loads regardless of the working directory
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ROOT = Path(__file__).resolve().parent.parent

Role = Literal["lead", "subagent", "critic", "citation"]


@dataclass(frozen=True)
class Settings:
    provider: str
    anthropic_api_key: str
    deepseek_api_key: str
    groq_api_key: str
    openrouter_api_key: str

    lead_model: str
    critic_model: str
    subagent_model: str
    citation_model: str

    search_provider: str
    exa_api_key: str
    tavily_api_key: str

    nlquery_mcp_command: str
    nlquery_mcp_args: str

    max_subagents: int
    max_research_rounds: int
    subagent_timeout_s: float
    max_cost_usd_per_run: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.getenv("LLM_PROVIDER", "deepseek").lower(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            lead_model=os.getenv("LEAD_MODEL", "deepseek-v4-pro"),
            critic_model=os.getenv("CRITIC_MODEL", "deepseek-v4-pro"),
            subagent_model=os.getenv("SUBAGENT_MODEL", "deepseek-v4-flash"),
            citation_model=os.getenv("CITATION_MODEL", "deepseek-v4-flash"),
            search_provider=os.getenv("SEARCH_PROVIDER", "exa").lower(),
            exa_api_key=os.getenv("EXA_API_KEY", ""),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
            nlquery_mcp_command=os.getenv("NLQUERY_MCP_COMMAND", "python"),
            nlquery_mcp_args=os.getenv("NLQUERY_MCP_ARGS", ""),
            max_subagents=int(os.getenv("MAX_SUBAGENTS", "5")),
            max_research_rounds=int(os.getenv("MAX_RESEARCH_ROUNDS", "3")),
            subagent_timeout_s=float(os.getenv("SUBAGENT_TIMEOUT_S", "120")),
            max_cost_usd_per_run=float(os.getenv("MAX_COST_USD_PER_RUN", "1.00")),
        )

    def model_for(self, role: Role) -> str:
        return {
            "lead": self.lead_model,
            "critic": self.critic_model,
            "subagent": self.subagent_model,
            "citation": self.citation_model,
        }[role]

    def api_key_for(self, provider: str) -> str:
        return {
            "anthropic": self.anthropic_api_key,
            "deepseek": self.deepseek_api_key,
            "groq": self.groq_api_key,
            "openrouter": self.openrouter_api_key,
            "mock": "n/a",
        }.get(provider.lower(), "")

    def missing(self) -> list[str]:
        """Config that must be filled in before a real (non-mock) run."""
        gaps = []
        if self.provider != "mock" and not self.api_key_for(self.provider):
            gaps.append(f"{self.provider.upper()}_API_KEY")
        if self.search_provider == "exa" and not self.exa_api_key:
            gaps.append("EXA_API_KEY")
        if self.search_provider == "tavily" and not self.tavily_api_key:
            gaps.append("TAVILY_API_KEY")
        return gaps


settings = Settings.from_env()
