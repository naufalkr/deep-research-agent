from __future__ import annotations

import asyncio
import time

from . import observability
from .config import Role, Settings, settings as default_settings
from .observability import Call

_MAX_RETRIES = 4

# These models reject temperature/top_p/top_k with a 400, so they must be omitted.
_REJECTS_SAMPLING = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)

# OpenAI-compatible providers: same SDK, different base_url.
_BASE_URLS = {
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def _is_retryable(exc: Exception) -> bool:
    name = type(exc).__name__
    if any(k in name for k in ("RateLimit", "Timeout", "Connection", "APIStatus")):
        return True
    return getattr(exc, "status_code", None) in (429, 500, 502, 503, 504)


class LLMClient:
    """One client, four roles. The model is chosen by role, not by call site."""

    def __init__(
        self, provider: str | None = None, settings: Settings | None = None
    ) -> None:
        self.settings = settings or default_settings
        self.provider = (provider or self.settings.provider).lower()
        self._client = self._build_client()

    def _build_client(self):
        if self.provider == "mock":
            return None

        key = self.settings.api_key_for(self.provider)
        if not key:
            raise RuntimeError(
                f"{self.provider.upper()}_API_KEY is not set - add it to .env"
            )

        if self.provider == "groq":
            from groq import Groq

            return Groq(api_key=key)
        if self.provider == "anthropic":
            from anthropic import Anthropic

            return Anthropic(api_key=key)
        if self.provider in _BASE_URLS:
            from openai import OpenAI

            return OpenAI(api_key=key, base_url=_BASE_URLS[self.provider])

        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider!r}")

    def complete(
        self,
        prompt: str,
        *,
        role: Role = "subagent",
        system: str = "",
        agent: str = "",
        step: str = "llm",
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> str:
        model = self.settings.model_for(role)
        start = time.perf_counter()
        for attempt in range(_MAX_RETRIES + 1):
            try:
                text, ptok, ctok = self._call(
                    model, prompt, system, max_tokens, temperature
                )
                break
            except Exception as exc:  # noqa: BLE001 - _is_retryable decides
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise
        observability.tracker.record(
            Call(agent or role, step, model, ptok, ctok, time.perf_counter() - start)
        )
        return text

    async def acomplete(self, prompt: str, **kwargs) -> str:
        """Async entry point for parallel subagents. LLM calls are I/O-bound,
        so a thread avoids maintaining a second set of provider bindings."""
        return await asyncio.to_thread(self.complete, prompt, **kwargs)

    def _call(
        self, model: str, prompt: str, system: str, max_tokens: int, temperature: float
    ) -> tuple[str, int, int]:
        if self.provider == "mock":
            text = f"[mock] response ({model})"
            return text, len(prompt) // 4, len(text) // 4

        if self.provider == "anthropic":
            kwargs = {}
            if not model.startswith(_REJECTS_SAMPLING):
                kwargs["temperature"] = temperature
            resp = self._client.messages.create(
                model=model,
                system=system or None,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                **kwargs,
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            return text, resp.usage.input_tokens, resp.usage.output_tokens

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (
            resp.choices[0].message.content or "",
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
        )
