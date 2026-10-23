from __future__ import annotations

import asyncio
import json
import re
import time

from . import observability
from .config import Role, Settings, settings as default_settings
from .observability import Call

_MAX_RETRIES = 4
_MAX_ESCALATIONS = 2   # 3x then 9x the requested budget
_TOKEN_CEILING = 32000

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


def parse_json(text: str) -> dict:
    """Pull one JSON object out of a reply, tolerating fences and stray prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in reply: {text[:200]}")
    return json.loads(text[start : end + 1])


class TruncatedResponse(RuntimeError):
    """The model hit max_tokens: it spent the budget thinking, or ran out mid-answer."""

    def __init__(self, message: str, text: str = "", prompt_tokens: int = 0, completion_tokens: int = 0):
        super().__init__(message)
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class EmptyReply(RuntimeError):
    """The model finished without writing anything, e.g. DeepSeek under load."""

    def __init__(self, message: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, EmptyReply):
        return True
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
        budget = max_tokens
        attempt = escalations = 0
        spent_p = spent_c = 0  # truncated attempts are billed too
        while True:
            try:
                text, ptok, ctok = self._call(
                    model, prompt, system, budget, temperature
                )
                break
            except TruncatedResponse as exc:
                # How much a reasoning model thinks depends on the input, so no
                # fixed budget is right for every call. Raise it here rather
                # than guessing a number at each call site.
                spent_p += exc.prompt_tokens
                spent_c += exc.completion_tokens
                if escalations >= _MAX_ESCALATIONS:
                    if not exc.text:
                        observability.tracker.record(
                            Call(agent or role, step, model, spent_p, spent_c, time.perf_counter() - start)
                        )
                        raise
                    text, ptok, ctok = exc.text, 0, 0  # a cut-off answer beats none
                    break
                escalations += 1
                budget = min(budget * 3, _TOKEN_CEILING)
            except Exception as exc:  # noqa: BLE001 - _is_retryable decides
                spent_p += getattr(exc, "prompt_tokens", 0)
                spent_c += getattr(exc, "completion_tokens", 0)
                if _is_retryable(exc) and attempt < _MAX_RETRIES:
                    attempt += 1
                    time.sleep(min(2**attempt, 8))
                    continue
                if spent_p or spent_c:
                    observability.tracker.record(
                        Call(agent or role, step, model, spent_p, spent_c, time.perf_counter() - start)
                    )
                raise
        observability.tracker.record(
            Call(agent or role, step, model, ptok + spent_p, ctok + spent_c, time.perf_counter() - start)
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
            if not text.strip():
                raise EmptyReply(
                    f"{model} returned an empty reply ({resp.stop_reason})",
                    resp.usage.input_tokens,
                    resp.usage.output_tokens,
                )
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
        choice = resp.choices[0]
        text = choice.message.content or ""
        # Reasoning models spend max_tokens on thinking before writing, so too
        # small a budget yields nothing, or an answer cut off mid-sentence.
        if choice.finish_reason == "length":
            what = "was cut off mid-answer" if text else "returned nothing"
            raise TruncatedResponse(
                f"{model} used all {max_tokens} tokens and {what} - raise max_tokens",
                text,
                resp.usage.prompt_tokens,
                resp.usage.completion_tokens,
            )
        # An empty reply is never an answer, whatever the finish reason says.
        if not text.strip():
            raise EmptyReply(
                f"{model} returned an empty reply ({choice.finish_reason})",
                resp.usage.prompt_tokens,
                resp.usage.completion_tokens,
            )
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens
