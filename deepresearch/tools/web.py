from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

from ..config import Settings, settings as default_settings

CACHE_DIR = os.path.join("runs", "cache", "search")

# The three search tools are one engine with a different filter each.
CATEGORIES = {"web": None, "news": "news", "papers": "research paper"}


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    published: str | None = None


def _cache_path(kind: str, key: str) -> str:
    digest = hashlib.sha256(f"{kind}:{key}".encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{digest}.json")


class WebSearch:
    """Exa search with per-instance URL dedup and a disk cache.

    The cache means repeated eval runs over the same questions don't re-spend
    API quota.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        seen_urls: set[str] | None = None,
        use_cache: bool = True,
    ) -> None:
        self.settings = settings or default_settings
        self.seen_urls = seen_urls if seen_urls is not None else set()
        self.use_cache = use_cache
        self._client = self._build_client()

    def _build_client(self):
        provider = self.settings.search_provider
        if provider == "mock":
            return None
        if provider == "exa":
            if not self.settings.exa_api_key:
                raise RuntimeError("EXA_API_KEY is not set - add it to .env")
            from exa_py import Exa

            return Exa(self.settings.exa_api_key)
        raise ValueError(f"Unsupported SEARCH_PROVIDER: {provider!r}")

    def search(
        self, query: str, *, num_results: int = 5, tool: str = "web"
    ) -> list[SearchResult]:
        if tool not in CATEGORIES:
            raise ValueError(f"unknown search tool: {tool!r}")
        raw = self._cached(tool, query, num_results)
        return self._dedupe([SearchResult(**r) for r in raw])

    def _cached(self, tool: str, query: str, num_results: int) -> list[dict]:
        # "search" keeps plain web searches on the cache keys they always had
        kind = "search" if tool == "web" else tool
        path = _cache_path(kind, f"{query}|{num_results}")
        if self.use_cache and os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)

        raw = [asdict(r) for r in self._call(tool, query, num_results)]
        if self.use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(raw, fh, indent=2, ensure_ascii=False)
        return raw

    def _call(self, tool: str, query: str, num_results: int) -> list[SearchResult]:
        if self.settings.search_provider == "mock":
            kind = "search" if tool == "web" else tool
            return [
                SearchResult(
                    title=f"[mock] {tool} result {i}",
                    url=f"https://example.com/{kind}/{i}",
                    snippet=f"[mock] snippet for {query!r}",
                )
                for i in range(num_results)
            ]

        category = CATEGORIES[tool]
        # highlights, not full pages: a page can be 5k+ tokens
        resp = self._client.search(
            query,
            num_results=num_results,
            type="auto",
            contents={"highlights": True},
            **({"category": category} if category else {}),
        )
        return [_to_result(r) for r in resp.results]

    def _dedupe(self, results: list[SearchResult]) -> list[SearchResult]:
        fresh = [r for r in results if r.url not in self.seen_urls]
        self.seen_urls.update(r.url for r in fresh)
        return fresh


def _to_result(r) -> SearchResult:
    highlights = getattr(r, "highlights", None) or []
    snippet = " ... ".join(highlights) if highlights else (getattr(r, "text", "") or "")
    return SearchResult(
        title=getattr(r, "title", "") or "(untitled)",
        url=r.url,
        snippet=snippet,
        published=getattr(r, "published_date", None),
    )
