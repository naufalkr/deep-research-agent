from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

from ..config import Settings, settings as default_settings

CACHE_DIR = os.path.join("runs", "cache", "search")


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    published: str | None = None

    def cite(self) -> str:
        return f"{self.title} ({self.url})"


def _cache_path(kind: str, key: str) -> str:
    digest = hashlib.sha256(f"{kind}:{key}".encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{digest}.json")


class WebSearch:
    """Search wrapper that dedupes URLs across subagents and caches to disk.

    Dedup state lives here rather than in each subagent so that parallel
    subagents don't spend tokens re-reading the same page. The cache means
    repeated eval runs over the same questions don't re-spend API quota.
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

    def search(self, query: str, *, num_results: int = 5) -> list[SearchResult]:
        """Search and return relevant excerpts, not whole pages."""
        return self._fetch("search", query, num_results)

    def find_similar(self, url: str, *, num_results: int = 5) -> list[SearchResult]:
        """Pages covering the same ground as `url` - the critic's cross-check."""
        return self._fetch("similar", url, num_results)

    def _fetch(self, kind: str, key: str, num_results: int) -> list[SearchResult]:
        raw = self._cached(kind, key, num_results)
        results = [SearchResult(**r) for r in raw]
        return self._dedupe(results)

    def _cached(self, kind: str, key: str, num_results: int) -> list[dict]:
        path = _cache_path(kind, f"{key}|{num_results}")
        if self.use_cache and os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)

        raw = [asdict(r) for r in self._call(kind, key, num_results)]
        if self.use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(raw, fh)
        return raw

    def _call(self, kind: str, key: str, num_results: int) -> list[SearchResult]:
        if self.settings.search_provider == "mock":
            return [
                SearchResult(
                    title=f"[mock] {kind} result {i}",
                    url=f"https://example.com/{kind}/{i}",
                    snippet=f"[mock] snippet for {key!r}",
                )
                for i in range(num_results)
            ]

        # highlights, not full text: a whole page can be 5k+ tokens and would
        # crowd out the subagent's context for no gain.
        if kind == "search":
            resp = self._client.search_and_contents(
                key, num_results=num_results, type="auto", highlights=True
            )
        else:
            resp = self._client.find_similar_and_contents(
                key, num_results=num_results, highlights=True
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
