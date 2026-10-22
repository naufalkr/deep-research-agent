import json
import os

import pytest

from deepresearch.config import Settings
from deepresearch.tools import web
from deepresearch.tools.web import SearchResult, WebSearch


@pytest.fixture
def mock_settings(monkeypatch) -> Settings:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    return Settings.from_env()


def test_search_returns_results(mock_settings):
    results = WebSearch(mock_settings).search("anything", num_results=3)
    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)


def test_urls_seen_once_are_not_returned_again(mock_settings):
    """Parallel subagents shouldn't spend tokens re-reading the same page."""
    searcher = WebSearch(mock_settings)
    first = searcher.search("query", num_results=3)
    second = searcher.search("query", num_results=3)
    assert len(first) == 3
    assert second == []


def test_dedup_state_is_shared_across_subagents(mock_settings):
    seen: set[str] = set()
    a = WebSearch(mock_settings, seen_urls=seen)
    b = WebSearch(mock_settings, seen_urls=seen)

    a.search("query", num_results=3)
    assert b.search("query", num_results=3) == []


def test_separate_searchers_do_not_share_state_by_default(mock_settings):
    a = WebSearch(mock_settings)
    b = WebSearch(mock_settings)
    a.search("query", num_results=2)
    assert len(b.search("query", num_results=2)) == 2


def test_results_are_cached_to_disk(mock_settings):
    WebSearch(mock_settings).search("cache me", num_results=2)
    files = os.listdir(web.CACHE_DIR)
    assert len(files) == 1
    with open(os.path.join(web.CACHE_DIR, files[0]), encoding="utf-8") as fh:
        assert len(json.load(fh)) == 2


def test_cache_prevents_a_second_api_call(mock_settings, monkeypatch):
    """Re-running an eval over the same questions must not re-spend quota."""
    WebSearch(mock_settings).search("same question", num_results=2)

    calls = {"n": 0}

    def counted(self, tool, query, num_results):
        calls["n"] += 1
        return []

    monkeypatch.setattr(WebSearch, "_call", counted)
    results = WebSearch(mock_settings).search("same question", num_results=2)

    assert calls["n"] == 0
    assert len(results) == 2


def test_cache_key_includes_result_count(mock_settings):
    searcher = WebSearch(mock_settings)
    searcher.search("q", num_results=2)
    searcher.search("q", num_results=5)
    assert len(os.listdir(web.CACHE_DIR)) == 2


def test_cache_can_be_turned_off(mock_settings):
    WebSearch(mock_settings, use_cache=False).search("q", num_results=2)
    assert not os.path.isdir(web.CACHE_DIR)


def test_each_tool_is_cached_separately(mock_settings):
    for tool in ("web", "news", "papers"):
        WebSearch(mock_settings).search("same query", num_results=2, tool=tool)
    assert len(os.listdir(web.CACHE_DIR)) == 3


def test_tool_results_come_from_their_own_source(mock_settings):
    results = WebSearch(mock_settings).search("q", num_results=1, tool="news")
    assert "/news/" in results[0].url


def test_an_unknown_tool_is_rejected(mock_settings):
    with pytest.raises(ValueError):
        WebSearch(mock_settings).search("q", tool="tiktok")


def test_web_searches_keep_their_original_cache_keys(mock_settings):
    """Results cached before the news and papers tools existed stay valid."""
    WebSearch(mock_settings).search("q", num_results=2)
    assert os.path.isfile(web._cache_path("search", "q|2"))


def test_missing_exa_key_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "exa")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="EXA_API_KEY"):
        WebSearch(Settings.from_env())


def test_unsupported_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "altavista")
    with pytest.raises(ValueError):
        WebSearch(Settings.from_env())


def test_highlights_become_the_snippet():
    r = web._to_result(
        type("R", (), {"url": "u", "title": "t", "highlights": ["a", "b"]})()
    )
    assert r.snippet == "a ... b"


def test_falls_back_to_text_when_there_are_no_highlights():
    r = web._to_result(
        type("R", (), {"url": "u", "title": "t", "highlights": [], "text": "body"})()
    )
    assert r.snippet == "body"


def test_missing_title_gets_a_placeholder():
    r = web._to_result(type("R", (), {"url": "u", "title": None, "highlights": []})())
    assert r.title == "(untitled)"
