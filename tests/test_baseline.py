import json

import pytest

from deepresearch.agents import baseline
from deepresearch.agents.baseline import BaselineAgent, _parse_action
from deepresearch.config import Settings
from deepresearch.tools.database import DatabaseTool, DatabaseUnavailable
from deepresearch.tools.web import SearchResult, WebSearch


@pytest.fixture
def settings(monkeypatch) -> Settings:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    return Settings.from_env()


class ScriptedLLM:
    """Replays a fixed list of replies, so a run is deterministic."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def acomplete(self, prompt, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else json.dumps(
            {"action": "answer", "text": "out of script"}
        )


def agent(settings, *replies, **kwargs) -> BaselineAgent:
    return BaselineAgent(
        settings,
        llm=ScriptedLLM(*replies),
        web=WebSearch(settings),
        database=DatabaseTool(settings, mock=True),
        **kwargs,
    )


def answer(text: str) -> str:
    return json.dumps({"action": "answer", "text": text})


def search(query: str) -> str:
    return json.dumps({"action": "search", "query": query})


def query_db(question: str) -> str:
    return json.dumps({"action": "query_database", "question": question})


# --- action parsing ---


def test_plain_json_is_parsed():
    assert _parse_action('{"action": "answer", "text": "hi"}')["action"] == "answer"


def test_markdown_fences_are_stripped():
    assert _parse_action('```json\n{"action": "search", "query": "x"}\n```')[
        "query"
    ] == "x"


def test_prose_around_the_object_is_ignored():
    reply = 'Let me search.\n{"action": "search", "query": "x"}\nHope that helps.'
    assert _parse_action(reply)["action"] == "search"


def test_a_reply_with_no_object_raises():
    with pytest.raises(ValueError):
        _parse_action("I do not feel like answering today")


# --- the loop ---


async def test_answers_without_tools_when_it_can(settings):
    result = await agent(settings, answer("42")).run("what is 6 times 7?")
    assert result.answer == "42"
    assert result.ok
    assert result.steps == []


async def test_search_results_become_numbered_sources(settings):
    result = await agent(settings, search("q"), answer("done [1]")).run("question")
    assert len(result.sources) == 5
    assert result.sources[0].n == 1
    assert "[1]" in result.report()


async def test_observations_are_fed_back_to_the_model(settings):
    scripted = ScriptedLLM(search("q"), answer("done"))
    a = BaselineAgent(
        settings,
        llm=scripted,
        web=WebSearch(settings),
        database=DatabaseTool(settings, mock=True),
    )
    await a.run("question")
    assert "gathered so far" in scripted.prompts[1]


async def test_database_answers_become_a_source(settings):
    result = await agent(settings, query_db("count rows"), answer("done [1]")).run("q")
    assert result.sources[0].title.startswith("internal database")
    assert result.steps[0].action == "query_database"


async def test_the_same_url_is_not_cited_twice(settings):
    """Two searches hitting one page should share a citation number."""
    seen: set[str] = set()
    web = WebSearch(settings, seen_urls=seen)
    a = BaselineAgent(
        settings,
        llm=ScriptedLLM(search("a"), search("b"), answer("done")),
        web=web,
        database=DatabaseTool(settings, mock=True),
    )
    result = await a.run("q")
    urls = [s.url for s in result.sources]
    assert len(urls) == len(set(urls))


async def test_an_unavailable_database_does_not_end_the_run(settings):
    """Research should continue on web sources alone."""

    class Broken(DatabaseTool):
        async def query(self, question, database="chinook"):
            raise DatabaseUnavailable("server is down")

    a = BaselineAgent(
        settings,
        llm=ScriptedLLM(query_db("x"), answer("carried on")),
        web=WebSearch(settings),
        database=Broken(settings, mock=True),
    )
    result = await a.run("q")
    assert result.answer == "carried on"
    assert "unavailable" in result.steps[0].result


async def test_malformed_json_is_reported_back_and_retried(settings):
    result = await agent(settings, "not json at all", answer("recovered")).run("q")
    assert result.answer == "recovered"


async def test_unknown_actions_are_reported_back(settings):
    reply = json.dumps({"action": "dance"})
    result = await agent(settings, reply, answer("recovered")).run("q")
    assert result.answer == "recovered"


async def test_running_out_of_steps_still_returns_an_answer(settings):
    a = agent(settings, search("a"), search("b"), max_steps=2)
    result = await a.run("q")
    assert result.answer
    assert result.ok is False


async def test_step_budget_is_flagged_in_the_result(settings):
    a = agent(settings, *[search("x")] * 5, max_steps=3)
    assert (await a.run("q")).ok is False


async def test_report_lists_the_sources(settings):
    result = await agent(settings, search("q"), answer("body")).run("q")
    report = result.report()
    assert "Sources:" in report
    assert "https://example.com" in report


async def test_report_omits_the_source_block_when_there_are_none(settings):
    result = await agent(settings, answer("body")).run("q")
    assert "Sources:" not in result.report()
