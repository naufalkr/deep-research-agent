import json

import pytest

from deepresearch.agents.subagents import Analyst, Researcher, parse_json
from deepresearch.config import Settings
from deepresearch.state import SourceIndex, SubTask
from deepresearch.tools.database import DatabaseTool, DatabaseUnavailable
from deepresearch.tools.web import WebSearch


@pytest.fixture
def settings(monkeypatch) -> Settings:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    return Settings.from_env()


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    async def acomplete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "{}"


def finding(url, claim="a claim", confidence="high"):
    return {
        "claim": claim,
        "evidence": "some evidence",
        "source_url": url,
        "confidence": confidence,
    }


# --- parsing ---


def test_fences_and_prose_are_tolerated():
    assert parse_json('here you go:\n```json\n{"a": 1}\n```')["a"] == 1


def test_a_reply_with_no_object_raises():
    with pytest.raises(ValueError):
        parse_json("nothing here")


# --- researcher ---


async def test_planned_queries_are_used(settings):
    llm = ScriptedLLM(
        json.dumps({"queries": ["first", "second"]}),
        json.dumps({"findings": []}),
    )
    r = Researcher(settings, llm=llm, web=WebSearch(settings), index=SourceIndex())
    await r.run(SubTask("t1", "objective", "web"))
    assert "first" in llm.prompts[1] or "second" in llm.prompts[1]


async def test_queries_are_capped(settings):
    llm = ScriptedLLM(
        json.dumps({"queries": ["a", "b", "c", "d", "e"]}),
        json.dumps({"findings": []}),
    )
    web = WebSearch(settings)
    calls = []
    original = web.search
    web.search = lambda q, **kw: (calls.append(q), original(q, **kw))[1]
    r = Researcher(settings, llm=llm, web=web, index=SourceIndex())
    await r.run(SubTask("t1", "objective", "web"))
    assert len(calls) == 3


async def test_the_objective_is_the_fallback_query(settings):
    """A malformed plan must not stop the subagent doing any work."""
    llm = ScriptedLLM("not json", json.dumps({"findings": []}))
    web = WebSearch(settings)
    calls = []
    original = web.search
    web.search = lambda q, **kw: (calls.append(q), original(q, **kw))[1]
    r = Researcher(settings, llm=llm, web=web, index=SourceIndex())
    await r.run(SubTask("t1", "the objective", "web"))
    assert calls == ["the objective"]


async def test_findings_carry_the_subtask_id(settings):
    llm = ScriptedLLM(
        json.dumps({"queries": ["q"]}),
        json.dumps({"findings": [finding("https://example.com/search/0")]}),
    )
    r = Researcher(settings, llm=llm, web=WebSearch(settings), index=SourceIndex())
    out = await r.run(SubTask("t7", "objective", "web"))
    assert out[0].subtask_id == "t7"


async def test_findings_citing_an_unseen_url_are_dropped(settings):
    """A subagent must not cite a page it was never shown."""
    llm = ScriptedLLM(
        json.dumps({"queries": ["q"]}),
        json.dumps(
            {
                "findings": [
                    finding("https://example.com/search/0"),
                    finding("https://hallucinated.com"),
                ]
            }
        ),
    )
    r = Researcher(settings, llm=llm, web=WebSearch(settings), index=SourceIndex())
    out = await r.run(SubTask("t1", "objective", "web"))
    assert len(out) == 1
    assert out[0].source_url == "https://example.com/search/0"


async def test_findings_without_a_claim_are_dropped(settings):
    llm = ScriptedLLM(
        json.dumps({"queries": ["q"]}),
        json.dumps({"findings": [finding("https://example.com/search/0", claim="  ")]}),
    )
    r = Researcher(settings, llm=llm, web=WebSearch(settings), index=SourceIndex())
    assert await r.run(SubTask("t1", "o", "web")) == []


async def test_an_invalid_confidence_falls_back_to_medium(settings):
    llm = ScriptedLLM(
        json.dumps({"queries": ["q"]}),
        json.dumps(
            {"findings": [finding("https://example.com/search/0", confidence="certain")]}
        ),
    )
    r = Researcher(settings, llm=llm, web=WebSearch(settings), index=SourceIndex())
    assert (await r.run(SubTask("t1", "o", "web")))[0].confidence == "medium"


async def test_unparseable_extraction_yields_nothing_rather_than_junk(settings):
    llm = ScriptedLLM(json.dumps({"queries": ["q"]}), "the model rambled")
    r = Researcher(settings, llm=llm, web=WebSearch(settings), index=SourceIndex())
    assert await r.run(SubTask("t1", "o", "web")) == []


async def test_searched_urls_are_registered_in_the_shared_index(settings):
    index = SourceIndex()
    llm = ScriptedLLM(json.dumps({"queries": ["q"]}), json.dumps({"findings": []}))
    r = Researcher(settings, llm=llm, web=WebSearch(settings), index=index)
    await r.run(SubTask("t1", "o", "web"))
    assert len(index.sources) == 5


# --- analyst ---


async def test_analyst_asks_then_records(settings):
    llm = ScriptedLLM(
        json.dumps({"question": "count rows", "database": "northwind"}),
        json.dumps({"findings": [finding("", claim="there are 77 products")]}),
    )
    a = Analyst(settings, llm=llm, database=DatabaseTool(settings, mock=True), index=SourceIndex())
    out = await a.run(SubTask("t1", "how many products?", "database"))
    assert out[0].claim == "there are 77 products"


async def test_analyst_defaults_to_chinook_on_a_bad_plan(settings):
    llm = ScriptedLLM("garbage", json.dumps({"findings": []}))
    a = Analyst(settings, llm=llm, database=DatabaseTool(settings, mock=True), index=SourceIndex())
    await a.run(SubTask("t1", "objective", "database"))
    assert "chinook" in llm.prompts[1]


async def test_an_unavailable_database_yields_no_findings_not_an_error(settings):
    class Broken(DatabaseTool):
        async def query(self, question, database="chinook"):
            raise DatabaseUnavailable("down")

    llm = ScriptedLLM(json.dumps({"question": "q", "database": "chinook"}))
    a = Analyst(settings, llm=llm, database=Broken(settings, mock=True), index=SourceIndex())
    assert await a.run(SubTask("t1", "o", "database")) == []


async def test_the_raw_answer_survives_an_unparseable_recording(settings):
    """Losing a real database figure to a formatting slip would be worse."""
    llm = ScriptedLLM(json.dumps({"question": "q", "database": "chinook"}), "not json")
    a = Analyst(settings, llm=llm, database=DatabaseTool(settings, mock=True), index=SourceIndex())
    out = await a.run(SubTask("t1", "o", "database"))
    assert len(out) == 1
    assert out[0].confidence == "high"


async def test_the_database_is_registered_as_a_source(settings):
    index = SourceIndex()
    llm = ScriptedLLM(
        json.dumps({"question": "q", "database": "chinook"}),
        json.dumps({"findings": []}),
    )
    a = Analyst(settings, llm=llm, database=DatabaseTool(settings, mock=True), index=index)
    await a.run(SubTask("t1", "o", "database"))
    assert "chinook" in str(index.sources[0])
