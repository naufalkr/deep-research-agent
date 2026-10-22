import json

import pytest

from deepresearch.agents.child import MAX_CALLS, Child
from deepresearch.config import Settings
from deepresearch.state import Finding, SourceIndex, SubTask
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


def calls(*pairs):
    return json.dumps({"calls": [{"tool": t, "query": q} for t, q in pairs]})


def found(*urls):
    return json.dumps(
        {"findings": [{"claim": f"claim {i}", "evidence": "e", "source_url": u, "confidence": "high"}
                      for i, u in enumerate(urls)]}
    )


def child(settings, llm, database=None, index=None):
    return Child(
        settings,
        llm=llm,
        web=WebSearch(settings, seen_urls=set()),
        database=database or DatabaseTool(settings, mock=True),
        index=index or SourceIndex(),
    )


async def test_each_tool_searches_its_own_source(settings):
    llm = ScriptedLLM(calls(("news", "a"), ("papers", "b")), found())
    await child(settings, llm).run(SubTask("t1", "objective"))
    assert "example.com/news/" in llm.prompts[1]
    assert "example.com/papers/" in llm.prompts[1]


async def test_the_database_tool_reaches_the_database(settings):
    plan = json.dumps({"calls": [{"tool": "database", "query": "count tracks", "database": "northwind"}]})
    llm = ScriptedLLM(plan, found("db://northwind"))
    index = SourceIndex()
    out = await child(settings, llm, index=index).run(SubTask("t1", "o"))
    assert "query_database" in llm.prompts[1]
    assert out[0].source_url == "db://northwind"
    assert "northwind" in str(index.sources[0])


async def test_calls_are_capped(settings):
    seen = []

    class Counting(WebSearch):
        def search(self, query, **kw):
            seen.append(query)
            return []

    llm = ScriptedLLM(calls(*[("web", f"q{i}") for i in range(9)]))
    c = Child(settings, llm=llm, web=Counting(settings), database=DatabaseTool(settings, mock=True))
    await c.run(SubTask("t1", "o"))
    assert len(seen) == MAX_CALLS


async def test_unknown_tools_are_dropped(settings):
    llm = ScriptedLLM(calls(("tiktok", "x"), ("web", "kept")), found())
    await child(settings, llm).run(SubTask("t1", "o"))
    assert "kept" in llm.prompts[1] and "tiktok" not in llm.prompts[1]


async def test_an_unparseable_plan_falls_back_to_a_web_search(settings):
    llm = ScriptedLLM("garbage", found())
    await child(settings, llm).run(SubTask("t1", "the objective"))
    assert "the objective" in llm.prompts[1]


async def test_findings_must_cite_a_url_the_child_was_shown(settings):
    llm = ScriptedLLM(calls(("web", "q")), found("https://example.com/search/0", "https://invented.com"))
    out = await child(settings, llm).run(SubTask("t1", "o"))
    assert [f.source_url for f in out] == ["https://example.com/search/0"]


async def test_findings_carry_the_task_and_its_side(settings):
    llm = ScriptedLLM(calls(("web", "q")), found("https://example.com/search/0"))
    out = await child(settings, llm).run(SubTask("pro", "banks should use AI", "pro"))
    assert out[0].subtask_id == "pro"
    assert out[0].stance == "pro"
    assert out[0].rebuttal is False


async def test_a_rebuttal_sees_the_other_side_and_is_marked(settings):
    opponent = [Finding("AI lending is biased", "e", "https://x.com", stance="con")]
    llm = ScriptedLLM(calls(("web", "q")), found("https://example.com/search/0"))
    out = await child(settings, llm).run(SubTask("pro-2", "m", "pro"), opponent)
    assert "AI lending is biased" in llm.prompts[0]
    assert "Rebuttal round" in llm.prompts[0]
    assert out[0].rebuttal is True


async def test_no_results_means_no_extract_call(settings):
    class Empty(WebSearch):
        def search(self, query, **kw):
            return []

    llm = ScriptedLLM(calls(("web", "q")))
    c = Child(settings, llm=llm, web=Empty(settings), database=DatabaseTool(settings, mock=True))
    assert await c.run(SubTask("t1", "o")) == []
    assert len(llm.prompts) == 1


async def test_an_unavailable_database_is_skipped_not_fatal(settings):
    class Broken(DatabaseTool):
        async def query(self, question, database="chinook"):
            raise DatabaseUnavailable("down")

    plan = json.dumps({"calls": [{"tool": "database", "query": "q"}, {"tool": "web", "query": "w"}]})
    llm = ScriptedLLM(plan, found("https://example.com/search/0"))
    out = await child(settings, llm, database=Broken(settings, mock=True)).run(SubTask("t1", "o"))
    assert len(out) == 1


async def test_a_malformed_extract_is_retried_once(settings):
    llm = ScriptedLLM(calls(("web", "q")), "not json", found("https://example.com/search/0"))
    out = await child(settings, llm).run(SubTask("t1", "o"))
    assert len(llm.prompts) == 3
    assert len(out) == 1


async def test_a_database_figure_survives_an_unparseable_write_up(settings):
    plan = json.dumps({"calls": [{"tool": "database", "query": "q"}]})
    llm = ScriptedLLM(plan, "not json", "still not json")
    out = await child(settings, llm).run(SubTask("t1", "o"))
    assert len(out) == 1
    assert out[0].confidence == "high"
    assert out[0].source_url == "db://chinook"


async def test_sources_are_registered_in_the_shared_index(settings):
    index = SourceIndex()
    llm = ScriptedLLM(calls(("web", "q")), found())
    await child(settings, llm, index=index).run(SubTask("t1", "o"))
    assert len(index.sources) == 5
