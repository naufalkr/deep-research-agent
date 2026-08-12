import asyncio
import json

import pytest

from deepresearch.agents.lead import Lead
from deepresearch.config import Settings
from deepresearch.graph import ResearchGraph
from deepresearch.state import SourceIndex
from deepresearch.tools.database import DatabaseTool
from deepresearch.tools.web import WebSearch


@pytest.fixture
def settings(monkeypatch) -> Settings:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    return Settings.from_env()


class RoutingLLM:
    """Answers by step name, so parallel calls do not depend on ordering."""

    def __init__(self, by_step: dict, delay: float = 0.0):
        self.by_step = by_step
        self.delay = delay
        self.steps = []

    async def acomplete(self, prompt, **kwargs):
        step = kwargs.get("step", "")
        self.steps.append(step)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.by_step.get(step, "{}")


def plan_reply(complexity, *tools):
    return json.dumps(
        {
            "complexity": complexity,
            "tasks": [
                {"id": f"t{i}", "objective": f"objective {i}", "tool": t}
                for i, t in enumerate(tools, 1)
            ],
        }
    )


def graph(settings, llm, **kw):
    return ResearchGraph(
        settings,
        llm=llm,
        web=WebSearch(settings, seen_urls=set()),
        database=DatabaseTool(settings, mock=True),
        **kw,
    )


# --- planning ---


async def test_complexity_caps_the_number_of_subagents(settings):
    llm = RoutingLLM({"plan": plan_reply("simple", "web", "web", "web")})
    complexity, tasks = await Lead(settings, llm=llm).plan("q")
    assert complexity == "simple"
    assert len(tasks) == 1


async def test_complex_questions_get_more_subagents(settings):
    llm = RoutingLLM({"plan": plan_reply("complex", *["web"] * 8)})
    _, tasks = await Lead(settings, llm=llm, max_subagents=6).plan("q")
    assert len(tasks) == 6


async def test_max_subagents_wins_over_complexity(settings):
    """The configured ceiling is a spend cap; a plan cannot exceed it."""
    llm = RoutingLLM({"plan": plan_reply("complex", *["web"] * 8)})
    _, tasks = await Lead(settings, llm=llm, max_subagents=2).plan("q")
    assert len(tasks) == 2


async def test_an_unparseable_plan_still_yields_one_task(settings):
    llm = RoutingLLM({"plan": "the model rambled"})
    _, tasks = await Lead(settings, llm=llm).plan("the question")
    assert len(tasks) == 1
    assert tasks[0].objective == "the question"


async def test_an_unknown_tool_falls_back_to_web(settings):
    reply = json.dumps(
        {
            "complexity": "simple",
            "tasks": [{"id": "t1", "objective": "o", "tool": "psychic"}],
        }
    )
    _, tasks = await Lead(settings, llm=RoutingLLM({"plan": reply})).plan("q")
    assert tasks[0].tool == "web"


async def test_tasks_without_an_objective_are_dropped(settings):
    reply = json.dumps(
        {
            "complexity": "moderate",
            "tasks": [
                {"id": "t1", "objective": "", "tool": "web"},
                {"id": "t2", "objective": "real", "tool": "web"},
            ],
        }
    )
    _, tasks = await Lead(settings, llm=RoutingLLM({"plan": reply})).plan("q")
    assert [t.objective for t in tasks] == ["real"]


# --- synthesis ---


async def test_no_findings_yields_an_honest_non_answer(settings):
    answer = await Lead(settings, llm=RoutingLLM({})).synthesize("q", [], SourceIndex())
    assert "no usable findings" in answer


# --- the graph ---


async def test_subagents_run_in_parallel_not_in_sequence(settings):
    """Three subagents at 0.2s a call should finish well under the serial time."""
    llm = RoutingLLM(
        {
            "plan": plan_reply("complex", "web", "web", "web"),
            "plan_queries": json.dumps({"queries": ["q"]}),
            "extract": json.dumps({"findings": []}),
            "synthesize": "report",
        },
        delay=0.2,
    )
    loop = asyncio.get_event_loop()
    start = loop.time()
    await graph(settings, llm).run("q")
    assert loop.time() - start < 1.2


async def test_both_kinds_of_subagent_are_dispatched(settings):
    llm = RoutingLLM(
        {
            "plan": plan_reply("moderate", "web", "database"),
            "plan_queries": json.dumps({"queries": ["q"]}),
            "extract": json.dumps({"findings": []}),
            "plan_query": json.dumps({"question": "q", "database": "chinook"}),
            "record": json.dumps({"findings": []}),
            "synthesize": "report",
        }
    )
    await graph(settings, llm).run("q")
    assert "extract" in llm.steps and "record" in llm.steps


async def test_a_page_two_subagents_both_find_gets_one_citation_number(settings):
    llm = RoutingLLM(
        {
            "plan": plan_reply("moderate", "web", "web"),
            "plan_queries": json.dumps({"queries": ["same query"]}),
            "extract": json.dumps({"findings": []}),
            "synthesize": "report",
        }
    )
    result = await graph(settings, llm).run("q")
    urls = [s.url for s in result.sources]
    assert len(urls) == len(set(urls))


async def test_sibling_subagents_are_not_starved_by_each_others_searches(settings):
    """Sharing one dedup set let whichever searched first take every result,
    leaving the others with nothing to extract."""
    llm = RoutingLLM(
        {
            "plan": plan_reply("complex", "web", "web", "web"),
            "plan_queries": json.dumps({"queries": ["identical query"]}),
            "extract": json.dumps({"findings": []}),
            "synthesize": "report",
        }
    )
    g = ResearchGraph(settings, llm=llm, database=DatabaseTool(settings, mock=True))
    await g.run("q")
    assert llm.steps.count("extract") == 3


async def test_a_failing_subagent_is_named_in_the_result(settings):
    class Exploding(WebSearch):
        def search(self, query, **kw):
            raise RuntimeError("network died")

    llm = RoutingLLM(
        {
            "plan": plan_reply("simple", "web"),
            "plan_queries": json.dumps({"queries": ["q"]}),
            "synthesize": "report",
        }
    )
    g = ResearchGraph(
        settings,
        llm=llm,
        web=Exploding(settings, seen_urls=set()),
        database=DatabaseTool(settings, mock=True),
    )
    result = await g.run("q")
    assert "t1" in result.failures
    assert "network died" in result.failures["t1"]
    assert result.ok is False


async def test_one_failing_subagent_does_not_sink_the_run(settings):
    class Exploding(WebSearch):
        def search(self, query, **kw):
            raise RuntimeError("network died")

    llm = RoutingLLM(
        {
            "plan": plan_reply("moderate", "web", "database"),
            "plan_queries": json.dumps({"queries": ["q"]}),
            "plan_query": json.dumps({"question": "q", "database": "chinook"}),
            "record": json.dumps(
                {"findings": [{"claim": "survived", "evidence": "e", "source_url": ""}]}
            ),
            "synthesize": "report",
        }
    )
    g = ResearchGraph(
        settings,
        llm=llm,
        web=Exploding(settings, seen_urls=set()),
        database=DatabaseTool(settings, mock=True),
    )
    result = await g.run("q")
    assert [f.claim for f in result.findings] == ["survived"]


async def test_a_hung_subagent_is_cut_off_by_the_timeout(settings, monkeypatch):
    monkeypatch.setenv("SUBAGENT_TIMEOUT_S", "0.1")
    slow = RoutingLLM(
        {
            "plan": plan_reply("simple", "web"),
            "plan_queries": json.dumps({"queries": ["q"]}),
            "extract": json.dumps({"findings": []}),
            "synthesize": "report",
        },
        delay=5.0,
    )
    g = graph(Settings.from_env(), slow)
    result = await asyncio.wait_for(g.run("q"), timeout=8)
    assert result.findings == []


async def test_report_lists_sources_and_the_answer(settings):
    llm = RoutingLLM(
        {
            "plan": plan_reply("simple", "web"),
            "plan_queries": json.dumps({"queries": ["q"]}),
            "extract": json.dumps(
                {
                    "findings": [
                        {
                            "claim": "c",
                            "evidence": "e",
                            "source_url": "https://example.com/search/0",
                        }
                    ]
                }
            ),
            "synthesize": "the report [1]",
        }
    )
    result = await graph(settings, llm).run("q")
    assert "the report [1]" in result.report()
    assert "Sources:" in result.report()
    assert result.ok


async def test_a_run_with_no_findings_is_flagged_not_ok(settings):
    llm = RoutingLLM(
        {
            "plan": plan_reply("simple", "web"),
            "plan_queries": json.dumps({"queries": ["q"]}),
            "extract": json.dumps({"findings": []}),
        }
    )
    assert (await graph(settings, llm).run("q")).ok is False


# --- coverage review and second rounds ---


def gaps_reply(*objectives):
    return json.dumps(
        {"gaps": [{"objective": o, "tool": "web", "boundaries": "b"} for o in objectives]}
    )


class SequencedLLM(RoutingLLM):
    """Like RoutingLLM, but a step can return a different reply each time."""

    def __init__(self, by_step: dict, sequences: dict):
        super().__init__(by_step)
        self.sequences = {k: list(v) for k, v in sequences.items()}

    async def acomplete(self, prompt, **kwargs):
        step = kwargs.get("step", "")
        self.steps.append(step)
        queue = self.sequences.get(step)
        if queue:
            return queue.pop(0)
        return self.by_step.get(step, "{}")


def base_steps(finding_url="https://example.com/search/0"):
    return {
        "plan": plan_reply("simple", "web"),
        "plan_queries": json.dumps({"queries": ["q"]}),
        "extract": json.dumps(
            {"findings": [{"claim": "c", "evidence": "e", "source_url": finding_url}]}
        ),
        "synthesize": "report",
    }


async def test_a_gap_triggers_a_second_round(settings):
    llm = SequencedLLM(base_steps(), {"review": [gaps_reply("the missing part"), '{"gaps": []}']})
    result = await graph(settings, llm).run("q")
    assert result.rounds == 2
    assert len(result.plan) == 2
    assert result.plan[1].id.startswith("r2-")


async def test_no_gaps_means_no_second_round(settings):
    llm = SequencedLLM(base_steps(), {"review": ['{"gaps": []}']})
    result = await graph(settings, llm).run("q")
    assert result.rounds == 1
    assert llm.steps.count("extract") == 1


async def test_rounds_stop_at_the_configured_ceiling(settings, monkeypatch):
    """A lead that always finds gaps must not loop forever."""
    monkeypatch.setenv("MAX_RESEARCH_ROUNDS", "2")
    llm = SequencedLLM(base_steps(), {"review": [gaps_reply("more") for _ in range(9)]})
    result = await graph(Settings.from_env(), llm).run("q")
    assert result.rounds == 2


async def test_findings_from_both_rounds_are_kept(settings):
    """Built the production way: each subagent gets its own searcher, so a
    round-two subagent is not starved by what round one already read."""
    llm = SequencedLLM(base_steps(), {"review": [gaps_reply("missing"), '{"gaps": []}']})
    g = ResearchGraph(settings, llm=llm, database=DatabaseTool(settings, mock=True))
    result = await g.run("q")
    assert len(result.findings) == 2


async def test_an_unparseable_review_ends_the_loop_rather_than_looping(settings):
    llm = SequencedLLM(base_steps(), {"review": ["the model rambled"]})
    assert (await graph(settings, llm).run("q")).rounds == 1


async def test_no_findings_means_no_review_call(settings):
    """With nothing gathered there is nothing to check coverage against."""
    steps = base_steps()
    steps["extract"] = json.dumps({"findings": []})
    llm = SequencedLLM(steps, {})
    result = await graph(settings, llm).run("q")
    assert "review" not in llm.steps
    assert result.rounds == 1


async def test_gap_tasks_carry_boundaries_so_round_two_does_not_repeat_itself(settings):
    llm = SequencedLLM(base_steps(), {"review": [gaps_reply("missing"), '{"gaps": []}']})
    result = await graph(settings, llm).run("q")
    assert result.plan[1].boundaries == "b"


async def test_each_lead_step_gets_its_own_prompt(settings):
    """One shared prompt let the report step answer with the review JSON."""
    from deepresearch.agents import lead as lead_mod

    seen = {}

    class PromptSpy(RoutingLLM):
        async def acomplete(self, prompt, **kwargs):
            seen[kwargs.get("step")] = kwargs.get("system", "")
            return await super().acomplete(prompt, **kwargs)

    llm = PromptSpy(
        {
            **base_steps(),
            "review": '{"gaps": []}',
        }
    )
    await graph(settings, llm).run("q")

    assert seen["plan"] is not seen["synthesize"]
    assert "complexity" in seen["plan"]
    assert "gaps" in seen["review"]
    assert "Write prose" in seen["synthesize"]
    assert "gaps" not in seen["synthesize"]


# --- phase 3: critic and citation pass ---


def verify_steps():
    return {
        **base_steps(),
        "review": '{"gaps": []}',
        "verify": json.dumps({"verdicts": [{"n": 1, "verdict": "supported"}]}),
        "cite": json.dumps({"citations": [{"s": 1, "n": [1]}]}),
    }


async def test_verification_runs_the_critic_then_the_citer(settings):
    llm = SequencedLLM(verify_steps(), {})
    await graph(settings, llm).run("q")
    assert llm.steps.index("verify") < llm.steps.index("synthesize")
    assert llm.steps.index("synthesize") < llm.steps.index("cite")


async def test_verify_off_reproduces_the_v0_2_pipeline(settings):
    """v0.2's numbers must stay reproducible, or the phase-3 delta is unreadable."""
    llm = SequencedLLM(verify_steps(), {})
    await graph(settings, llm, verify=False).run("q")
    assert "verify" not in llm.steps
    assert "cite" not in llm.steps


async def test_the_critics_verdict_reaches_the_finding(settings):
    llm = SequencedLLM(verify_steps(), {})
    result = await graph(settings, llm).run("q")
    assert result.findings[0].verdict == "supported"


async def test_contradictions_are_carried_into_the_result(settings):
    steps = verify_steps()
    steps["verify"] = json.dumps(
        {
            "verdicts": [{"n": 1, "verdict": "contradicted"}],
            "contradictions": [{"n": [1, 1], "note": "figures disagree"}],
        }
    )
    llm = SequencedLLM(steps, {})
    result = await graph(settings, llm).run("q")
    assert len(result.contradictions) == 1
    assert "figures disagree" in result.contradictions[0].note


async def test_contradictions_are_shown_to_the_writer(settings):
    steps = verify_steps()
    steps["verify"] = json.dumps(
        {"contradictions": [{"n": [1, 1], "note": "figures disagree"}]}
    )
    seen = {}

    class Spy(SequencedLLM):
        async def acomplete(self, prompt, **kwargs):
            seen[kwargs.get("step")] = prompt
            return await super().acomplete(prompt, **kwargs)

    await graph(settings, Spy(steps, {})).run("q")
    assert "figures disagree" in seen["synthesize"]


async def test_the_writer_is_told_not_to_cite(settings):
    """Citing is the citer's job; two agents doing it produces invented numbers."""
    from deepresearch.agents.lead import REPORT_PROMPT

    assert "Do not add citation markers" in REPORT_PROMPT


async def test_a_broken_critic_still_yields_a_report(settings):
    """Verification improves a report; losing it must not cost the report."""

    class Boom(SequencedLLM):
        async def acomplete(self, prompt, **kwargs):
            if kwargs.get("step") == "verify":
                raise RuntimeError("critic exploded")
            return await super().acomplete(prompt, **kwargs)

    result = await graph(settings, Boom(verify_steps(), {})).run("q")
    assert result.answer
    assert "critic exploded" in result.failures["critic"]


async def test_a_broken_citer_ships_the_uncited_report(settings):
    class Boom(SequencedLLM):
        async def acomplete(self, prompt, **kwargs):
            if kwargs.get("step") == "cite":
                raise RuntimeError("citer exploded")
            return await super().acomplete(prompt, **kwargs)

    result = await graph(settings, Boom(verify_steps(), {})).run("q")
    assert result.answer == "report"
    assert "citer exploded" in result.failures["citer"]


async def test_a_broken_critic_does_not_lose_the_findings(settings):
    class Boom(SequencedLLM):
        async def acomplete(self, prompt, **kwargs):
            if kwargs.get("step") == "verify":
                raise RuntimeError("boom")
            return await super().acomplete(prompt, **kwargs)

    result = await graph(settings, Boom(verify_steps(), {})).run("q")
    assert len(result.findings) == 1
    assert result.findings[0].verdict == "unchecked"
