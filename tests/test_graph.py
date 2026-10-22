import asyncio
import json

import pytest

from deepresearch.agents.lead import Lead
from deepresearch.config import Settings
from deepresearch.graph import ResearchGraph
from deepresearch.state import Plan, SourceIndex
from deepresearch.tools.database import DatabaseTool
from deepresearch.tools.web import WebSearch

URL = "https://example.com/search/0"


@pytest.fixture
def settings(monkeypatch) -> Settings:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    return Settings.from_env()


class RoutingLLM:
    """Answers by step name, so parallel calls do not depend on ordering.
    A list value is replayed one reply per call; a callable gets the prompt."""

    def __init__(self, by_step: dict, delay: float = 0.0):
        self.by_step = {k: list(v) if isinstance(v, list) else v for k, v in by_step.items()}
        self.delay = delay
        self.steps: list[str] = []
        self.prompts: dict[str, list[str]] = {}
        self.systems: dict[str, str] = {}

    async def acomplete(self, prompt, **kwargs):
        step = kwargs.get("step", "")
        self.steps.append(step)
        self.prompts.setdefault(step, []).append(prompt)
        self.systems[step] = kwargs.get("system", "")
        if self.delay:
            await asyncio.sleep(self.delay)
        reply = self.by_step.get(step, "{}")
        if callable(reply):
            return reply(prompt)
        if isinstance(reply, list):
            return reply.pop(0) if reply else "{}"
        return reply


def facts_plan(complexity, n):
    return json.dumps(
        {"mode": "facts", "complexity": complexity,
         "tasks": [{"objective": f"objective {i}"} for i in range(1, n + 1)]}
    )


def debate_plan(motion="Banks should use AI to approve loans"):
    return json.dumps({"mode": "debate", "motion": motion})


def one_finding(url=URL, claim="c"):
    return json.dumps({"findings": [{"claim": claim, "evidence": "e", "source_url": url}]})


def steps(**overrides):
    base = {
        "plan": facts_plan("simple", 1),
        "plan_calls": json.dumps({"calls": [{"tool": "web", "query": "q"}]}),
        "extract": one_finding(),
        "review": '{"gaps": []}',
        "verify": json.dumps({"verdicts": [{"n": 1, "verdict": "supported"}]}),
        "synthesize": "The report.",
        "cite": json.dumps({"citations": [{"s": 1, "n": [1]}]}),
    }
    return {**base, **overrides}


def graph(settings, llm, **kw):
    """Built the production way: every child gets its own searcher."""
    return ResearchGraph(settings, llm=llm, database=DatabaseTool(settings, mock=True), **kw)


# --- the lead's plan ---


async def test_complexity_caps_the_number_of_children(settings):
    plan = await Lead(settings, llm=RoutingLLM({"plan": facts_plan("simple", 3)})).plan("q")
    assert plan.mode == "facts"
    assert len(plan.tasks) == 1


async def test_max_subagents_wins_over_complexity(settings):
    llm = RoutingLLM({"plan": facts_plan("complex", 8)})
    plan = await Lead(settings, llm=llm, max_subagents=2).plan("q")
    assert len(plan.tasks) == 2


async def test_fact_tasks_are_neutral(settings):
    plan = await Lead(settings, llm=RoutingLLM({"plan": facts_plan("moderate", 2)})).plan("q")
    assert [t.stance for t in plan.tasks] == ["neutral", "neutral"]
    assert [t.id for t in plan.tasks] == ["t1", "t2"]


async def test_a_debate_gets_one_child_per_side(settings):
    plan = await Lead(settings, llm=RoutingLLM({"plan": debate_plan("X is good")})).plan("q")
    assert plan.mode == "debate"
    assert plan.motion == "X is good"
    assert [(t.id, t.stance, t.objective) for t in plan.tasks] == [
        ("pro", "pro", "X is good"),
        ("con", "con", "X is good"),
    ]


async def test_a_debate_without_a_motion_falls_back_to_facts(settings):
    llm = RoutingLLM({"plan": json.dumps({"mode": "debate", "motion": ""})})
    plan = await Lead(settings, llm=llm).plan("the question")
    assert plan.mode == "facts"
    assert plan.tasks[0].objective == "the question"


async def test_an_unparseable_plan_still_yields_one_task(settings):
    plan = await Lead(settings, llm=RoutingLLM({"plan": "rambling"})).plan("the question")
    assert plan.mode == "facts"
    assert [t.objective for t in plan.tasks] == ["the question"]


async def test_tasks_without_an_objective_are_dropped(settings):
    reply = json.dumps({"mode": "facts", "complexity": "moderate",
                        "tasks": [{"objective": ""}, {"objective": "real"}]})
    plan = await Lead(settings, llm=RoutingLLM({"plan": reply})).plan("q")
    assert [t.objective for t in plan.tasks] == ["real"]


async def test_no_findings_yields_an_honest_non_answer(settings):
    plan = Plan(mode="facts", tasks=[])
    answer = await Lead(settings, llm=RoutingLLM({})).synthesize("q", plan, [], SourceIndex())
    assert "no usable findings" in answer


# --- facts mode ---


async def test_children_run_in_parallel_not_in_sequence(settings):
    """Three children at 0.2s a call finish well under the serial time."""
    llm = RoutingLLM(steps(plan=facts_plan("complex", 3), extract='{"findings": []}'), delay=0.2)
    start = asyncio.get_event_loop().time()
    await graph(settings, llm).run("q")
    assert asyncio.get_event_loop().time() - start < 1.6
    assert llm.steps.count("plan_calls") == 3


async def test_a_gap_triggers_a_second_round(settings):
    gaps = json.dumps({"gaps": [{"objective": "the missing part", "boundaries": "b"}]})
    llm = RoutingLLM(steps(review=[gaps, '{"gaps": []}']))
    result = await graph(settings, llm).run("q")
    assert result.rounds == 2
    assert [t.id for t in result.plan] == ["t1", "r2-1"]
    assert result.plan[1].boundaries == "b"


async def test_no_gaps_means_one_round(settings):
    result = await graph(settings, RoutingLLM(steps())).run("q")
    assert result.rounds == 1


async def test_rounds_stop_at_the_configured_ceiling(settings, monkeypatch):
    monkeypatch.setenv("MAX_RESEARCH_ROUNDS", "2")
    gaps = json.dumps({"gaps": [{"objective": "more"}]})
    result = await graph(Settings.from_env(), RoutingLLM(steps(review=gaps))).run("q")
    assert result.rounds == 2


async def test_findings_from_both_rounds_are_kept(settings):
    """Each child gets its own searcher, so round two is not starved by round one."""
    gaps = json.dumps({"gaps": [{"objective": "missing"}]})
    llm = RoutingLLM(steps(review=[gaps, '{"gaps": []}'], verify="{}"))
    assert len((await graph(settings, llm).run("q")).findings) == 2


async def test_no_findings_means_no_review(settings):
    llm = RoutingLLM(steps(extract='{"findings": []}'))
    result = await graph(settings, llm).run("q")
    assert "review" not in llm.steps
    assert result.ok is False


async def test_facts_mode_writes_a_report_not_a_verdict(settings):
    llm = RoutingLLM(steps())
    await graph(settings, llm).run("q")
    assert "case against" not in llm.systems["synthesize"].lower()
    assert "rebuttal" not in llm.steps


# --- debate mode ---


def debate_steps(**overrides):
    return steps(plan=debate_plan(), **overrides)


def by_side(pro, con, rebuttal="{}"):
    """Extract replies chosen by the brief, since both sides run at once."""

    def reply(prompt):
        if "Rebuttal round" in prompt:
            return rebuttal
        return pro if "FOR" in prompt else con

    return reply


async def test_a_debate_runs_both_sides_then_rebuttals(settings):
    llm = RoutingLLM(debate_steps(verify="{}"))
    result = await graph(settings, llm).run("should banks use AI?")
    assert result.mode == "debate"
    assert [t.id for t in result.plan] == ["pro", "con", "pro-2", "con-2"]
    assert result.rounds == 2
    assert {(f.stance, f.rebuttal) for f in result.findings} == {
        ("pro", False), ("con", False), ("pro", True), ("con", True)
    }


async def test_each_side_rebuts_the_other_sides_findings(settings):
    extract = by_side(one_finding(claim="pro point"), one_finding(claim="con point"))
    llm = RoutingLLM(debate_steps(extract=extract))
    await graph(settings, llm).run("q")
    rebuttal_briefs = [b for b in llm.prompts["plan_calls"] if "Rebuttal round" in b]
    assert any("con point" in b and "FOR" in b for b in rebuttal_briefs)
    assert any("pro point" in b and "AGAINST" in b for b in rebuttal_briefs)


async def test_debate_has_no_coverage_review(settings):
    llm = RoutingLLM(debate_steps())
    await graph(settings, llm).run("q")
    assert "review" not in llm.steps


async def test_a_side_with_nothing_to_answer_skips_its_rebuttal(settings):
    extract = by_side(one_finding(claim="pro point"), '{"findings": []}')
    llm = RoutingLLM(debate_steps(extract=extract))
    result = await graph(settings, llm).run("q")
    assert [t.id for t in result.plan] == ["pro", "con", "con-2"]


async def test_no_findings_on_either_side_means_no_rebuttal_round(settings):
    llm = RoutingLLM(debate_steps(extract='{"findings": []}'))
    result = await graph(settings, llm).run("q")
    assert result.rounds == 1
    assert llm.steps.count("plan_calls") == 2


async def test_the_verdict_sees_the_motion_and_both_sides(settings):
    extract = by_side(one_finding(claim="pro point"), one_finding(claim="con point"))
    llm = RoutingLLM(debate_steps(extract=extract, verify="{}"))
    await graph(settings, llm).run("q")
    prompt = llm.prompts["synthesize"][0]
    assert "Position debated: Banks should use AI to approve loans" in prompt
    assert "side: for" in prompt and "side: against" in prompt
    assert "Verdict" in llm.systems["synthesize"]


# --- checking and citing ---


async def test_the_critic_runs_before_writing_and_the_citer_after(settings):
    llm = RoutingLLM(steps())
    await graph(settings, llm).run("q")
    assert llm.steps.index("verify") < llm.steps.index("synthesize") < llm.steps.index("cite")


async def test_the_critics_verdict_reaches_the_finding(settings):
    result = await graph(settings, RoutingLLM(steps())).run("q")
    assert result.findings[0].verdict == "supported"


async def test_contradictions_are_shown_to_the_writer(settings):
    verify = json.dumps({"contradictions": [{"n": [1, 1], "note": "figures disagree"}]})
    llm = RoutingLLM(steps(verify=verify))
    result = await graph(settings, llm).run("q")
    assert "figures disagree" in result.contradictions[0].note
    assert "figures disagree" in llm.prompts["synthesize"][0]


async def test_the_report_lists_only_cited_sources(settings):
    """Five results were read; one was cited."""
    result = await graph(settings, RoutingLLM(steps())).run("q")
    assert result.answer == "The report. [1]"
    assert [s.url for s in result.sources] == [URL]
    assert "Sources:" in result.report()


async def test_a_broken_critic_still_yields_a_report(settings):
    class Boom(RoutingLLM):
        async def acomplete(self, prompt, **kwargs):
            if kwargs.get("step") == "verify":
                raise RuntimeError("critic exploded")
            return await super().acomplete(prompt, **kwargs)

    result = await graph(settings, Boom(steps())).run("q")
    assert result.answer
    assert result.findings[0].verdict == "unchecked"
    assert "critic exploded" in result.failures["critic"]


async def test_a_broken_citer_ships_the_uncited_report_with_all_sources(settings):
    class Boom(RoutingLLM):
        async def acomplete(self, prompt, **kwargs):
            if kwargs.get("step") == "cite":
                raise RuntimeError("citer exploded")
            return await super().acomplete(prompt, **kwargs)

    result = await graph(settings, Boom(steps())).run("q")
    assert result.answer == "The report."
    assert len(result.sources) == 5
    assert "citer exploded" in result.failures["citer"]


# --- failures ---


class Exploding(WebSearch):
    def search(self, query, **kw):
        raise RuntimeError("network died")


async def test_a_failing_child_is_named_in_the_result(settings):
    g = ResearchGraph(settings, llm=RoutingLLM(steps()), web=Exploding(settings),
                      database=DatabaseTool(settings, mock=True))
    result = await g.run("q")
    assert "network died" in result.failures["t1"]
    assert result.ok is False


async def test_one_failing_child_does_not_sink_the_run(settings):
    plan_calls = [
        json.dumps({"calls": [{"tool": "web", "query": "w"}]}),
        json.dumps({"calls": [{"tool": "database", "query": "d"}]}),
    ]
    llm = RoutingLLM(steps(plan=facts_plan("moderate", 2), plan_calls=plan_calls,
                           extract=one_finding("db://chinook", "survived"), verify="{}"))
    g = ResearchGraph(settings, llm=llm, web=Exploding(settings),
                      database=DatabaseTool(settings, mock=True))
    result = await g.run("q")
    assert [f.claim for f in result.findings] == ["survived"]
    assert "t1" in result.failures


async def test_a_hung_child_is_cut_off_by_the_timeout(settings, monkeypatch):
    monkeypatch.setenv("SUBAGENT_TIMEOUT_S", "0.1")
    slow = RoutingLLM(steps(), delay=1.0)
    result = await asyncio.wait_for(graph(Settings.from_env(), slow).run("q"), timeout=10)
    assert result.findings == []
    assert "timed out" in result.failures["t1"]


# --- the progress events a UI listens to ---


def recorder():
    seen = []

    async def on_event(kind, data):
        seen.append((kind, data))

    return seen, on_event


async def test_a_facts_run_reports_every_stage(settings):
    seen, on_event = recorder()
    await graph(settings, RoutingLLM(steps()), on_event=on_event).run("q")
    kinds = [k for k, _ in seen]
    assert kinds[0] == "start"
    for stage in ("planned", "child_start", "child_done", "reviewed", "verified", "writing"):
        assert stage in kinds, stage
    assert dict(seen)["planned"]["plan"].mode == "facts"


async def test_a_debate_reports_its_rebuttal_round(settings):
    seen, on_event = recorder()
    await graph(settings, RoutingLLM(debate_steps()), on_event=on_event).run("q")
    rebuttal = dict(seen)["rebuttal"]
    assert [t.id for t in rebuttal["tasks"]] == ["pro-2", "con-2"]


async def test_a_failing_child_is_reported_as_it_happens(settings):
    seen, on_event = recorder()
    g = ResearchGraph(settings, llm=RoutingLLM(steps()), web=Exploding(settings),
                      database=DatabaseTool(settings, mock=True), on_event=on_event)
    await g.run("q")
    failed = [d for k, d in seen if k == "child_failed"]
    assert failed and "network died" in failed[0]["reason"]


async def test_a_broken_listener_does_not_lose_the_research(settings):
    async def on_event(kind, data):
        raise RuntimeError("the UI fell over")

    result = await graph(settings, RoutingLLM(steps()), on_event=on_event).run("q")
    assert result.answer
