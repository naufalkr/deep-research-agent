import json

import pytest

from deepresearch.agents.verify import Citer, Critic, split_sentences
from deepresearch.config import Settings
from deepresearch.state import Finding, SourceIndex


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


def findings(n=2):
    return [
        Finding(f"claim {i}", f"evidence {i}", f"https://a.com/{i}", "high")
        for i in range(1, n + 1)
    ]


def index_for(fs):
    index = SourceIndex()
    for f in fs:
        index.add("t", f.source_url)
    return index


# --- critic verdicts ---


async def test_a_supported_claim_keeps_its_confidence(settings):
    fs = findings(1)
    llm = ScriptedLLM(json.dumps({"verdicts": [{"n": 1, "verdict": "supported"}]}))
    out, _ = await Critic(settings, llm=llm).run(fs, index_for(fs))
    assert out[0].verdict == "supported"
    assert out[0].confidence == "high"


async def test_an_unsupported_claim_is_demoted_not_deleted(settings):
    """The reader may still need the fact; the report just has to hedge it."""
    fs = findings(1)
    llm = ScriptedLLM(
        json.dumps(
            {"verdicts": [{"n": 1, "verdict": "unsupported", "note": "evidence is adjacent"}]}
        )
    )
    out, _ = await Critic(settings, llm=llm).run(fs, index_for(fs))
    assert len(out) == 1
    assert out[0].verdict == "unsupported"
    assert out[0].confidence == "low"
    assert "evidence is adjacent" in out[0].evidence


async def test_a_contradicted_claim_is_also_demoted(settings):
    fs = findings(1)
    llm = ScriptedLLM(json.dumps({"verdicts": [{"n": 1, "verdict": "contradicted"}]}))
    out, _ = await Critic(settings, llm=llm).run(fs, index_for(fs))
    assert out[0].confidence == "low"


async def test_findings_the_critic_skipped_are_left_untouched(settings):
    fs = findings(2)
    llm = ScriptedLLM(json.dumps({"verdicts": [{"n": 1, "verdict": "supported"}]}))
    out, _ = await Critic(settings, llm=llm).run(fs, index_for(fs))
    assert out[1].verdict == "unchecked"
    assert out[1].confidence == "high"


async def test_an_unknown_verdict_is_ignored(settings):
    fs = findings(1)
    llm = ScriptedLLM(json.dumps({"verdicts": [{"n": 1, "verdict": "dubious"}]}))
    out, _ = await Critic(settings, llm=llm).run(fs, index_for(fs))
    assert out[0].verdict == "unchecked"


async def test_an_out_of_range_index_does_not_crash(settings):
    fs = findings(1)
    llm = ScriptedLLM(json.dumps({"verdicts": [{"n": 99, "verdict": "supported"}]}))
    out, _ = await Critic(settings, llm=llm).run(fs, index_for(fs))
    assert out[0].verdict == "unchecked"


async def test_an_unparseable_reply_leaves_findings_as_they_were(settings):
    fs = findings(2)
    llm = ScriptedLLM("the model rambled")
    out, clashes = await Critic(settings, llm=llm).run(fs, index_for(fs))
    assert [f.verdict for f in out] == ["unchecked", "unchecked"]
    assert clashes == []


async def test_no_findings_means_no_critic_call(settings):
    llm = ScriptedLLM()
    out, clashes = await Critic(settings, llm=llm).run([], SourceIndex())
    assert out == [] and clashes == []
    assert llm.prompts == []


# --- contradictions ---


async def test_a_contradiction_names_both_claims(settings):
    fs = findings(2)
    llm = ScriptedLLM(
        json.dumps(
            {"verdicts": [], "contradictions": [{"n": [1, 2], "note": "4.6% vs 3.1%"}]}
        )
    )
    _, clashes = await Critic(settings, llm=llm).run(fs, index_for(fs))
    assert len(clashes) == 1
    assert clashes[0].claims == ["claim 1", "claim 2"]
    assert "4.6% vs 3.1%" in clashes[0].render()


async def test_a_contradiction_needs_two_valid_claims(settings):
    fs = findings(2)
    llm = ScriptedLLM(json.dumps({"contradictions": [{"n": [1, 99], "note": "x"}]}))
    _, clashes = await Critic(settings, llm=llm).run(fs, index_for(fs))
    assert clashes == []


# --- sentence splitting ---


def test_decimals_and_abbreviations_are_not_split():
    lines = split_sentences("The rate was 4.62%. It rose in the U.S. too.")
    assert lines == [["The rate was 4.62%.", "It rose in the U.S. too."]]


def test_an_abbreviation_before_a_capital_does_not_end_the_sentence():
    text = "U.S. Bank approved more loans, e.g. Pagaya ones. Then it stopped."
    assert split_sentences(text) == [
        ["U.S. Bank approved more loans, e.g. Pagaya ones.", "Then it stopped."]
    ]


REPORT = "## Heading\n\nOne. Two.\n- a bullet"


def test_line_structure_survives_the_round_trip():
    assert [len(x) for x in split_sentences(REPORT)] == [1, 1, 2, 1]


async def test_a_report_with_no_citations_comes_back_unchanged(settings):
    fs = findings(1)
    llm = ScriptedLLM(json.dumps({"citations": []}))
    assert await Citer(settings, llm=llm).run(REPORT, fs, index_for(fs)) == REPORT


# --- citer ---


async def test_citations_are_attached_to_the_right_sentence(settings):
    fs = findings(1)
    llm = ScriptedLLM(json.dumps({"citations": [{"s": 2, "n": [1]}]}))
    out = await Citer(settings, llm=llm).run(
        "The rate rose sharply. It stayed high.", fs, index_for(fs)
    )
    assert out == "The rate rose sharply. It stayed high. [1]"


async def test_several_findings_on_one_sentence(settings):
    fs = findings(2)
    llm = ScriptedLLM(json.dumps({"citations": [{"s": 1, "n": [2, 1]}]}))
    out = await Citer(settings, llm=llm).run("One sentence.", fs, index_for(fs))
    assert out == "One sentence. [1][2]"


async def test_the_citer_never_rewrites_the_prose(settings):
    """The map cannot express an edit, so the words are safe by construction."""
    fs = findings(1)
    llm = ScriptedLLM(json.dumps({"citations": [{"s": 3, "n": [1]}]}))
    out = await Citer(settings, llm=llm).run(REPORT, fs, index_for(fs))
    assert out.replace(" [1]", "") == REPORT


async def test_an_out_of_range_sentence_is_ignored(settings):
    fs = findings(1)
    llm = ScriptedLLM(json.dumps({"citations": [{"s": 99, "n": [1]}]}))
    out = await Citer(settings, llm=llm).run("One sentence.", fs, index_for(fs))
    assert out == "One sentence."


async def test_the_marker_is_the_findings_source_number(settings):
    """Findings are picked by position; the reader sees the source's number."""
    fs = findings(2)
    index = SourceIndex()
    for i in range(10):
        index.add("filler", f"https://filler.com/{i}")
    for f in fs:
        index.add("t", f.source_url)  # sources 11 and 12
    llm = ScriptedLLM(json.dumps({"citations": [{"s": 1, "n": [2]}]}))
    out = await Citer(settings, llm=llm).run("One sentence.", fs, index)
    assert out == "One sentence. [12]"


async def test_findings_sharing_a_page_cite_it_once(settings):
    fs = [Finding("a", "e", "https://same.com"), Finding("b", "e", "https://same.com")]
    llm = ScriptedLLM(json.dumps({"citations": [{"s": 1, "n": [1, 2]}]}))
    out = await Citer(settings, llm=llm).run("One sentence.", fs, index_for(fs))
    assert out == "One sentence. [1]"


async def test_the_citer_lists_findings_by_position_not_source_number(settings):
    fs = findings(1)
    index = SourceIndex()
    index.add("filler", "https://filler.com")
    index.add("t", fs[0].source_url)
    llm = ScriptedLLM(json.dumps({"citations": []}))
    await Citer(settings, llm=llm).run("One.", fs, index)
    assert "1. claim 1" in llm.prompts[0]
    assert "[2]" not in llm.prompts[0]


async def test_an_invented_finding_number_is_ignored(settings):
    """The citer may only point at findings it was actually given."""
    fs = findings(1)
    llm = ScriptedLLM(json.dumps({"citations": [{"s": 1, "n": [7]}]}))
    out = await Citer(settings, llm=llm).run("One sentence.", fs, index_for(fs))
    assert out == "One sentence."


async def test_an_unparseable_map_returns_the_uncited_report(settings):
    fs = findings(1)
    llm = ScriptedLLM("the model rambled")
    out = await Citer(settings, llm=llm).run("One sentence.", fs, index_for(fs))
    assert out == "One sentence."


async def test_the_citer_sees_numbered_sentences_and_findings(settings):
    fs = findings(1)
    llm = ScriptedLLM(json.dumps({"citations": []}))
    await Citer(settings, llm=llm).run("One. Two.", fs, index_for(fs))
    assert "1. One." in llm.prompts[0]
    assert "2. Two." in llm.prompts[0]
    assert "claim 1" in llm.prompts[0]


async def test_nothing_to_cite_skips_the_call(settings):
    llm = ScriptedLLM()
    assert await Citer(settings, llm=llm).run("report", [], SourceIndex()) == "report"
    assert llm.prompts == []


async def test_an_empty_report_skips_the_call(settings):
    llm = ScriptedLLM()
    fs = findings(1)
    assert await Citer(settings, llm=llm).run("   ", fs, index_for(fs)) == "   "
    assert llm.prompts == []


async def test_debate_sides_are_shown_to_the_critic(settings):
    fs = [Finding("AI cuts costs", "e", "https://a.com", stance="pro")]
    llm = ScriptedLLM(json.dumps({"verdicts": []}))
    await Critic(settings, llm=llm).run(fs, index_for(fs))
    assert "side: for" in llm.prompts[0]


async def test_findings_are_checked_in_parallel_batches(settings):
    from deepresearch.agents.verify import BATCH

    fs = findings(BATCH + 5)
    llm = ScriptedLLM(
        json.dumps({"verdicts": [{"n": 1, "verdict": "unsupported"}],
                    "contradictions": [{"n": [1, 2], "note": "a"}]}),
        json.dumps({"verdicts": [{"n": 1, "verdict": "unsupported"}],
                    "contradictions": [{"n": [1, 2], "note": "b"}]}),
    )
    out, clashes = await Critic(settings, llm=llm).run(fs, index_for(fs))
    assert len(llm.prompts) == 2
    assert len(out) == BATCH + 5
    # numbers are local to each batch and map back to the right finding
    assert [i for i, f in enumerate(out) if f.verdict == "unsupported"] == [0, BATCH]
    assert sorted(c.claims[0] for c in clashes) == ["claim 1", f"claim {BATCH + 1}"]
