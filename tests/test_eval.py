from evals.compare import changes, table
from evals.judge import Verdict, check_exact, grade
from evals.run_eval import GROUPS, load_cases, summarise


def test_all_questions_parse_and_are_well_formed():
    cases = load_cases(None)
    assert len(cases) >= 20
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        assert c["check"] in ("exact", "judge")
        assert c["group"] in GROUPS, c["id"]
        assert c.get("mode") in (None, "facts", "debate"), c["id"]
        assert c["question"] and c["expect"]


def test_every_group_has_questions():
    groups = {c["group"] for c in load_cases(None)}
    assert groups == set(GROUPS)


def test_debate_questions_expect_debate_mode():
    for c in load_cases("debate-"):
        assert c["mode"] == "debate"


def row(id, group, score, mode=None, expected=None):
    return {"id": id, "group": group, "score": score, "cost_usd": 0.01, "seconds": 10,
            "reason": "r", "expected_mode": expected, **({"mode": mode} if mode else {})}


def test_summary_splits_by_group_and_scores_mode_choice():
    rows = [row("a", "simple", 3, "facts", "facts"), row("b", "debate", 1, "facts", "debate")]
    s = summarise(rows)
    assert s["all"] == {"n": 2, "passed": 1, "mean": 2.0, "cost_per_q": 0.01, "secs_per_q": 10}
    assert s["debate"]["mean"] == 1.0
    assert s["mode_choice"] == "1/2"


def test_baseline_runs_have_no_mode_score():
    assert "mode_choice" not in summarise([row("a", "simple", 3, expected="facts")])


def test_compare_shows_both_runs_and_what_changed():
    a = {"tag": "before", "summary": summarise([row("x", "debate", 1)]), "cases": [row("x", "debate", 1)]}
    b = {"tag": "after", "summary": summarise([row("x", "debate", 3)]), "cases": [row("x", "debate", 3)]}
    out = table(a, b)
    assert "before" in out and "after" in out
    assert "1.00 (0/1)" in out and "3.00 (1/1)" in out
    assert "`x` 1 -> 3" in changes(a, b)


def test_exact_cases_expect_a_list_and_judge_cases_a_string():
    for c in load_cases(None):
        expected = list if c["check"] == "exact" else str
        assert isinstance(c["expect"], expected), c["id"]


def test_only_filter_matches_by_prefix():
    assert all(c["id"].startswith("db-") for c in load_cases("db-"))
    assert load_cases("nope-") == []


def test_only_filter_takes_several_prefixes():
    ids = [c["id"] for c in load_cases("debate-nuclear, sources-")]
    assert ids == ["debate-nuclear-id", "sources-deepfake", "sources-lending"]


def test_exact_check_passes_when_every_value_is_present():
    v = check_exact("Iron Maiden leads with 213 tracks", ["Iron Maiden", "213"])
    assert v.score == 3 and v.passed


def test_exact_check_is_case_insensitive():
    assert check_exact("iron maiden", ["Iron Maiden"]).score == 3


def test_exact_check_accepts_any_listed_spelling():
    assert check_exact("1. United States 2. Canada", ["USA|United States", "Canada"]).score == 3
    assert check_exact("USA, Canada", ["USA|United States", "Canada"]).score == 3


def test_exact_check_names_what_is_missing():
    v = check_exact("Iron Maiden leads", ["Iron Maiden", "213"])
    assert "213" in v.reason
    assert not v.passed


def test_exact_check_scores_zero_when_nothing_matches():
    assert check_exact("no idea", ["Iron Maiden", "213"]).score == 0


def test_empty_answer_fails_without_calling_the_judge():
    case = {"check": "judge", "question": "q", "expect": "e"}
    assert grade(None, case, "   ").score == 0


def test_a_score_of_two_counts_as_a_pass():
    """Honest uncertainty should not be graded as failure."""
    assert Verdict(2, "").passed
    assert not Verdict(1, "").passed


def test_an_empty_answer_scores_zero_not_two_for_honesty():
    """Reporting nothing gracefully is still reporting nothing."""
    assert "reports nothing at all" in judge_system()
    assert "however gracefully" in judge_system()


def judge_system():
    from evals.judge import SYSTEM

    return SYSTEM


def test_the_judge_is_told_todays_date():
    """Otherwise it takes its training cutoff for today and calls recent news invented."""
    from datetime import date

    class Spy:
        prompt = ""

        def complete(self, prompt, **kwargs):
            Spy.prompt = prompt
            return '{"score": 3, "reason": "ok"}'

    grade(Spy(), {"check": "judge", "question": "q", "expect": "e"}, "an answer")
    assert date.today().isoformat() in Spy.prompt
