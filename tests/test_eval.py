import json

import pytest

from evals.judge import Verdict, check_exact, grade
from evals.run_eval import load_cases


def test_all_questions_parse_and_are_well_formed():
    cases = load_cases(None)
    assert len(cases) >= 8
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        assert c["check"] in ("exact", "judge")
        assert c["question"] and c["expect"]


def test_exact_cases_expect_a_list_and_judge_cases_a_string():
    for c in load_cases(None):
        expected = list if c["check"] == "exact" else str
        assert isinstance(c["expect"], expected), c["id"]


def test_only_filter_matches_by_prefix():
    assert all(c["id"].startswith("db-") for c in load_cases("db-"))
    assert load_cases("nope-") == []


def test_exact_check_passes_when_every_value_is_present():
    v = check_exact("Iron Maiden leads with 213 tracks", ["Iron Maiden", "213"])
    assert v.score == 3 and v.passed


def test_exact_check_is_case_insensitive():
    assert check_exact("iron maiden", ["Iron Maiden"]).score == 3


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
