import pytest

from deepresearch.observability import BudgetExceeded, Call, Tracker


def call(agent="lead", model="claude-opus-5", pt=1_000_000, ct=0, latency=1.0) -> Call:
    return Call(agent, "step", model, pt, ct, latency)


def test_cost_uses_the_pricing_table():
    # claude-opus-5 is $5.00 per 1M input tokens
    assert call(pt=1_000_000, ct=0).cost_usd == pytest.approx(5.00)
    assert call(pt=0, ct=1_000_000).cost_usd == pytest.approx(25.00)


def test_unknown_model_costs_zero_but_is_flagged():
    c = call(model="some-open-model")
    assert c.cost_usd == 0.0
    assert c.priced is False

    t = Tracker()
    t.record(c)
    assert t.unpriced_models == ["some-open-model"]
    assert "no pricing data" in t.summary()


def test_budget_guard_trips_when_the_cap_is_passed():
    t = Tracker(budget_usd=1.00)
    with pytest.raises(BudgetExceeded):
        t.record(call(pt=1_000_000))  # $5.00, well over the $1 cap


def test_offending_call_is_kept_in_the_trace():
    """You need to see what blew the budget, so the call is recorded first."""
    t = Tracker(budget_usd=1.00)
    with pytest.raises(BudgetExceeded):
        t.record(call(pt=1_000_000))
    assert len(t.calls) == 1


def test_zero_budget_disables_the_guard():
    t = Tracker(budget_usd=0.0)
    t.record(call(pt=10_000_000))  # $50, no exception
    assert t.total_cost == pytest.approx(50.0)


def test_totals_add_up_across_calls():
    t = Tracker()
    t.record(call(pt=1000, ct=500))
    t.record(call(pt=2000, ct=1000))
    assert t.total_tokens == 4500
    assert len(t.calls) == 2


def test_summary_groups_by_agent():
    t = Tracker()
    t.record(call(agent="lead"))
    t.record(call(agent="researcher:1"))
    t.record(call(agent="researcher:1"))
    out = t.summary()
    assert "lead" in out and "researcher:1" in out
    assert "TOTAL" in out


def test_reset_clears_calls_and_can_set_budget():
    t = Tracker()
    t.record(call())
    t.reset(budget_usd=2.0)
    assert t.calls == []
    assert t.budget_usd == 2.0
