from auto_router.economics import SuccessModel
from auto_router.policies import (Context, Conversation, EscalatePolicy, EVSwitchPolicy, ExpectedCostPolicy,
                                  NaivePolicy, StaticPolicy, TurnRequest)
from auto_router.quota import QuotaDecision
from tests.helpers import CHEAP, FRONTIER, MID, SUB, catalog

NOW = 10_000.0


def ctx(*models, quota=None):
    return Context(catalog(*models), SuccessModel(), quota or {})


def req(d, prompt=40_000, now=NOW, **kw):
    return TurnRequest(category="coding", difficulty=d, prompt_tokens=prompt, output_tokens=1500, now=now, **kw)


def test_static_picks_strongest():
    assert StaticPolicy().choose(Conversation(), req(0.1), ctx()).model == "frontier"


def test_naive_picks_cheapest_capable_per_turn():
    c = ctx()
    assert NaivePolicy().choose(Conversation(), req(0.1), c).model == "cheap"
    assert NaivePolicy().choose(Conversation(), req(0.7), c).model == "frontier"


def test_ev_switch_escalates_immediately_but_refuses_a_risky_downgrade():
    c = ctx()
    conv = Conversation()
    conv.record_call(FRONTIER, 40_000, 1500, NOW - 10)
    # moderately easy turn: the mid model is the naive target but the frontier cache is warm
    assert NaivePolicy().choose(Conversation(), req(0.5), c).model == "mid"
    choice = EVSwitchPolicy().choose(conv, req(0.5, stakes_usd=5.0), c)
    assert choice.model == "frontier", choice.reason


def test_escalate_policy_downgrades_once_the_cache_expired():
    c = ctx()
    conv = Conversation()
    conv.record_call(FRONTIER, 40_000, 1500, NOW - 3600)
    choice = EscalatePolicy().choose(conv, req(0.1, now=NOW), c)
    assert choice.model == "cheap"
    assert "expired" in choice.reason


def test_failure_raises_the_conversation_floor_and_escalates():
    c = ctx()
    conv = Conversation()
    policy = EscalatePolicy()
    first = policy.choose(conv, req(0.3), c)
    conv.record_call(c.catalog[first.model], 40_000, 1500, NOW)
    retry = policy.on_failure(conv, req(0.3), c, first.model, {first.model})
    assert retry and c.catalog[retry.model].cap("coding") > c.catalog[first.model].cap("coding")
    assert conv.floor > 0.3
    # the next follow-up turn starts above the original cheap choice
    nxt = policy.choose(conv, req(0.3, now=NOW + 30, follow_up=1.0), c)
    assert c.catalog[nxt.model].cap("coding") > c.catalog[first.model].cap("coding")


def test_closed_subscription_is_never_chosen():
    closed = {"claude": QuotaDecision(False, 1.0, 0.9, "closed")}
    choice = ExpectedCostPolicy().choose(Conversation(), req(0.8), ctx(CHEAP, MID, FRONTIER, SUB, quota=closed))
    assert choice.model != "sub"


def test_open_free_subscription_wins_hard_turns():
    open_ = {"claude": QuotaDecision(True, 0.0, 0.2, "open")}
    choice = ExpectedCostPolicy().choose(Conversation(), req(0.8), ctx(CHEAP, MID, FRONTIER, SUB, quota=open_))
    assert choice.model == "sub"


def test_expected_cost_prefers_cheap_on_trivial_low_stakes_turns():
    choice = ExpectedCostPolicy().choose(Conversation(), req(0.05, stakes_usd=0.05), ctx())
    assert choice.model == "cheap"


def test_expected_cost_pays_for_capability_when_stakes_are_high():
    choice = ExpectedCostPolicy().choose(Conversation(), req(0.6, stakes_usd=20.0, detect_prob=0.2), ctx())
    assert choice.model == "frontier"
