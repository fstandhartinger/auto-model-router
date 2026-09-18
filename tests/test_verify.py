"""Verify-and-escalate: who gets checked, what a verdict does, what it costs.

The gate is the interesting part. A judge that is asked about answers it cannot
grade produces false alarms, and a false alarm is a second call nobody needed -
so most of these tests are about the cases where the judge is *not* asked.
"""

import pytest

from auto_router import verify
from auto_router.catalog import Catalog
from auto_router.config import Provider, RouterConfig
from auto_router.jev import Classification, Judgement
from auto_router.policies import Context, Conversation, ExpectedCostPolicy, TurnRequest
from auto_router.router import Router
from auto_router.economics import SuccessModel
from tests.helpers import CHEAP, FRONTIER, MID, SUB, model


def stub_classifier(category="coding", difficulty=0.1, long_context=0.0):
    def classify(text, context):
        return Classification(category, {}, difficulty, 1.0, 0.9, 0.0, long_context, 0.2, 0.3, 0.01)
    return classify


def make_router(judge=None, policy=None, **kw):
    cfg = RouterConfig(providers={"p": Provider("p", "https://example.invalid/v1")},
                       catalog=Catalog([CHEAP, MID, FRONTIER]), policy=kw.pop("policy_config", {}))
    return Router(cfg, policy=policy or ExpectedCostPolicy(), quota_reader=lambda: {},
                  classifier=stub_classifier(**kw), judge=judge)


def judged(p, failure="wrong"):
    calls = []

    def judge(request, response, **kw):
        calls.append((request, response, kw))
        return Judgement(p_adequate=p, latency_s=0.6, failure=failure)
    judge.calls = calls
    return judge


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def test_a_cheap_route_is_checked_and_a_frontier_route_is_not():
    policy = verify.VerifyPolicy()
    assert policy.applies(CHEAP, "coding")[0]
    applies, why = policy.applies(FRONTIER, "coding")
    assert not applies and "above the verified cheap tier" in why


def test_a_free_frontier_route_is_still_not_checked():
    """Price alone would wave it through; the capability ceiling does not."""
    free_frontier = model("free-frontier", 0.0, 0.0, cap=80.0)
    assert not verify.VerifyPolicy().applies(free_frontier, "coding")[0]


def test_a_subscription_route_is_not_checked():
    assert not verify.VerifyPolicy().applies(SUB, "coding")[0]


def test_the_judge_is_not_asked_about_a_document_it_cannot_see():
    policy = verify.VerifyPolicy()
    applies, why = policy.applies(CHEAP, "long_context")
    assert not applies and "cannot see the document" in why
    assert not policy.applies(CHEAP, "coding", needs_long_context=True)[0]
    assert not policy.applies(CHEAP, "coding", request_chars=20_000)[0]


def test_without_a_judge_nothing_is_checked():
    assert not verify.VerifyPolicy().applies(CHEAP, "coding", judge_available=False)[0]
    assert not verify.VerifyPolicy(enabled=False).applies(CHEAP, "coding")[0]


def test_always_and_never_lists_win_over_the_price_rule():
    policy = verify.VerifyPolicy(always=("frontier",), never=("cheap",))
    assert policy.applies(FRONTIER, "coding")[0]
    assert not policy.applies(CHEAP, "coding")[0]


def test_config_reads_thresholds_without_losing_the_defaults():
    policy = verify.VerifyPolicy.from_config(
        {"verify": {"thresholds": {"math": 0.5}, "skip_categories": ["long_context", "design"]}})
    assert policy.threshold("math") == 0.5
    assert policy.threshold("coding") == verify.DEFAULT_THRESHOLDS["coding"]
    assert policy.threshold("knowledge") == verify.DEFAULT_THRESHOLDS["default"]
    assert policy.skip_categories == ("long_context", "design")


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------
def test_a_low_score_escalates_and_a_high_one_does_not():
    policy = verify.VerifyPolicy()
    low = verify.verdict_from(Judgement(0.05, 0.6, failure="wrong"), policy, "coding", "cheap")
    assert low.escalate and low.failure == "wrong"
    high = verify.verdict_from(Judgement(0.93, 0.6, failure="fine"), policy, "coding", "cheap")
    assert not high.escalate
    assert "adequate (0.93)" in high.chip()


def test_an_unreachable_judge_never_escalates():
    verdict = verify.verdict_from(Judgement(0.5, 0.0, failed=True), verify.VerifyPolicy(),
                                  "coding", "cheap")
    assert verdict.verified and not verdict.escalate and verdict.judge_failed
    assert verdict.p_adequate is None


def test_the_second_attempt_is_clean_unless_carrying_is_configured():
    messages = [{"role": "user", "content": "add two numbers"}]
    verdict = verify.Verdict(verified=True, escalate=True, failure="incomplete")
    assert verify.retry_messages(messages, "def add(", verdict, verify.VerifyPolicy()) == messages
    carried = verify.retry_messages(messages, "def add(", verdict,
                                    verify.VerifyPolicy(carry_failed_attempt=True))
    assert len(carried) == 2 and "is incomplete" in carried[-1]["content"]
    assert "def add(" in carried[-1]["content"]


# ---------------------------------------------------------------------------
# the router
# ---------------------------------------------------------------------------
def test_a_rejected_answer_escalates_and_raises_the_difficulty_floor():
    judge = judged(0.02)
    r = make_router(judge=judge)
    convo = [{"role": "user", "content": "write a function that returns the median"}]
    first = r.route(convo, None, None, now=1000.0)
    assert first.model.name == "cheap"

    verdict = r.check(first, "write a function that returns the median", "def median(): pass")
    assert verdict.verified and verdict.escalate
    assert judge.calls[0][2]["category"] == "coding"

    retry, messages, verdict = r.escalate_after_verdict(first, verdict, convo, "def median(): pass")
    assert retry is not None and retry.model.cap("coding") > first.model.cap("coding")
    assert verdict.escalated_to == retry.model.name
    assert messages == convo                       # clean retry by default
    assert r.conversations[first.conversation_id].floor >= verify.DEFAULT_THRESHOLDS["default"]
    assert first.explanation.to_dict()["verification"]["escalate"] is True
    assert retry.headers["X-Router-Verified"] == "0.02"


def test_an_accepted_answer_changes_nothing():
    r = make_router(judge=judged(0.95, failure="fine"))
    convo = [{"role": "user", "content": "write a function that returns the median"}]
    first = r.route(convo, None, None, now=1000.0)
    verdict = r.check(first, convo[0]["content"], "def median(xs): ...")
    assert not verdict.escalate
    assert r.escalate_after_verdict(first, verdict, convo, "x")[0] is None
    assert r.conversations[first.conversation_id].floor == 0.0


def test_a_turn_the_judge_may_not_see_is_reported_as_not_checked():
    judge = judged(0.01)
    r = make_router(judge=judge, category="long_context", long_context=1.0)
    convo = [{"role": "user", "content": "what does the attached contract say about notice periods?"}]
    result = r.route(convo, None, None, now=1000.0)
    verdict = r.check(result, convo[0]["content"], "it says nothing")
    assert not verdict.verified and not verdict.escalate
    assert "cannot see the document" in verdict.reason
    assert judge.calls == []
    assert result.explanation.to_dict()["verification"]["verified"] is False


def test_a_judge_that_raises_does_not_break_the_turn():
    def broken(request, response, **kw):
        raise RuntimeError("judge exploded")

    r = make_router(judge=broken)
    convo = [{"role": "user", "content": "write a function that returns the median"}]
    result = r.route(convo, None, None, now=1000.0)
    verdict = r.check(result, convo[0]["content"], "def median(): pass")
    assert not verdict.verified and not verdict.escalate


def test_no_judge_configured_means_no_check_and_no_escalation():
    r = make_router(judge=None)
    convo = [{"role": "user", "content": "write a function that returns the median"}]
    result = r.route(convo, None, None, now=1000.0)
    verdict = r.check(result, convo[0]["content"], "def median(): pass")
    assert not verdict.verified and "no judge configured" in verdict.reason


# ---------------------------------------------------------------------------
# the cost model
# ---------------------------------------------------------------------------
def _ctx():
    return Context(Catalog([CHEAP, MID, FRONTIER]), SuccessModel(evidence_discount=0.0))


def _req(**kw):
    return TurnRequest(category="coding", difficulty=0.6, prompt_tokens=2000, output_tokens=600,
                       now=1000.0, stakes_usd=2.0, detect_prob=0.5, **kw)


def test_the_judge_makes_a_cheap_route_cheaper_in_expectation():
    """Catching a failure for $0.0004 is worth more than it costs."""
    ctx, req, conv = _ctx(), _req(), Conversation()
    plain = ExpectedCostPolicy()
    checked = ExpectedCostPolicy(verify=verify.VerifyPolicy(), judge_available=True)
    unchecked_value, _ = plain.value(CHEAP, conv, req, ctx, req.difficulty, [CHEAP, MID, FRONTIER])
    checked_value, _ = checked.value(CHEAP, conv, req, ctx, req.difficulty, [CHEAP, MID, FRONTIER])
    assert checked_value < unchecked_value
    # ... and the frontier route, which is never checked, is priced identically.
    assert (plain.value(FRONTIER, conv, req, ctx, req.difficulty, [CHEAP, MID, FRONTIER])
            == checked.value(FRONTIER, conv, req, ctx, req.difficulty, [CHEAP, MID, FRONTIER]))


def test_a_judge_that_never_catches_anything_only_adds_its_own_cost():
    ctx, req, conv = _ctx(), _req(), Conversation()
    useless = verify.VerifyPolicy(catch_rate={"default": 0.0}, false_flag_rate={"default": 0.0})
    plain = ExpectedCostPolicy()
    checked = ExpectedCostPolicy(verify=useless, judge_available=True)
    before, _ = plain.value(CHEAP, conv, req, ctx, req.difficulty, [CHEAP, MID, FRONTIER])
    after, _ = checked.value(CHEAP, conv, req, ctx, req.difficulty, [CHEAP, MID, FRONTIER])
    assert after == pytest.approx(before + useless.judge_usd, rel=1e-6)


def test_false_alarms_are_charged_for():
    ctx, req, conv = _ctx(), _req(), Conversation()
    honest = ExpectedCostPolicy(verify=verify.VerifyPolicy(), judge_available=True)
    jumpy = ExpectedCostPolicy(verify=verify.VerifyPolicy(false_flag_rate={"default": 0.5}),
                               judge_available=True)
    assert (jumpy.value(CHEAP, conv, req, ctx, req.difficulty, [CHEAP, MID, FRONTIER])[0]
            > honest.value(CHEAP, conv, req, ctx, req.difficulty, [CHEAP, MID, FRONTIER])[0])


def test_a_deployment_without_a_judge_prices_nothing_as_checked():
    ctx, req, conv = _ctx(), _req(), Conversation()
    configured = ExpectedCostPolicy(verify=verify.VerifyPolicy(), judge_available=False)
    assert not configured.checks(CHEAP, req, ctx)
    assert (configured.value(CHEAP, conv, req, ctx, req.difficulty, [CHEAP, MID, FRONTIER])
            == ExpectedCostPolicy().value(CHEAP, conv, req, ctx, req.difficulty, [CHEAP, MID, FRONTIER]))
