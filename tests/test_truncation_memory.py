"""A route that repeatedly runs out of output budget is avoided next time.

``test_truncation.py`` pins what happens *within* one turn: a provider-flagged
length stop is recorded as ``truncated`` and the turn retries on another route.
These tests pin what happens on the *next comparable* request, through
``outcome_memory.py``:

* success - completed answers never cause avoidance;
* truncation - repeated observed length stops, on the same category and output
  budget, move the next request to a usable route before any call is made;
* privacy - the memory and the decision record hold route names, fixed labels
  and counts, never prompt, answer, provider body or credential;
* cold start - with no observations routing is unchanged and deterministic;
* fallback - avoidance never removes the last usable route, and never moves a
  request to a route that can write *less* than the one it replaces;
* budget integrity - a stated ``max_tokens`` that is not a positive integer is
  rejected at the edge instead of being pooled with the unknown-budget bucket.

Everything is local and deterministic: the upstream is an in-process
``httpx.MockTransport``; no socket is opened and no provider is called.
"""

import json
from dataclasses import replace

import pytest

from auto_router import server
from auto_router.catalog import Catalog
from auto_router.config import Provider, RouterConfig
from auto_router.decision import ObservedOutcome
from auto_router.ledger import RoutingLedger
from auto_router.outcome_memory import (AvoidanceRule, OutcomeKey, OutcomeMemory,
                                        budget_bucket, explicit_budget)
from auto_router.router import Router

from .test_truncation import SECRET, _ask, _complete, _model, _truncated, harness  # noqa: F401

CHEAP, OTHER = "twelve-k", "other-route"


def _topic(i):
    """A distinct first message per request, so each is its own conversation."""
    return f"Request {i}: build a responsive analytics dashboard with a collapsible sidebar. " * 60


def _cheap_truncates(model):
    return _truncated(model) if model.endswith(CHEAP) else _complete(model)


def _router(models=None, policy=None, tmp_path=None):
    config = RouterConfig(
        providers={"stub": Provider("stub", "http://127.0.0.1:1/v1", cache="openai")},
        catalog=Catalog(models or [_model(CHEAP, 0.5, 2.0, 70), _model(OTHER, 1.0, 4.0, 70)]),
        policy={"success": {"evidence_discount": 0.5}, **(policy or {})})
    ledger = RoutingLedger(tmp_path / "d.jsonl") if tmp_path else RoutingLedger(None)
    return Router(config, classifier=None, ledger=ledger)


def _route(router, i=0, max_tokens=12000, now=1_000_000.0):
    return router.route([{"role": "user", "content": _topic(i)}], None, None, max_tokens, now)


def _observe(router, result, status):
    router.observe(result, ObservedOutcome(model=result.model.name, status=status,
                                           http_status=200))


# ---------------------------------------------------------------------------
# the memory itself
# ---------------------------------------------------------------------------
def test_budget_buckets_are_a_fixed_vocabulary():
    assert budget_bucket(None) == "default"
    assert budget_bucket(0) == "default"
    assert budget_bucket(True) == "default"
    assert budget_bucket(500) == "<=1024"
    assert budget_bucket(12000) == "<=16384"
    assert budget_bucket(10 ** 6) == ">65536"


def test_only_a_positive_integer_is_a_stated_budget():
    """The one predicate the bucket, the output floor and the edge all share."""
    assert explicit_budget(12000) == 12000 and explicit_budget(1) == 1
    for value in (None, 0, -5, True, False, 12000.0, 0.5, "12000", "", [], {},
                  float("inf"), float("nan")):
        assert explicit_budget(value) is None, value
    # ...and every one of those buckets as the unknown budget, which is why the
    # edge must not let a stated-but-malformed value get this far.
    for value in (12000.0, "12000", True, 0, -5):
        assert budget_bucket(value) == "default"


def test_one_truncation_is_not_a_pattern():
    memory = OutcomeMemory()
    key = OutcomeKey("design", "<=16384")
    memory.record(CHEAP, key, "truncated", now=0.0)
    assert memory.should_avoid(CHEAP, key, now=1.0) is None


def test_repeated_truncation_meets_the_rule_with_a_stated_basis():
    memory = OutcomeMemory()
    key = OutcomeKey("design", "<=16384")
    memory.record(CHEAP, key, "truncated", now=0.0)
    memory.record(CHEAP, key, "truncated", now=1.0)
    record = memory.should_avoid(CHEAP, key, now=2.0)
    assert record is not None and (record.truncated, record.observed) == (2, 2)
    assert "2 of the last 2 observed outcomes" in record.basis()


def test_successes_dilute_the_rate_below_the_threshold():
    memory = OutcomeMemory()
    key = OutcomeKey("design", "<=16384")
    for i, status in enumerate(["truncated", "ok", "ok", "truncated", "ok"]):
        memory.record(CHEAP, key, status, now=float(i))
    assert memory.should_avoid(CHEAP, key, now=10.0) is None      # 2/5 < 0.5


@pytest.mark.parametrize("status", ["upstream_error", "transport_error", "not_taken", "pending"])
def test_only_finish_statuses_count(status):
    memory = OutcomeMemory()
    key = OutcomeKey("design", "<=16384")
    for i in range(4):
        assert memory.record(CHEAP, key, status, now=float(i)) is False
    assert memory.evidence(CHEAP, key, now=5.0) is None


def test_old_observations_expire():
    memory = OutcomeMemory(AvoidanceRule(ttl_s=60.0))
    key = OutcomeKey("design", "<=16384")
    memory.record(CHEAP, key, "truncated", now=0.0)
    memory.record(CHEAP, key, "truncated", now=1.0)
    assert memory.should_avoid(CHEAP, key, now=30.0) is not None
    assert memory.should_avoid(CHEAP, key, now=120.0) is None


def test_the_rule_is_configurable_and_can_be_switched_off():
    rule = AvoidanceRule.from_config({"truncation_memory": {"enabled": False, "min_truncations": 0}})
    assert rule.enabled is False and rule.min_truncations == 1
    memory = OutcomeMemory(rule)
    key = OutcomeKey("design", "<=16384")
    memory.record(CHEAP, key, "truncated", now=0.0)
    assert memory.should_avoid(CHEAP, key, now=1.0) is None


# ---------------------------------------------------------------------------
# cold start
# ---------------------------------------------------------------------------
def test_cold_start_routes_exactly_as_before_and_deterministically():
    first, second = _router(), _router()
    a, b = _route(first), _route(second)
    assert a.model.name == b.model.name == CHEAP
    assert a.reason == b.reason
    assert a.explanation.selection.truncation_memory is None
    assert a.explanation.to_dict()["selection"]["truncation_memory"] is None


def test_a_launched_job_does_not_feed_the_memory():
    router = _router()
    job = router.route_job("write a page", steps=2)
    assert job.outcome_key is None
    _observe(router, job, "truncated")
    assert router.outcomes.stats["observations"] == 0


# ---------------------------------------------------------------------------
# success and truncation at the selection step
# ---------------------------------------------------------------------------
def test_completed_answers_never_cause_avoidance():
    router = _router()
    for i in range(5):
        result = _route(router, i)
        assert result.model.name == CHEAP
        _observe(router, result, "ok")
    assert _route(router, 99).model.name == CHEAP


def test_repeated_truncation_moves_the_next_comparable_request():
    router = _router()
    for i in range(2):
        result = _route(router, i)
        assert result.model.name == CHEAP
        _observe(router, result, "truncated")
    after = _route(router, 2)
    assert after.model.name == OTHER
    memory = after.explanation.selection.truncation_memory
    assert memory["applied"] is True and memory["route"] == CHEAP
    assert memory["fallback"] == OTHER and memory["truncated"] == 2
    assert after.reason.startswith(f"avoid {CHEAP}")
    assert any("Avoided twelve-k on observed evidence" in n for n in after.explanation.notes)


def test_a_different_output_budget_is_not_comparable():
    router = _router()
    for i in range(3):
        _observe(router, _route(router, i, max_tokens=12000), "truncated")
    assert _route(router, 5, max_tokens=12000).model.name == OTHER
    assert _route(router, 6, max_tokens=800).model.name == CHEAP


def test_the_estimate_and_the_capability_are_not_edited():
    """Avoidance is a selection input from observations; estimates stay estimates."""
    router = _router()
    cold = _route(router, 0).explanation.to_dict()
    for i in range(2):
        _observe(router, _route(router, i), "truncated")
    warm = _route(router, 3).explanation.to_dict()
    before = {c["model"]: (c["estimated_p_success"], c["capability"]) for c in cold["candidates"]}
    after = {c["model"]: (c["estimated_p_success"], c["capability"]) for c in warm["candidates"]}
    assert before == after
    assert warm["estimated_outcome"]["kind"] == "estimate"
    assert warm["estimated_outcome"]["model"] == OTHER
    assert "truncat" not in json.dumps(warm["estimated_outcome"])
    assert "truncat" not in json.dumps(warm["classification"])
    assert warm["observed_outcome"] is None


# ---------------------------------------------------------------------------
# fallback
# ---------------------------------------------------------------------------
def test_the_only_route_is_never_avoided():
    router = _router(models=[_model(CHEAP, 0.5, 2.0, 70)])
    for i in range(3):
        _observe(router, _route(router, i), "truncated")
    result = _route(router, 4)
    assert result.model.name == CHEAP
    memory = result.explanation.selection.truncation_memory
    assert memory["applied"] is False and memory["fallback"] is None
    assert any("no usable unflagged route" in n for n in result.explanation.notes)


def test_a_fallback_that_also_truncates_is_not_chosen_over_nothing():
    router = _router()
    key = OutcomeKey("general", budget_bucket(12000))
    for name in (CHEAP, OTHER):
        for t in (0.0, 1.0):
            router.outcomes.record(name, key, "truncated", now=1_000_000.0 - 10 + t)
    result = _route(router, 7)
    memory = result.explanation.selection.truncation_memory
    assert result.model.name == CHEAP and memory["applied"] is False
    assert memory["flagged_routes"] == sorted([CHEAP, OTHER])


def test_retries_inherit_the_comparable_key():
    router = _router()
    first = _route(router, 0)
    retry = router.escalate(first, availability=True)
    assert retry is not None and retry.outcome_key == first.outcome_key


# ---------------------------------------------------------------------------
# end to end through the HTTP server
# ---------------------------------------------------------------------------
def test_end_to_end_the_third_request_skips_the_truncating_route(harness):  # noqa: F811
    client, router, upstream, ledger = harness(_cheap_truncates)
    for i in range(2):
        assert _ask(client, _topic(i)).status_code == 200
    assert upstream.calls == [f"stub/{CHEAP}", f"stub/{OTHER}"] * 2
    upstream.calls.clear()
    response = _ask(client, _topic(2))
    assert response.status_code == 200
    assert upstream.calls == [f"stub/{OTHER}"]            # no wasted truncating attempt
    assert response.headers["X-Router-Model"] == OTHER
    assert response.json()["choices"][0]["finish_reason"] == "stop"


def test_end_to_end_successes_leave_routing_unchanged(harness):  # noqa: F811
    client, _router_, upstream, _ = harness(_complete)
    for i in range(4):
        assert _ask(client, _topic(i)).status_code == 200
    assert upstream.calls == [f"stub/{CHEAP}"] * 4


# ---------------------------------------------------------------------------
# privacy
# ---------------------------------------------------------------------------
def test_nothing_identifying_reaches_the_memory_or_the_ledger(harness):  # noqa: F811
    client, router, _upstream, ledger = harness(_cheap_truncates)
    for i in range(3):
        _ask(client, _topic(i))
    stored = repr(router.outcomes._seen)
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    decisions = json.dumps([d.to_dict() for d in router.decisions])
    for blob in (stored, json.dumps(records), decisions, json.dumps(router.stats)):
        assert SECRET not in blob
        assert "half a page" not in blob
        assert "analytics dashboard" not in blob
        assert "Request 1" not in blob
    for (model, key), entries in router.outcomes._seen.items():
        assert model in (CHEAP, OTHER)
        assert key.category == "general" and key.budget == "<=16384"
        assert all(status in ("ok", "truncated") and isinstance(at, float)
                   for at, status in entries)
    applied = [r["selection"]["truncation_memory"] for r in records
               if r["selection"]["truncation_memory"]]
    assert applied
    assert set(applied[0]) == {"key", "route", "basis", "observed", "truncated",
                               "flagged_routes", "applied", "fallback",
                               "output_floor_tokens", "below_output_floor"}


# ---------------------------------------------------------------------------
# the fallback's own output ceiling (REVIEW 2026-09-20 finding B1)
# ---------------------------------------------------------------------------
# ``Catalog.eligible`` filters candidates on context, vision and tools, never on
# ``max_output_tokens``. Running out of output budget is the one failure this
# rule exists to avoid, so a fallback that can write *less* than the route it
# replaces would make that failure more likely - and on the Anthropic path the
# gateway clamps the caller's own ``max_tokens`` down to the served route's
# ceiling, so the caller would silently get a smaller budget than it asked for.
BASE, SMALL, BIG = "base-route", "small-out", "big-out"


def _ceiling(name, inp, out, ceiling, cap=70):
    return replace(_model(name, inp, out, cap), max_output_tokens=ceiling)


def test_the_fallback_never_has_a_smaller_output_ceiling_than_the_flagged_route():
    """The cheaper unflagged route is skipped because it can write less."""
    router = _router(models=[_ceiling(BASE, 0.5, 2.0, 8000),
                             _ceiling(SMALL, 1.0, 4.0, 4096)])
    for i in range(2):
        result = _route(router, i)
        assert result.model.name == BASE
        _observe(router, result, "truncated")
    after = _route(router, 2)
    assert after.model.name == BASE, "routed onto a route with a smaller output ceiling"
    memory = after.explanation.selection.truncation_memory
    assert memory["applied"] is False and memory["fallback"] is None
    # Truthful evidence: SMALL was never observed, so it is not "flagged".
    assert memory["flagged_routes"] == [BASE]
    assert memory["below_output_floor"] == [SMALL]
    assert memory["output_floor_tokens"] == 12000        # the caller asked for 12000
    assert any("can write 12000 output tokens" in n for n in after.explanation.notes)


def test_a_dearer_route_is_preferred_when_the_cheaper_one_cannot_write_as_much():
    """The rule still applies - it just skips past the too-small route."""
    router = _router(models=[_ceiling(BASE, 0.5, 2.0, 32000),
                             _ceiling(SMALL, 1.0, 4.0, 4096),
                             _ceiling(BIG, 2.0, 8.0, 32000)])
    for i in range(2):
        result = _route(router, i)
        assert result.model.name == BASE
        _observe(router, result, "truncated")
    after = _route(router, 2)
    assert after.model.name == BIG, "picked the cheap route with the smaller ceiling"
    memory = after.explanation.selection.truncation_memory
    assert memory["applied"] is True and memory["fallback"] == BIG
    assert memory["below_output_floor"] == [SMALL] and memory["flagged_routes"] == [BASE]
    assert "at least 32000 output tokens" in after.reason


def test_an_explicit_budget_above_the_fallbacks_ceiling_keeps_the_policys_choice():
    """The floor is the caller's stated budget when that is the larger number."""
    router = _router(models=[_ceiling(BASE, 0.5, 2.0, 8000),
                             _ceiling(SMALL, 1.0, 4.0, 10000)])
    for i in range(2):
        _observe(router, _route(router, i, max_tokens=12000), "truncated")
    after = _route(router, 2, max_tokens=12000)
    # SMALL clears the flagged route's own 8000 ceiling but not the caller's 12000.
    assert after.model.name == BASE
    memory = after.explanation.selection.truncation_memory
    assert memory["output_floor_tokens"] == 12000
    assert memory["applied"] is False and memory["below_output_floor"] == [SMALL]


def test_without_an_explicit_budget_the_floor_is_the_flagged_routes_ceiling():
    """No stated budget, so only the route's own ceiling constrains the fallback."""
    router = _router(models=[_ceiling(BASE, 0.5, 2.0, 8000),
                             _ceiling(SMALL, 1.0, 4.0, 10000)])
    for i in range(2):
        _observe(router, _route(router, i, max_tokens=None), "truncated")
    after = _route(router, 2, max_tokens=None)
    assert after.model.name == SMALL
    memory = after.explanation.selection.truncation_memory
    assert memory["output_floor_tokens"] == 8000 and memory["applied"] is True
    assert memory["below_output_floor"] == []


def test_equal_ceilings_route_exactly_as_they_did_before_the_floor_existed():
    """Every shipped example leaves max_output_tokens at the default: no change."""
    router = _router()
    for i in range(2):
        _observe(router, _route(router, i), "truncated")
    after = _route(router, 2)
    assert after.model.name == OTHER
    memory = after.explanation.selection.truncation_memory
    assert memory["applied"] is True and memory["below_output_floor"] == []
    assert memory["output_floor_tokens"] == 32000        # the catalog default


def test_the_output_floor_adds_no_prompt_or_answer_text_to_the_record():
    router = _router(models=[_ceiling(BASE, 0.5, 2.0, 8000),
                             _ceiling(SMALL, 1.0, 4.0, 4096)])
    for i in range(2):
        _observe(router, _route(router, i), "truncated")
    memory = _route(router, 2).explanation.selection.truncation_memory
    blob = json.dumps(memory)
    assert "analytics dashboard" not in blob and "Request 1" not in blob
    assert isinstance(memory["output_floor_tokens"], int)
    assert all(n in (BASE, SMALL) for n in memory["below_output_floor"])


# ---------------------------------------------------------------------------
# an explicit budget is a positive integer, or the request is rejected (B2)
# ---------------------------------------------------------------------------
# ``budget_bucket`` maps everything that is not a positive int to ``"default"``,
# the bucket documented as "no explicit budget: the provider's own default,
# which the router does not know". Silently dropping a *stated* budget into
# that bucket pools observations the module says are not comparable, and the
# same value goes on into ``TurnRequest.output_tokens`` and the cost
# arithmetic. The edge rejects it instead.
BAD_BUDGETS = [12000.0, 0.5, "12000", "", True, False, 0, -5, [], {},
               float("inf"), float("-inf"), float("nan")]


@pytest.mark.parametrize("value", BAD_BUDGETS)
def test_an_invalid_max_tokens_is_not_an_explicit_budget(value):
    with pytest.raises(ValueError):
        server.explicit_max_tokens({"max_tokens": value})


@pytest.mark.parametrize("body", [{}, {"max_tokens": None}])
def test_an_absent_or_null_max_tokens_stays_the_unknown_budget(body):
    assert server.explicit_max_tokens(body) is None


def test_a_valid_max_tokens_passes_through_unchanged():
    assert server.explicit_max_tokens({"max_tokens": 1}) == 1
    assert server.explicit_max_tokens({"max_tokens": 12000}) == 12000


@pytest.mark.parametrize("value", [12000.0, "12000", True, 0, -5])
def test_the_openai_edge_rejects_an_invalid_budget(harness, value):  # noqa: F811
    client, router, upstream, _ = harness(_complete)
    response = client.post("/v1/chat/completions", json={
        "model": "auto", "max_tokens": value,
        "messages": [{"role": "user", "content": _topic(0)}]})
    assert response.status_code == 400, response.text
    assert "max_tokens" in response.json()["detail"]
    # Nothing was routed, nothing was called, nothing was remembered.
    assert upstream.calls == []
    assert list(router.decisions) == []
    assert router.outcomes.stats["observations"] == 0


@pytest.mark.parametrize("value", [12000.0, "12000", True, 0, -5])
def test_the_anthropic_edge_rejects_an_invalid_budget_in_its_own_envelope(harness, value):  # noqa: F811
    client, router, upstream, _ = harness(_complete)
    response = client.post("/v1/messages", json={
        "model": "auto", "max_tokens": value,
        "messages": [{"role": "user", "content": _topic(0)}]})
    assert response.status_code == 400, response.text
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["message"] == server.MAX_TOKENS_ERROR
    assert upstream.calls == [] and list(router.decisions) == []


def test_a_stated_budget_is_never_pooled_with_the_unknown_budget_population(harness):  # noqa: F811
    """The whole point of B2: a rejected request cannot flag a route."""
    client, router, upstream, _ = harness(_cheap_truncates)
    for i in range(4):
        rejected = client.post("/v1/chat/completions", json={
            "model": "auto", "max_tokens": "12000",
            "messages": [{"role": "user", "content": _topic(i)}]})
        assert rejected.status_code == 400
    assert router.outcomes.stats["keys"] == 0
    # An honest request with no budget at all still uses the "default" bucket.
    assert client.post("/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": _topic(9)}]}).status_code == 200
    assert {k.budget for _m, k in router.outcomes._seen} == {"default"}


def test_a_valid_budget_still_reaches_the_right_bucket_end_to_end(harness):  # noqa: F811
    client, router, _upstream, _ = harness(_complete)
    assert _ask(client, _topic(0)).status_code == 200
    assert [k.budget for _m, k in router.outcomes._seen] == ["<=16384"]


def test_a_non_finite_max_tokens_is_rejected_rather_than_bucketed(harness):  # noqa: F811
    """``NaN`` is not JSON-compliant output but every JSON *parser* here accepts it."""
    client, router, upstream, _ = harness(_complete)
    for literal in ("NaN", "Infinity", "-Infinity"):
        raw = json.dumps({"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
        raw = raw[:-1] + f', "max_tokens": {literal}}}'
        response = client.post("/v1/chat/completions", content=raw.encode(),
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 400, (literal, response.text)
        assert "max_tokens" in response.json()["detail"]
    assert upstream.calls == [] and router.outcomes.stats["keys"] == 0
