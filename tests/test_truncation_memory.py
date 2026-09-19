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
* fallback - avoidance never removes the last usable route.

Everything is local and deterministic: the upstream is an in-process
``httpx.MockTransport``; no socket is opened and no provider is called.
"""

import json

import pytest

from auto_router.catalog import Catalog
from auto_router.config import Provider, RouterConfig
from auto_router.decision import ObservedOutcome
from auto_router.ledger import RoutingLedger
from auto_router.outcome_memory import (AvoidanceRule, OutcomeKey, OutcomeMemory,
                                        budget_bucket)
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
                               "flagged_routes", "applied", "fallback"}
