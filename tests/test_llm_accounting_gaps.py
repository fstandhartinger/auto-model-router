"""Offline regressions ported from the independent P2 review's seven observations.

F1 permission failures are inherited sandbox copytree behavior, not P2 defects.
The withdrawn identity mutation assertion is documented by the existing identity
and replay tests: changing identity keys must not change default-off routing.
No allowance, installer or product changes belong in this repair.
"""
import httpx
import pytest

from experiments import llm
from auto_router.cache_index import estimate_tokens
from tests.test_harness_budget_guard import (
    client, rows, METERED, FREE, MESSAGES, MAX_TOKENS, OK_BODY, RESERVE,
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)


@pytest.mark.parametrize("body", [[], {
    "choices": [{"message": {"content": "hi"}}],
    "usage": {"prompt_tokens": "unknown"},
}])
def test_completed_post_processing_failure_reserves_once_per_attempt(tmp_path, body):
    c, posts = client(tmp_path, [(200, body)] * 3)
    result = c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS)
    assert not result.ok
    assert len(posts) == len(rows(c)) == 3
    assert [(r["attempt"], r["final"]) for r in rows(c)] == [(0, False), (1, False), (2, True)]
    assert all(r["cost_basis"] == llm.RESERVATION_BASIS for r in rows(c))
    assert c.spent() == pytest.approx(3 * RESERVE)


def test_truthy_usage_without_accounting_reserves(tmp_path):
    body = {"choices": [{"message": {"content": "hi"}}], "usage": {"is_byok": True}}
    c, posts = client(tmp_path, [(200, body)])
    result = c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS)
    assert result.ok and len(posts) == len(rows(c)) == 1
    assert rows(c)[0]["cost_usd"] == 0
    assert rows(c)[0]["cost_basis"] == llm.RESERVATION_BASIS
    assert c.spent() == pytest.approx(RESERVE)


def test_reservation_prices_submitted_message_override(tmp_path):
    replacement = [{"role": "user", "content": "x " * 100000}]
    c, posts = client(tmp_path, [(500, {})])
    c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS, retries=0, extra={"messages": replacement})
    actual = (estimate_tokens(posts[0]["messages"]) * 2 + MAX_TOKENS * 10) * 2 / 1e6
    assert actual == pytest.approx(0.440032)
    assert rows(c)[0]["list_cost_usd"] == pytest.approx(actual)


@pytest.mark.parametrize("usage,expected", [({"cost": 0.5}, 0.5),
    ({"prompt_tokens": 200000, "completion_tokens": 0}, 0.4),
    ({"cost": 0.01}, RESERVE)])
def test_non_200_preserves_reported_accounting_and_reservation(tmp_path, usage, expected):
    c, _ = client(tmp_path, [(400, {"error": {"message": "failed after inference"}, "usage": usage})])
    result = c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS, retries=0)
    assert not result.ok and rows(c)[0]["final"]
    assert c.spent() == pytest.approx(expected)
    assert rows(c)[0]["cost_usd"] == pytest.approx(usage.get("cost", 0.4))
    assert rows(c)[0]["list_cost_usd"] >= RESERVE


def test_routed_failure_prices_known_paid_model(tmp_path):
    c, _ = client(tmp_path, [])
    c.router_catalog = {METERED.name: METERED, FREE.name: FREE}
    c.http = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(
        500, json={}, headers={"x-router-model": METERED.name})))
    c.chat(FREE, MESSAGES, max_tokens=MAX_TOKENS, retries=0)
    assert rows(c)[0]["model"] == METERED.name
    assert c.spent() == pytest.approx(RESERVE)


def test_injected_free_rule_falls_through_to_expected():
    from auto_router.router import Router
    from auto_router.policies import ExpectedCostPolicy
    from tests.paid_rule_replay import round2_config
    r = Router(round2_config(), policy=ExpectedCostPolicy(
        paid_rule_route="kimi-k3", paid_rule_min_max_tokens=10000),
        classifier=None, quota_reader=lambda: {})
    r.classifier = None
    answer = r.route([{"role": "user", "content": "hi"}], max_tokens=12000, now=1000.)
    assert answer.model.prices.is_free
    assert not answer.reason.startswith("operator rule:")
    assert "not a metered route" in answer.reason


def test_processing_failure_retry_success_has_no_duplicate_row(tmp_path):
    c, posts = client(tmp_path, [(200, []), (200, OK_BODY)])
    result = c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS)
    assert result.ok and len(posts) == len(rows(c)) == 2
    assert c.spent() == pytest.approx(RESERVE + OK_BODY["usage"]["cost"])


def test_processing_failure_guard_stops_before_retry(tmp_path):
    c, posts = client(tmp_path, [(200, []), (200, OK_BODY)], budget=RESERVE / 2)
    with pytest.raises(llm.BudgetExceeded):
        c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS)
    assert len(posts) == len(rows(c)) == 1


@pytest.mark.parametrize("body", [[], {"choices": [None]}, {"choices": [{}], "usage": {"is_byok": True}}])
def test_free_processing_failures_stay_zero(tmp_path, body):
    c, _ = client(tmp_path, [(200, body)])
    c.chat(FREE, MESSAGES, max_tokens=MAX_TOKENS, retries=0)
    assert len(rows(c)) == 1 and c.spent() == 0


def test_reservation_prices_config_extras_and_tool_override(tmp_path):
    replacement = [{"role": "user", "content": "large " * 1000}]
    tools = [{"type": "function", "function": {"name": "f", "description": "x" * 10000}}]
    c, posts = client(tmp_path, [(500, {})])
    c.extras[METERED.name] = {"messages": replacement, "tools": tools, "max_tokens": 20000}
    c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS, retries=0)
    sent = posts[0]
    expected = (estimate_tokens(sent["messages"], None, sent["tools"]) * 2 + 20000 * 10) * 2 / 1e6
    assert rows(c)[0]["list_cost_usd"] == pytest.approx(expected)


def test_unknown_routed_failure_keeps_original_envelope(tmp_path):
    c, _ = client(tmp_path, [])
    c.router_catalog = {FREE.name: FREE}
    c.http = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(
        500, json={}, headers={"x-router-model": "unknown"})))
    c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS, retries=0)
    assert rows(c)[0]["model"] == METERED.name
    assert c.spent() == pytest.approx(RESERVE)
