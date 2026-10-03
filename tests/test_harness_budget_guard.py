"""The experiment client's budget guard sees every attempt, not every call.

``Client.chat`` makes up to ``retries + 1`` attempts. Each one is a ledger row
(``attempt``, ``final``), a metered attempt without a billed figure reserves its
worst case, and the cap is re-checked before every attempt. Offline: the
transport is an ``httpx.MockTransport``.
"""

from __future__ import annotations

import inspect
import json

import httpx
import pytest

from auto_router.cache_index import estimate_tokens
from auto_router.catalog import Catalog, ModelInfo, Prices
from auto_router.config import Provider, RouterConfig
from experiments import llm
from experiments.llm import BILLED_TO_LIST_RATIO, RESERVATION_BASIS, BudgetExceeded, Client

METERED = ModelInfo("gpt-5.6-sol", "openrouter", "openai/gpt-5.6-sol", Prices(2.0, 10.0, 0.2, 2.5))
FREE = ModelInfo("kimi-k3", "host", "moonshotai/Kimi-K3-TEE", Prices.free())
MESSAGES = [{"role": "system", "content": "Reply briefly."}, {"role": "user", "content": "Say hi. " * 50}]
MAX_TOKENS = 12000
#: What one unpriced metered attempt reserves: (prompt estimate x input + max_tokens x output) x 2.0.
RESERVE = (estimate_tokens(MESSAGES) * 2.0 + MAX_TOKENS * 10.0) * BILLED_TO_LIST_RATIO / 1e6
BILLED = 0.0123

OK_BODY = {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 120, "completion_tokens": 30, "cost": BILLED}}


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)


def client(tmp_path, responses, budget=30.0):
    """A client whose transport plays ``responses`` in order: a status code with
    a body, or an exception to raise. Returns (client, number of posts so far)."""
    config = RouterConfig(providers={"openrouter": Provider("openrouter", "https://example.invalid/api/v1"),
                                     "host": Provider("host", "https://example.invalid/v1")},
                          catalog=Catalog([METERED, FREE]))
    c = Client(config, tmp_path / "calls.jsonl", budget_usd=budget)
    posts = []
    queue = list(responses)

    def handle(request):
        posts.append(json.loads(request.content))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        status, body = item
        return httpx.Response(status, json=body)

    c.http = httpx.Client(transport=httpx.MockTransport(handle))
    return c, posts


def rows(c):
    return [json.loads(line) for line in c.ledger.read_text().splitlines()]


def per_row_max(c):
    return sum(max(r["cost_usd"], r["list_cost_usd"]) for r in rows(c))


def test_retries_stay_at_two():
    assert inspect.signature(Client.chat).parameters["retries"].default == 2


def test_500_500_200_is_three_rows_two_reservations_and_one_billed(tmp_path):
    err = {"error": {"message": "upstream"}}
    c, posts = client(tmp_path, [(500, err), (500, err), (200, OK_BODY)])
    result = c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS, tag="heldout:router:t1")
    assert result.ok and result.cost_usd == pytest.approx(BILLED)
    assert len(posts) == 3
    first, second, third = rows(c)
    for row, attempt in ((first, 0), (second, 1)):
        assert row["attempt"] == attempt and row["final"] is False and row["ok"] is False
        assert row["tag"] == "heldout:router:t1"
        assert row["cost_usd"] == 0
        assert row["cost_basis"] == RESERVATION_BASIS
        assert row["list_cost_usd"] == pytest.approx(RESERVE)
    assert third["attempt"] == 2 and third["final"] is True and third["ok"] is True
    assert third["tag"] == "heldout:router:t1"
    assert third["cost_usd"] == pytest.approx(BILLED)
    assert third["cost_basis"] == "gateway-reported cost"
    # Priced exactly as before: the list cost of the reported usage.
    assert third["list_cost_usd"] == pytest.approx((120 * 2.0 + 30 * 10.0) / 1e6)
    # The guard counts each row at its worst case.
    assert c.spent() == pytest.approx(per_row_max(c))
    assert c.spent() == pytest.approx(2 * RESERVE + BILLED)


def test_the_final_row_is_what_the_caller_saw(tmp_path):
    c, _ = client(tmp_path, [(503, {}), (200, OK_BODY)])
    result = c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS, tag="t")
    [final] = [r for r in rows(c) if r["final"]]
    assert final["cost_usd"] == pytest.approx(result.cost_usd)
    assert final["list_cost_usd"] == pytest.approx(result.list_cost_usd)
    # Billed cash over all attempts excludes reservations (their cost_usd is 0).
    assert sum(r["cost_usd"] for r in rows(c) if r["tag"] == "t") == pytest.approx(BILLED)


def test_a_cap_reached_between_attempts_stops_before_the_next_attempt(tmp_path):
    # Below the cap before attempt 0; the reservation of attempt 0 crosses it.
    c, posts = client(tmp_path, [(500, {}), (200, OK_BODY)], budget=RESERVE / 2)
    with pytest.raises(BudgetExceeded):
        c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS, tag="t")
    assert len(posts) == 1
    [row] = rows(c)
    assert row["attempt"] == 0 and row["cost_basis"] == RESERVATION_BASIS


def test_a_cap_reached_between_attempts_1_and_2_stops_before_attempt_2(tmp_path):
    c, posts = client(tmp_path, [(500, {}), (500, {}), (200, OK_BODY)], budget=1.5 * RESERVE)
    with pytest.raises(BudgetExceeded):
        c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS, tag="t")
    assert len(posts) == 2
    assert [r["attempt"] for r in rows(c)] == [0, 1]


def test_the_first_attempt_is_guarded_as_before(tmp_path):
    c, posts = client(tmp_path, [(200, OK_BODY)], budget=0.0)
    with pytest.raises(BudgetExceeded):
        c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS, tag="t")
    assert posts == [] and not c.ledger.exists()


def test_a_timeout_on_a_metered_route_is_a_reservation_row(tmp_path):
    c, _ = client(tmp_path, [httpx.ReadTimeout("slow")])
    result = c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS, tag="t", retries=0)
    assert not result.ok and result.error == "ReadTimeout"
    [row] = rows(c)
    assert row["attempt"] == 0 and row["final"] is True and row["ok"] is False
    assert row["cost_usd"] == 0 and row["cost_basis"] == RESERVATION_BASIS
    assert row["list_cost_usd"] == pytest.approx(RESERVE)
    assert c.spent() == pytest.approx(RESERVE)


def test_timeouts_on_every_attempt_reserve_every_attempt(tmp_path):
    c, posts = client(tmp_path, [httpx.ReadTimeout("slow")] * 3)
    c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS, tag="t")
    assert len(posts) == 3
    assert [(r["attempt"], r["final"]) for r in rows(c)] == [(0, False), (1, False), (2, True)]
    assert c.spent() == pytest.approx(3 * RESERVE)


def test_a_metered_answer_without_usage_is_reserved(tmp_path):
    body = {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}
    c, _ = client(tmp_path, [(200, body)])
    result = c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS, tag="t")
    assert result.ok and result.content == "hi"
    [row] = rows(c)
    assert row["final"] is True and row["cost_usd"] == 0
    assert row["cost_basis"] == RESERVATION_BASIS and row["list_cost_usd"] == pytest.approx(RESERVE)


def test_a_non_retryable_error_is_one_final_reservation(tmp_path):
    c, posts = client(tmp_path, [(400, {"error": {"message": "bad request"}})])
    c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS, tag="t")
    assert len(posts) == 1
    [row] = rows(c)
    assert row["final"] is True and row["cost_basis"] == RESERVATION_BASIS


def test_free_route_attempt_rows_stay_at_zero(tmp_path):
    free_ok = {"choices": [{"message": {"role": "assistant", "content": "hi"}}],
               "usage": {"prompt_tokens": 120, "completion_tokens": 30}}
    c, posts = client(tmp_path, [(500, {}), httpx.ReadTimeout("slow"), (200, free_ok)])
    result = c.chat(FREE, MESSAGES, max_tokens=MAX_TOKENS, tag="t")
    assert result.ok and len(posts) == 3
    assert [(r["attempt"], r["final"]) for r in rows(c)] == [(0, False), (1, False), (2, True)]
    for row in rows(c):
        assert row["cost_usd"] == 0 and row["list_cost_usd"] == 0
        assert row["cost_basis"] != RESERVATION_BASIS
    assert rows(c)[-1]["cost_basis"] == "route configured as free"
    assert c.spent() == 0


def test_the_reservation_uses_the_max_tokens_actually_sent(tmp_path):
    c, posts = client(tmp_path, [(500, {})])
    c.chat(METERED, MESSAGES, max_tokens=MAX_TOKENS, tag="t", retries=0, extra={"max_tokens": 20000})
    assert posts[0]["max_tokens"] == 20000
    expected = (estimate_tokens(MESSAGES) * 2.0 + 20000 * 10.0) * BILLED_TO_LIST_RATIO / 1e6
    assert rows(c)[0]["list_cost_usd"] == pytest.approx(expected)
