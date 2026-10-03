"""The operator paid-routing rule on ``F_expected`` (``policy.paid_rule_*``).

Default-off: with the rule unset every choice must stay byte-identical to the
policy before the rule existed. ``paid_rule_regression.json`` holds the
choices the pre-rule code (e6d4f28) made on two replay sets, recorded by
running ``paid_rule_replay.replay_records()`` against that tree:

* the held-out task set (seed 20260925) on the catalog registered for the
  round-2 run (``policy_identity``: three free routes and ``gpt-5.6-sol``), on
  the classifier-disabled path that run used and with a fixed stub classifier;
* the frozen 2026-09-29 paired A/B tasks on that study's catalog
  (``evaluation/paired-router-ab-20260929/router.yaml``).

Each task is routed at its own harness budget and at no, 4000, 8000, 9999 and
12000 stated ``max_tokens``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_router.catalog import Catalog, ModelInfo, Prices
from auto_router.config import RouterConfig, load_config, validate_paid_rule
from auto_router.policies import Conversation, ExpectedCostPolicy, TurnRequest, candidates
from auto_router.router import Router
from experiments.tasks_heldout import build_tasks
from tests.paid_rule_replay import (PROVIDERS, ROUND2, ROUND2_POLICY, _router, replay_records,
                                    round2_config, stub_classifier)

GOLDEN = Path(__file__).with_name("paid_rule_regression.json")


# ---------------------------------------------------------------------------
# rule unset: byte-identical to the pre-rule policy
# ---------------------------------------------------------------------------
def test_rule_unset_every_replayed_choice_is_byte_identical_to_before():
    golden = json.loads(GOLDEN.read_text())
    now = replay_records()
    assert len(now) == len(golden) > 0
    diffs = [(g, n) for g, n in zip(golden, now) if g != n]
    assert not diffs, f"{len(diffs)} choices changed, first: {diffs[0]}"


# ---------------------------------------------------------------------------
# rule set: fires on the stated budget, and only then
# ---------------------------------------------------------------------------
RULE = {"paid_rule_route": "gpt-5.6-sol", "paid_rule_min_max_tokens": 10000}
DESIGN = next(t for t in build_tasks() if t["category"] == "design")


def _pair(max_tokens, classify=None, **rule):
    """(choice with the rule, choice of plain F_expected) for the same turn."""
    ruled = _router(round2_config(**(rule or RULE)), classify)
    plain = _router(round2_config(), classify)
    messages = [{"role": "user", "content": DESIGN["prompt"]}]
    a = ruled.route(messages, system=DESIGN["system"], max_tokens=max_tokens, now=1000.0)
    b = plain.route(messages, system=DESIGN["system"], max_tokens=max_tokens, now=1000.0)
    return a, b


def test_the_rule_is_wired_from_the_config_policy_block():
    router = _router(round2_config(**RULE), None)
    assert isinstance(router.policy, ExpectedCostPolicy)
    assert router.policy.paid_rule_route == "gpt-5.6-sol"
    assert router.policy.paid_rule_min_max_tokens == 10000
    plain = _router(round2_config(), None).policy
    assert plain.paid_rule_route is None and plain.paid_rule_min_max_tokens is None


@pytest.mark.parametrize("classify", [None, stub_classifier(DESIGN)], ids=["no-classifier", "classifier"])
def test_a_stated_12000_goes_to_the_rule_route(classify):
    ruled, plain = _pair(12000, classify)
    assert plain.model.prices.is_free              # F_expected's own choice: a free route
    assert ruled.model.name == "gpt-5.6-sol"
    assert ruled.reason == "operator rule: stated max_tokens 12000 >= 10000 -> gpt-5.6-sol"
    assert ruled.request.stated_max_tokens == 12000


def test_the_rule_choice_carries_the_policys_own_value_and_p():
    router = _router(round2_config(**RULE), None)
    res = router.route([{"role": "user", "content": DESIGN["prompt"]}], system=DESIGN["system"],
                       max_tokens=12000, now=1000.0)
    ctx, conv, policy = router.context(), Conversation(), router.policy
    choice = policy.choose(conv, res.request, ctx)
    pool = candidates(res.request, ctx, allow_subscription=True)
    v, p = policy.value(ctx.catalog["gpt-5.6-sol"], conv, res.request, ctx, res.request.difficulty, pool)
    assert (choice.model, choice.expected_cost, choice.p_success) == ("gpt-5.6-sol", v, p)


@pytest.mark.parametrize("max_tokens", [9999, 8000, None])
@pytest.mark.parametrize("classify", [None, stub_classifier(DESIGN)], ids=["no-classifier", "classifier"])
def test_below_the_threshold_or_unstated_is_f_expecteds_own_choice(max_tokens, classify):
    ruled, plain = _pair(max_tokens, classify)
    assert (ruled.model.name, ruled.reason) == (plain.model.name, plain.reason)
    assert "operator rule" not in ruled.reason


def test_a_budget_that_is_not_an_int_is_not_a_stated_budget():
    router = _router(round2_config(**RULE), None)
    for bogus in (12000.0, True, -12000, 0):
        res = router.route([{"role": "user", "content": DESIGN["prompt"]}], max_tokens=bogus, now=1000.0)
        assert res.request.stated_max_tokens is None and res.model.name != "gpt-5.6-sol"


def test_a_rule_route_filtered_out_of_the_pool_falls_through_and_says_why():
    # gpt-5.6-sol cannot take this prompt: its context window is too small,
    # so it is not a candidate for the turn while a free route is.
    small = [m.with_(context_tokens=1000) if m.name == "gpt-5.6-sol" else m for m in ROUND2]
    cfg = RouterConfig(providers=PROVIDERS, catalog=Catalog(small), policy={**ROUND2_POLICY, **RULE})
    plain_cfg = RouterConfig(providers=PROVIDERS, catalog=Catalog(small), policy=dict(ROUND2_POLICY))
    messages = [{"role": "user", "content": "x " * 4000}]
    ruled = _router(cfg, None).route(messages, max_tokens=12000, now=1000.0)
    plain = _router(plain_cfg, None).route(messages, max_tokens=12000, now=1000.0)
    assert "gpt-5.6-sol" not in {m.name for m in candidates(ruled.request, _router(cfg, None).context(),
                                                             allow_subscription=True)}
    assert ruled.model.name == plain.model.name != "gpt-5.6-sol"
    assert ruled.reason == (plain.reason + "; operator rule not applied: gpt-5.6-sol is not a "
                            "candidate for this turn")


def test_only_one_setting_never_fires():
    policy = ExpectedCostPolicy(paid_rule_route="gpt-5.6-sol")
    assert not policy.rule_applies(TurnRequest("design", 0.5, 100, 4000, 0.0, stated_max_tokens=12000))
    policy = ExpectedCostPolicy(paid_rule_min_max_tokens=10000)
    assert not policy.rule_applies(TurnRequest("design", 0.5, 100, 4000, 0.0, stated_max_tokens=12000))


def test_tool_loop_turns_stay_on_the_turns_model():
    router = _router(round2_config(**RULE), None)
    tools = [{"name": "bash", "input_schema": {"type": "object"}}]
    convo = [{"role": "user", "content": "build the page"}]
    first = router.route(convo, None, tools, max_tokens=4000, now=1000.0)
    router.commit(first, 2000, 100)
    convo += [{"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "bash", "input": {}}]},
              {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}]
    second = router.route(convo, None, tools, max_tokens=12000, now=1005.0)
    assert not second.turn_start and second.model.name == first.model.name
    assert second.reason == "inside a tool loop: stay on the turn's model"


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------
SUB = ModelInfo("plan", "host", "plan", Prices.free(), subscription="claude")


@pytest.mark.parametrize("route, match", [
    ("kimi-k3", "free route"),
    ("plan", "subscription route"),
    ("no-such-route", "not a route in the catalog"),
])
def test_a_free_subscription_or_unknown_rule_route_is_refused(route, match):
    catalog = Catalog(list(ROUND2) + [SUB])
    with pytest.raises(ValueError, match=match):
        validate_paid_rule({"paid_rule_route": route, "paid_rule_min_max_tokens": 10000}, catalog)
    with pytest.raises(ValueError, match=match):
        Router(RouterConfig(providers=PROVIDERS, catalog=catalog,
                            policy={"paid_rule_route": route, "paid_rule_min_max_tokens": 10000}),
               classifier=None, quota_reader=lambda: {})


@pytest.mark.parametrize("threshold", [0, -1, 10000.0, True, "10000"])
def test_a_threshold_that_is_not_a_positive_int_is_refused(threshold):
    with pytest.raises(ValueError, match="paid_rule_min_max_tokens"):
        validate_paid_rule({"paid_rule_route": "gpt-5.6-sol", "paid_rule_min_max_tokens": threshold},
                           Catalog(list(ROUND2)))


def _write_config(tmp_path, route):
    path = tmp_path / "router.yaml"
    path.write_text(
        "providers:\n"
        "  host: {base_url: 'https://example.invalid/v1'}\n"
        "  openrouter: {base_url: 'https://example.invalid/v1'}\n"
        "policy:\n"
        f"  paid_rule_route: {route}\n"
        "  paid_rule_min_max_tokens: 10000\n"
        "models:\n"
        "  - {name: kimi-k3, provider: host, upstream_id: moonshotai/Kimi-K3-TEE, free: true}\n"
        "  - name: gpt-5.6-sol\n"
        "    provider: openrouter\n"
        "    upstream_id: openai/gpt-5.6-sol\n"
        "    prices: {input: 2.0, output: 10.0, cache_read: 0.2, cache_write: 2.5}\n")
    return path


@pytest.mark.parametrize("route", ["kimi-k3", "nope"])
def test_load_config_refuses_to_start_on_a_bad_rule_route(tmp_path, route):
    with pytest.raises(ValueError, match="paid_rule_route"):
        load_config(_write_config(tmp_path, route), use_bench=False)


def test_load_config_accepts_a_metered_rule_route(tmp_path):
    config = load_config(_write_config(tmp_path, "gpt-5.6-sol"), use_bench=False)
    assert config.policy["paid_rule_route"] == "gpt-5.6-sol"


# ---------------------------------------------------------------------------
# the hashed identity
# ---------------------------------------------------------------------------
def test_both_settings_are_in_the_hashed_policy_identity():
    from experiments.supplement import policy_identity
    base = policy_identity(round2_config(**RULE))
    assert base["policy_settings"]["paid_rule_route"] == "gpt-5.6-sol"
    assert base["policy_settings"]["paid_rule_min_max_tokens"] == 10000
    other_route = policy_identity(round2_config(**{**RULE, "paid_rule_route": "other"}))
    other_min = policy_identity(round2_config(**{**RULE, "paid_rule_min_max_tokens": 12000}))
    unset = policy_identity(round2_config())
    assert len({base["sha256"], other_route["sha256"], other_min["sha256"], unset["sha256"]}) == 4
