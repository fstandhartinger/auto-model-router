"""Cold versus warm prompt cache, end to end through the conversation state.

``test_economics.py`` prices one call; these follow a conversation: the model
that served the last turn holds a warm prefix, every other model is cold,
the warmth expires with the provider's TTL, and switching throws it away.
"""

import pytest

from auto_router.cache_index import PrefixCacheIndex, prefix_hashes
from auto_router.catalog import CacheRules
from auto_router.config import RouterConfig
from auto_router.economics import SuccessModel, turn_cost
from auto_router.policies import Context, Conversation, TurnRequest, turn_call_cost
from auto_router.router import Router
from tests.helpers import FRONTIER, MID, catalog

PROMPT = 60_000


def _req(now, prompt=PROMPT):
    return TurnRequest("coding", 0.5, prompt_tokens=prompt, output_tokens=1000, now=now)


def test_the_serving_model_is_warm_and_every_other_model_is_cold():
    conv = Conversation()
    conv.record_call(FRONTIER, PROMPT, 1000, now=1000.0)
    assert conv.warm_tokens(FRONTIER, 1100.0) == PROMPT + 1000
    assert conv.warm_tokens(MID, 1100.0) == 0
    ctx = Context(catalog(FRONTIER, MID), SuccessModel())
    warm = turn_call_cost(FRONTIER, _req(1100.0), conv.warm_tokens(FRONTIER, 1100.0), ctx)
    cold = turn_call_cost(FRONTIER, _req(1100.0), 0, ctx)
    # the warm prefix is read at 0.5 $/M instead of written at 6.25 $/M
    assert cold - warm == pytest.approx(PROMPT * (6.25 - 0.5) / 1e6)


def test_warmth_expires_with_the_provider_ttl():
    conv = Conversation()
    conv.record_call(FRONTIER, PROMPT, 1000, now=1000.0)
    ttl = FRONTIER.cache.ttl_seconds
    assert conv.warm_tokens(FRONTIER, 1000.0 + ttl - 30) > 0
    assert conv.warm_tokens(FRONTIER, 1000.0 + ttl + 1) == 0
    long_ttl = FRONTIER.with_(cache=CacheRules(ttl_seconds=3600, min_tokens=1024, hit_rate=1.0))
    conv.record_call(long_ttl, PROMPT, 1000, now=1000.0)
    assert conv.warm_tokens(long_ttl, 1000.0 + ttl + 1) > 0


def test_switching_models_loses_the_cache_and_pays_the_write_again():
    conv = Conversation()
    conv.record_call(FRONTIER, PROMPT, 1000, now=1000.0)
    conv.record_call(MID, PROMPT + 2000, 1000, now=1060.0)
    assert conv.switches == 1 and conv.current == MID.name
    # MID is now warm on the grown prefix; FRONTIER still holds only its old one
    assert conv.warm_tokens(MID, 1100.0) == PROMPT + 3000
    staying = turn_cost(MID, PROMPT + 5000, conv.warm_tokens(MID, 1100.0), 0)
    switching_back = turn_cost(FRONTIER, PROMPT + 5000, conv.warm_tokens(FRONTIER, 1100.0), 0)
    fresh = turn_cost(FRONTIER, PROMPT + 5000, 0, 0)
    assert staying < switching_back < fresh


def test_a_prompt_below_the_minimum_is_never_recorded_warm():
    conv = Conversation()
    conv.record_call(FRONTIER, 500, 10, now=1000.0)
    assert conv.warm_tokens(FRONTIER, 1001.0) == 0


def test_prefix_index_is_per_model_and_expires():
    index = PrefixCacheIndex(safety_seconds=30)
    messages = [{"role": "user", "content": "x" * 10}, {"role": "assistant", "content": "y"}]
    hashes = prefix_hashes(messages)
    index.record(FRONTIER, hashes, [20_000, 21_000], started_at=1000.0)
    assert index.lookup(FRONTIER.name, hashes, now=1100.0).prefix_tokens == 21_000
    assert index.lookup(MID.name, hashes, now=1100.0) is None
    assert index.lookup(FRONTIER.name, hashes, now=1000.0 + 300 - 29) is None
    # a longer conversation still finds the shared prefix
    longer = prefix_hashes(messages + [{"role": "user", "content": "next"}])
    assert index.lookup(FRONTIER.name, longer, now=1100.0).message_index == 1


def test_decision_record_reports_cold_then_warm():
    router = Router(RouterConfig(providers={}, catalog=catalog(FRONTIER),
                                 policy={"success": {"evidence_discount": 0.0}}),
                    classifier=None, judge=None)
    messages = [{"role": "user", "content": "word " * 30_000}]
    first = router.route(messages, now=1000.0)
    assert first.explanation.cache.status == "cold"
    router.commit(first, prompt_tokens=first.request.prompt_tokens, output_tokens=100)
    second = router.route(messages + [{"role": "assistant", "content": "ok"},
                                      {"role": "user", "content": "and now?"}], now=1060.0)
    assert second.explanation.cache.status == "warm"
    assert second.explanation.cache.estimated_usd_avoided > 0
    late = router.route(messages + [{"role": "user", "content": "much later"}], now=5000.0)
    assert late.explanation.cache.status == "cold"
