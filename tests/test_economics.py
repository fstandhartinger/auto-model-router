import pytest

from auto_router.catalog import CacheRules, Prices
from auto_router.economics import SuccessModel, horizon_cost, is_warm, turn_cost
from tests.helpers import CHEAP, FRONTIER, MID, model


def test_cold_prompt_pays_write_price_warm_pays_read_price():
    cold = turn_cost(FRONTIER, 50_000, 0, 0)
    warm = turn_cost(FRONTIER, 50_000, 50_000, 0)
    assert cold == pytest.approx(50_000 * 6.25 / 1e6)
    assert warm == pytest.approx(50_000 * 0.5 / 1e6)
    assert cold / warm == pytest.approx(12.5)


def test_no_write_premium_means_writes_cost_input():
    assert turn_cost(MID, 10_000, 0, 0) == pytest.approx(10_000 * 1.0 / 1e6)


def test_misses_are_billed_as_writes():
    flaky = FRONTIER.with_(cache=CacheRules(hit_rate=0.1))
    assert turn_cost(flaky, 47_000, 47_000, 200) > 5 * turn_cost(FRONTIER, 47_000, 47_000, 200)


def test_below_minimum_prefix_everything_is_plain_input():
    assert turn_cost(FRONTIER, 500, 500, 0) == pytest.approx(500 * 5.0 / 1e6)


def test_free_models_cost_nothing():
    free = model("free", 0, 0, read=0, write=0)
    assert free.prices.is_free and turn_cost(free, 100_000, 0, 5000) == 0.0


def test_ttl_expiry():
    assert is_warm(FRONTIER, 1000.0, 1000.0 + 200)
    assert not is_warm(FRONTIER, 1000.0, 1000.0 + 299)   # safety margin
    assert not is_warm(FRONTIER, None, 1000.0)


def test_horizon_rewards_warm_cheap_follow_ups():
    stay = horizon_cost(FRONTIER, 100_000, 100_000, 1000, 4, 2000)
    move = horizon_cost(CHEAP, 100_000, 0, 1000, 4, 2000)
    assert move < stay


def test_success_model_is_monotone():
    s = SuccessModel()
    assert s.p(FRONTIER, "coding", 0.5) > s.p(MID, "coding", 0.5) > s.p(CHEAP, "coding", 0.5)
    assert s.p(MID, "coding", 0.2) > s.p(MID, "coding", 0.8)
    d = s.difficulty_at(MID, "coding", 0.5)
    assert s.p(MID, "coding", d) == pytest.approx(0.5, abs=0.02)


def test_benchmaxxing_penalty_lowers_capability():
    inflated = MID.with_(benchmaxxing=10.0)
    honest = MID.with_(benchmaxxing=-8.0)
    assert inflated.cap("coding") == pytest.approx(50.0)
    assert honest.cap("coding") == pytest.approx(55.0), "negative gaps are not rewarded"


def test_measured_rates_override_the_curve():
    s = SuccessModel(measured={("mid", "coding", "hard"): 0.33})
    assert s.p(MID, "coding", 0.9) == 0.33
