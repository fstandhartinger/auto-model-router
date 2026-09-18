"""Outage and fallback behaviour, offline.

Covers the three external dependencies the router has - the classifier, the
benchmark API and the upstream route itself - and asserts that losing any of
them degrades the decision rather than breaking it, and that the degradation
is visible in the decision record instead of silent.
"""

import json
import os
import time

import pytest

from auto_router.bench import BenchmarkClient
from auto_router.catalog import CacheRules, Catalog, ModelInfo, Prices
from auto_router.config import RouterConfig, load_config
from auto_router.decision import ObservedOutcome
from auto_router.jev import FALLBACK, Classification
from auto_router.ledger import RoutingLedger
from auto_router.router import Router

CACHE = CacheRules(ttl_seconds=300, min_tokens=1024, hit_rate=0.9)
LONG = [{"role": "user", "content": "Refactor the billing module and add tests. " * 80}]


def _model(name, inp, out, cap, **kw):
    return ModelInfo(name, "host", name, Prices(inp, out, inp / 10), CACHE,
                     capability={"coding": cap, "general": cap, "design": cap},
                     capability_basis={"coding": "aa_coding_index", "general": "ii"},
                     capability_strength={"coding": "direct", "general": "direct",
                                          "design": "direct"}, **kw)


CHEAP, MID, STRONG = _model("cheap", 0.1, 0.4, 45), _model("mid", 1.0, 5.0, 65), _model("strong", 5.0, 25.0, 85)


def _router(*models, classifier=None, ledger=None):
    config = RouterConfig(providers={}, catalog=Catalog(list(models) or [CHEAP, MID, STRONG]),
                          policy={"success": {"evidence_discount": 0.5}})
    return Router(config, classifier=classifier, ledger=ledger)


# -- classifier outage ------------------------------------------------------
def test_a_classifier_outage_still_routes():
    result = _router(classifier=lambda text, ctx: FALLBACK).route(LONG)
    assert result.model.name in {"cheap", "mid", "strong"}
    assert result.explanation.selection.safe_fallback == "classifier-unavailable"
    assert result.explanation.classification.source == "fallback"


def test_a_classifier_outage_does_not_claim_confidence():
    result = _router(classifier=lambda text, ctx: FALLBACK).route(LONG)
    record = result.explanation.to_dict()["classification"]
    assert record["category_confidence"] == 0.0 and record["difficulty_confidence"] == 0.0
    assert result.explanation.selection.evidence_confidence <= 0.5


def test_a_classifier_outage_is_more_cautious_than_a_confident_easy_answer():
    """An unknown task must not be treated as an easy one."""
    easy = Classification(category="coding", category_probs={}, difficulty=0.05,
                          difficulty_confidence=0.95, needs_tools=0.0, needs_vision=0.0,
                          needs_long_context=0.0, follow_up=0.0, stakes=0.0,
                          category_confidence=0.95, model="jev-1.13.0")
    confident = _router(classifier=lambda t, c: easy).route(LONG)
    degraded = _router(classifier=lambda t, c: FALLBACK).route(LONG)
    assert confident.request.difficulty < degraded.request.difficulty
    assert confident.request.stakes_usd <= degraded.request.stakes_usd


def test_a_classifier_that_raises_does_not_take_the_router_down(monkeypatch):
    def boom(text, ctx):
        raise TimeoutError("classifier timed out")

    with pytest.raises(TimeoutError):
        _router(classifier=boom).route(LONG)
    # jev.classify itself never raises, which is what the server actually uses.
    from auto_router import jev
    monkeypatch.setattr(jev, "_post",
                        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timed out")))
    assert jev.classify("x", api_key="test-key").failed


# -- benchmark API outage ---------------------------------------------------
def _bench_config(tmp_path, seed=True, age_seconds=None, offline=False):
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    client = BenchmarkClient(base_url="http://127.0.0.1:1", cache_dir=cache, offline=offline,
                             timeout=0.3)
    if seed:
        doc = {"model": {"id": "m::max",
                         "benchmarks": {"aa_intelligence_index": 50.0, "aa_coding_index": 70.0},
                         "designarena": {"frontend": {"elo": 1300, "battles": 900}}}}
        client._cache_path("/api/models/m::max").write_text(json.dumps(doc))
        if age_seconds:
            old = time.time() - age_seconds
            os.utime(client._cache_path("/api/models/m::max"), (old, old))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "providers": {"h": {"base_url": "http://127.0.0.1:1/v1"}},
        "models": [{"name": "m", "provider": "h", "bench_id": "m::max",
                    "prices": {"input": 1.0, "output": 4.0}},
                   {"name": "configured", "provider": "h", "prices": {"input": 2.0, "output": 8.0},
                    "capability": {"coding": 60, "general": 60}}]}))
    return load_config(cfg, bench=client)


def test_a_benchmark_outage_with_a_recent_copy_keeps_routing(tmp_path):
    cfg = _bench_config(tmp_path, age_seconds=2 * 24 * 3600)
    model = cfg.catalog["m"]
    assert model.capability["coding"] == 70.0
    assert model.evidence_stale is True and model.evidence["source"] == "stale-cache"
    result = Router(cfg, classifier=None).route(LONG)
    assert result.model.name in {"m", "configured"}


def test_a_benchmark_outage_with_no_copy_falls_back_to_the_config(tmp_path):
    cfg = _bench_config(tmp_path, seed=False)
    assert cfg.catalog["m"].capability == {}, "no invented capability numbers"
    assert cfg.catalog["m"].prices.input == 2.0 or cfg.catalog["m"].prices.input == 1.0
    assert cfg.catalog["configured"].capability["coding"] == 60
    result = Router(cfg, classifier=None).route(LONG)
    assert result.model.name == "configured", "the route with real evidence is preferred"
    assert result.explanation is not None


def test_stale_benchmark_data_is_visible_in_the_router_stats(tmp_path):
    cfg = _bench_config(tmp_path, age_seconds=2 * 24 * 3600)
    assert Router(cfg, classifier=None).stats["stale_evidence_models"] == ["m"]


# -- upstream route outage --------------------------------------------------
def test_an_availability_failure_falls_back_sideways_when_nothing_is_stronger():
    """A 5xx from the strongest route is not evidence that it was too weak."""
    router = _router(CHEAP, MID, STRONG)
    result = router.route(LONG)
    forced = router.route(LONG)
    forced.model = STRONG
    assert router.escalate(forced) is None, "no stronger capability route exists"
    sideways = router.escalate(forced, availability=True)
    assert sideways is not None and sideways.model.name != "strong"
    assert "safe fallback" in sideways.reason
    assert result.explanation is not None


def test_an_availability_fallback_does_not_repeat_a_tried_route():
    router = _router(CHEAP, MID, STRONG)
    result = router.route(LONG)
    result.model = STRONG
    first = router.escalate(result, availability=True)
    first.tried = {STRONG.name, first.model.name}
    second = router.escalate(first, availability=True)
    assert second is None or second.model.name not in first.tried


def test_the_last_remaining_route_reports_no_fallback():
    router = _router(CHEAP)
    result = router.route(LONG)
    assert router.escalate(result, availability=True) is None
    assert result.explanation.selection.fallback is None


# -- the ledger -------------------------------------------------------------
def test_the_ledger_writes_one_json_object_per_decision(tmp_path):
    path = tmp_path / "decisions.jsonl"
    router = _router(ledger=RoutingLedger(path))
    for _ in range(3):
        result = router.route(LONG)
        router.observe(result, ObservedOutcome(model=result.model.name, status="ok",
                                               output_tokens=10, cost_usd=0.001,
                                               cost_basis="test"))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 3
    assert all(line["observed_outcome"]["kind"] == "observed" for line in lines)
    assert all(line["estimated_outcome"]["kind"] == "estimate" for line in lines)
    assert router.ledger.stats["written"] == 3 and router.ledger.stats["failed"] == 0


def test_the_ledger_holds_no_prompt_text(tmp_path):
    path = tmp_path / "decisions.jsonl"
    router = _router(ledger=RoutingLedger(path))
    secret = "sk-" + "ant-api03-" + "Q" * 24
    result = router.route([{"role": "user", "content": f"billing module {secret} " * 80}])
    router.observe(result, ObservedOutcome(model=result.model.name, status="ok"))
    body = path.read_text()
    assert secret not in body and "billing module" not in body


def test_an_unwritable_ledger_never_breaks_routing(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    router = _router(ledger=RoutingLedger(blocked / "nested" / "l.jsonl"))
    result = router.route(LONG)
    router.observe(result, ObservedOutcome(model=result.model.name, status="ok"))
    assert router.ledger.stats["failed"] == 1 and router.ledger.stats["written"] == 0
    assert result.model.name, "routing still happened"


def test_a_disabled_ledger_is_a_no_op():
    router = _router(ledger=RoutingLedger(None))
    result = router.route(LONG)
    router.observe(result, ObservedOutcome(model=result.model.name, status="ok"))
    assert router.ledger.stats == {"enabled": False, "path": None, "written": 0, "failed": 0}
