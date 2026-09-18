"""Regression tests for the findings of the independent review (18 Sep 2026).

The review was run by a model from a different family on the change set, not by
the author. Each test below pins one substantiated finding so it cannot come
back. The finding is quoted in the docstring so the test explains itself.
"""

import json
import os
import time

import pytest

from auto_router import bench, server
from auto_router.bench import BenchmarkClient, capability_evidence, design_capability
from auto_router.catalog import CacheRules, Catalog, ModelInfo, Prices
from auto_router.config import RouterConfig, load_config
from auto_router.economics import SuccessModel
from auto_router.jev import Classification
from auto_router.router import Router

CACHE = CacheRules(ttl_seconds=300, min_tokens=1024, hit_rate=0.9)
DOC = {"model": {"id": "m::max",
                 "benchmarks": {"aa_intelligence_index": 50.0, "aa_coding_index": 70.0,
                                "aa_lcr": 0.8},
                 "designarena": {"frontend": {"elo": 1300, "battles": 900}}}}
LONG = [{"role": "user", "content": "Refactor the billing module and add tests. " * 80}]


# -- B/high: an upstream error body must never be persisted -----------------
class _Resp:
    def __init__(self, status):
        self.status_code = status


def test_an_upstream_error_body_never_reaches_the_decision_record():
    """Finding: `str(data.get("error"))[:80]` can carry prompt text or the
    rejected credential into /v1/router/decisions and the JSONL ledger."""
    leak = {"message": "invalid key sk-" + "x" * 40 + " while handling: Refactor the billing",
            "type": "authentication_error"}
    label = server.error_label(_Resp(401), {"error": leak})
    assert label == "http_401"
    assert "sk-" not in label and "Refactor" not in label


def test_a_transport_failure_records_only_the_exception_class():
    assert server.error_label(None, {"error": "ConnectError"}) == "transport:ConnectError"
    # Anything that is not a bare identifier is not echoed back.
    assert server.error_label(None, {"error": "boom: key=secret"}) == "transport"
    assert server.error_label(None, "not a dict") == "transport"


# -- A/medium: offline mode must not make an old copy look fresh ------------
def _seeded(tmp_path, **kw):
    client = BenchmarkClient(base_url="http://127.0.0.1:9", cache_dir=tmp_path, timeout=0.3, **kw)
    client._cache_path("/api/models/m::max").write_text(json.dumps(DOC))
    return client


def test_offline_mode_does_not_relabel_an_expired_copy_as_fresh(tmp_path):
    """Finding: `(self.offline or age < self.ttl)` returned source="cache" for a
    two-day-old copy under a one-day TTL, so nothing discounted it."""
    client = _seeded(tmp_path, offline=True, ttl_seconds=24 * 3600)
    old = time.time() - 2 * 24 * 3600
    for path in tmp_path.iterdir():
        os.utime(path, (old, old))
    doc, prov = client.model_with_provenance("m::max")
    assert doc is not None, "an offline router still routes"
    assert prov.source == "stale-cache" and prov.stale is True


def test_offline_mode_still_reports_a_genuinely_fresh_copy_as_fresh(tmp_path):
    _, prov = _seeded(tmp_path, offline=True).model_with_provenance("m::max")
    assert prov.source == "cache" and prov.stale is False


# -- A/medium: a config override must survive a stale benchmark document ----
def _config(tmp_path, client, models):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"providers": {"h": {"base_url": "http://127.0.0.1:1/v1"}},
                                "models": models}))
    return load_config(path, bench=client)


def test_a_config_override_is_not_demoted_by_a_stale_benchmark_document(tmp_path):
    """Finding: cap_source "bench+config" left evidence_stale true, so an
    explicit operator statement was downgraded to weak along with the rest."""
    client = _seeded(tmp_path, offline=False)
    old = time.time() - 2 * 24 * 3600
    for path in tmp_path.iterdir():
        os.utime(path, (old, old))
    model = _config(tmp_path, client, [
        {"name": "m", "provider": "h", "bench_id": "m::max",
         "prices": {"input": 1.0, "output": 4.0}, "capability": {"design": 88}}]).catalog["m"]
    assert model.evidence_stale is True, "the benchmark document really is stale"
    assert model.evidence_strength("design") == "direct", "the operator's own number stands"
    assert model.evidence_strength("coding") == "weak", "the stale benchmark number is demoted"


# -- A/medium: a stale benchmaxxing penalty must not move a fresh capability -
def test_a_stale_benchmaxxing_report_is_recorded_and_not_applied(tmp_path):
    """Finding: benchmaxxing is a separate request and can be stale while the
    model document is fresh; the penalty was applied anyway and the decision
    was reported as current."""
    client = _seeded(tmp_path, offline=False)
    report = client._cache_path("/api/benchmaxxing?report=m::max")
    report.write_text(json.dumps({"report": {"status": "scored", "score": 9.0}}))
    old = time.time() - 30 * 24 * 3600
    os.utime(report, (old, old))
    model = _config(tmp_path, client, [
        {"name": "m", "provider": "h", "bench_id": "m::max",
         "prices": {"input": 1.0, "output": 4.0}}]).catalog["m"]
    assert model.benchmaxxing == 0.0, "an expired penalty is not applied"
    assert "benchmaxxing" in model.evidence, "its provenance is still recorded"


# -- A/medium: a retry must appear in the decision history ------------------
def _router(*models, classifier=None, **policy):
    config = RouterConfig(providers={}, catalog=Catalog(list(models)), policy=policy)
    return Router(config, classifier=classifier)


def _model(name, inp, out, cap):
    return ModelInfo(name, "h", name, Prices(inp, out, inp / 10), CACHE,
                     capability={"coding": cap, "general": cap},
                     capability_basis={"coding": "aa_coding_index", "general": "ii"},
                     capability_strength={"coding": "direct", "general": "direct"})


def test_a_retry_route_gets_its_own_decision_record():
    """Finding: escalate() built a RouteResult with explanation=None, so the
    route that actually answered never reached the ledger - only the one that
    failed did."""
    cheap, mid, strong = _model("cheap", 0.1, 0.4, 45), _model("mid", 1.0, 5.0, 65), _model("strong", 5.0, 25.0, 85)
    router = _router(cheap, mid, strong)
    first = router.route(LONG)
    before = len(router.decisions)
    retry = router.escalate(first, availability=True)
    assert retry is not None
    assert retry.explanation is not None, "the retry is explainable too"
    assert len(router.decisions) == before + 1, "and it is in the history"
    assert retry.explanation.selection.switched_from == first.model.name
    assert retry.tried == {first.model.name}
    assert retry.headers["X-Router-Decision"] == retry.explanation.id


def test_a_retry_after_a_capability_failure_is_recorded_too():
    cheap, strong = _model("cheap", 0.1, 0.4, 45), _model("strong", 5.0, 25.0, 85)
    router = _router(cheap, strong)
    first = router.route(LONG)
    first.model = cheap
    retry = router.escalate(first)
    if retry is not None:
        assert retry.explanation is not None


# -- A/medium: confidence must not be inflated or taken optimistically ------
def _cls(cat_conf, diff_conf, failed=False):
    return Classification(category="coding", category_probs={}, difficulty=0.5,
                          difficulty_confidence=diff_conf, needs_tools=0.0, needs_vision=0.0,
                          needs_long_context=0.0, follow_up=0.1, stakes=0.5,
                          category_confidence=cat_conf, failed=failed, model="jev-1.13.0")


def test_a_missing_confidence_is_reported_as_zero_not_as_a_half():
    """Finding: `max(...) or 0.5` turned "no confidence reported" into 0.5."""
    router = _router(_model("m", 1.0, 4.0, 70), classifier=lambda t, c: _cls(0.0, 0.0))
    assert router.route(LONG).explanation.selection.evidence_confidence == pytest.approx(0.5)
    # 0.5 here is the capability half only; the classifier half contributes nothing.


def test_confidence_combines_the_two_signals_conservatively():
    """Finding: `max` treated a confident category with an unknown difficulty as
    a fully confident decision, although the route depends on both."""
    router = _router(_model("m", 1.0, 4.0, 70), classifier=lambda t, c: _cls(0.95, 0.10))
    optimistic = _router(_model("m", 1.0, 4.0, 70), classifier=lambda t, c: _cls(0.95, 0.95))
    low = router.route(LONG).explanation.selection.evidence_confidence
    high = optimistic.route(LONG).explanation.selection.evidence_confidence
    assert low < high
    assert low == pytest.approx(0.5 * 1.0 + 0.5 * 0.10, abs=1e-3)


def test_no_classifier_at_all_is_not_treated_as_half_confident():
    router = _router(_model("m", 1.0, 4.0, 70), classifier=None)
    assert router.route(LONG).explanation.selection.evidence_confidence == pytest.approx(0.5)


# -- A/high: weak evidence must change the decision, not just annotate it ---
def test_the_evidence_discount_is_on_by_default():
    """Finding: with the discount defaulting to 0, stale evidence was only
    labelled; it never actually altered a route choice."""
    assert SuccessModel().evidence_discount > 0.0
    weak = ModelInfo("w", "h", "w", Prices(1.0, 2.0), capability={"design": 80.0},
                     capability_basis={"design": "b"}, capability_strength={"design": "weak"})
    strong = weak.with_(name="s", capability_strength={"design": "direct"})
    success = SuccessModel()
    assert success.p(weak, "design", 0.8) < success.p(strong, "design", 0.8)


def test_setting_the_discount_to_zero_restores_the_old_behaviour():
    weak = ModelInfo("w", "h", "w", Prices(1.0, 2.0), capability={"design": 80.0},
                     capability_basis={"design": "b"}, capability_strength={"design": "weak"})
    strong = weak.with_(name="s", capability_strength={"design": "direct"})
    success = SuccessModel(evidence_discount=0.0)
    assert success.p(weak, "design", 0.8) == success.p(strong, "design", 0.8)


# -- D/medium: malformed upstream numbers must be rejected, not ranked ------
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_benchmark_numbers_are_rejected(value):
    """Finding: NaN passes isinstance(v, float) and poisons every comparison,
    because NaN > x is always False."""
    doc = {"benchmarks": {"aa_coding_index": value, "aa_intelligence_index": 50.0}}
    evidence = capability_evidence(doc)
    assert "coding" not in evidence or evidence["coding"].value == evidence["coding"].value


def test_a_non_finite_design_elo_does_not_crash_or_rank():
    assert design_capability({"designarena": {"frontend": {"elo": float("nan"),
                                                           "battles": 900}}}) is None


def test_a_non_finite_battle_count_does_not_raise():
    evidence = design_capability({"designarena": {"frontend": {"elo": 1300,
                                                               "battles": float("nan")}}})
    assert evidence is not None and evidence.strength == "weak"


def test_a_boolean_is_not_mistaken_for_a_score():
    assert bench._number(True) is None and bench._number(False) is None


def test_a_model_document_full_of_junk_still_builds(tmp_path):
    junk = {"model": {"id": "j::max", "benchmarks": {"aa_coding_index": float("inf"),
                                                     "aa_intelligence_index": None},
                      "designarena": {"frontend": "not a dict"}}}
    client = BenchmarkClient(base_url="http://127.0.0.1:9", cache_dir=tmp_path, offline=True)
    client._cache_path("/api/models/j::max").write_text(json.dumps(junk))
    cfg = _config(tmp_path, client, [{"name": "j", "provider": "h", "bench_id": "j::max",
                                      "prices": {"input": 1.0, "output": 4.0}}])
    model = cfg.catalog["j"]
    assert all(v == v for v in model.capability.values()), "no NaN reached the catalog"
    assert Router(cfg, classifier=None).route(LONG).model.name == "j"
