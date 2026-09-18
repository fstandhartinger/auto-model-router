"""Benchmark provenance, freshness and evidence-driven specialisation.

Three properties are checked here:

1. A capability number knows where it came from and how strong that basis is.
2. Missing or stale evidence lowers confidence and triggers the documented
   fallback rather than being silently treated as a measurement.
3. A specialised route (web/UI design) is chosen from served evidence, never
   from a name hard-coded in the router.
"""

import json
import os
import time

import pytest

from auto_router import bench
from auto_router.bench import (
    BenchmarkClient,
    capability_evidence,
    capability_from_model,
    design_capability,
)
from auto_router.catalog import EVIDENCE_WEIGHT, ModelInfo, Prices
from auto_router.config import load_config


#: Three synthetic models with near-identical coding evidence but very
#: different design evidence - the case the router has to get right.
DESIGN_DOC = {
    "id": "arena-strong::max",
    "benchmarks": {"aa_intelligence_index": 50.0, "aa_coding_index": 76.0, "aa_lcr": 0.8},
    "designarena": {"frontend": {"elo": 1309, "battles": 1624},
                    "fullstack": {"elo": 1331, "battles": 1178}},
}
DESIGN_WEAK_DOC = {
    "id": "arena-weak::max",
    "benchmarks": {"aa_intelligence_index": 50.0, "aa_coding_index": 76.0, "aa_lcr": 0.8},
    "designarena": {"frontend": {"elo": 1204, "battles": 2254},
                    "fullstack": {"elo": 1168, "battles": 1612}},
}
NO_ARENA_DOC = {
    "id": "arena-absent::max",
    "benchmarks": {"aa_intelligence_index": 50.0, "aa_coding_index": 76.0, "aa_lcr": 0.8},
    "designarena": {},
}


def test_design_capability_comes_from_the_served_arena_data():
    strong = design_capability(DESIGN_DOC)
    weak = design_capability(DESIGN_WEAK_DOC)
    assert strong.strength == "direct" and weak.strength == "direct"
    assert strong.value > weak.value
    # 50 + (mean elo - 1200) / 4
    assert strong.value == pytest.approx(50 + (1320 - 1200) / 4)
    assert "designarena" in strong.basis and "battles" in strong.basis


def test_a_thin_arena_sample_is_reported_as_weak_evidence():
    thin = design_capability({"designarena": {"frontend": {"elo": 1400, "battles": 12}}})
    assert thin.strength == "weak"


def test_no_arena_data_falls_back_and_says_so():
    assert design_capability(NO_ARENA_DOC) is None
    ev = capability_evidence(NO_ARENA_DOC)["design"]
    assert ev.strength == "derived"
    assert ev.basis.startswith("fallback:")
    assert ev.value == capability_evidence(NO_ARENA_DOC)["coding"].value


def test_design_is_not_predictable_from_coding():
    """The whole point of a specialised route: coding rank does not decide design rank."""
    coding = {d["id"]: capability_evidence(d)["coding"].value
              for d in (DESIGN_DOC, DESIGN_WEAK_DOC)}
    design = {d["id"]: capability_evidence(d)["design"].value
              for d in (DESIGN_DOC, DESIGN_WEAK_DOC)}
    assert len(set(coding.values())) == 1, "coding evidence is identical"
    assert len(set(design.values())) == 2, "design evidence separates them"


def test_summarisation_is_labelled_derived_never_measured():
    ev = capability_evidence(DESIGN_DOC)["summarisation"]
    assert ev.strength == "derived"
    assert "mean(" in ev.basis


def test_every_capability_carries_a_basis():
    for doc in (DESIGN_DOC, DESIGN_WEAK_DOC, NO_ARENA_DOC):
        for category, ev in capability_evidence(doc).items():
            assert ev.basis, category
            assert ev.strength in EVIDENCE_WEIGHT, category


def test_capability_from_model_is_still_a_plain_mapping():
    plain = capability_from_model(DESIGN_DOC)
    assert plain["design"] == pytest.approx(80.0)
    assert all(isinstance(v, float) for v in plain.values())


# -- provenance -------------------------------------------------------------
def _seed(tmp_path, offline=True, **kw):
    client = BenchmarkClient(base_url="http://127.0.0.1:9", cache_dir=tmp_path, offline=offline,
                             timeout=0.5, **kw)
    client._cache_path("/api/models/arena-strong::max").write_text(
        json.dumps({"model": DESIGN_DOC}))
    return client


def test_a_fresh_cache_read_is_not_stale(tmp_path):
    doc, prov = _seed(tmp_path).model_with_provenance("arena-strong::max")
    assert doc["id"] == "arena-strong::max"
    assert prov.source == "cache" and not prov.stale and prov.usable
    assert prov.to_dict()["origin"] == "http://127.0.0.1:9"


def test_an_outage_falling_back_to_an_old_copy_is_marked_stale(tmp_path):
    client = _seed(tmp_path, offline=False)
    old = time.time() - 2 * 24 * 3600
    for path in tmp_path.iterdir():
        os.utime(path, (old, old))
    doc, prov = client.model_with_provenance("arena-strong::max")
    assert doc is not None, "routing keeps working during an outage"
    assert prov.source == "stale-cache" and prov.stale
    assert prov.error, "the fetch failure is named"
    assert client.errors


def test_a_copy_past_the_stale_limit_is_treated_as_missing(tmp_path):
    client = _seed(tmp_path, offline=True, max_stale_seconds=3600)
    old = time.time() - 48 * 3600
    for path in tmp_path.iterdir():
        os.utime(path, (old, old))
    doc, prov = client.model_with_provenance("arena-strong::max")
    assert doc is None and prov.source == "missing" and prov.stale


def test_a_corrupt_cache_file_does_not_crash_routing(tmp_path):
    client = _seed(tmp_path)
    client._cache_path("/api/models/arena-strong::max").write_text("{not json")
    doc, prov = client.model_with_provenance("arena-strong::max")
    assert doc is None and prov.source == "missing"


def test_cache_files_are_scoped_to_the_api_origin(tmp_path):
    first = BenchmarkClient(base_url="https://one.invalid", cache_dir=tmp_path)
    second = BenchmarkClient(base_url="https://two.invalid", cache_dir=tmp_path)
    assert first._cache_path("/api/models/x") != second._cache_path("/api/models/x")


def test_a_future_dated_cache_file_is_refetched_not_trusted(tmp_path):
    client = _seed(tmp_path, offline=True)
    ahead = time.time() + 10_000
    for path in tmp_path.iterdir():
        os.utime(path, (ahead, ahead))
    doc, prov = client.model_with_provenance("arena-strong::max")
    assert doc is None and prov.source == "missing"


# -- config wiring ----------------------------------------------------------
def _config(tmp_path, client, extra=None):
    cfg_path = tmp_path / "cfg.json"
    models = [{"name": "arena-strong", "provider": "host", "bench_id": "arena-strong::max",
               "prices": {"input": 1.0, "output": 4.0}}]
    if extra:
        models.append(extra)
    cfg_path.write_text(json.dumps({
        "providers": {"host": {"base_url": "https://example.invalid/v1"}}, "models": models}))
    return load_config(cfg_path, bench=client)


def test_config_records_capability_basis_and_provenance(tmp_path):
    model = _config(tmp_path, _seed(tmp_path)).catalog["arena-strong"]
    assert model.capability_strength["design"] == "direct"
    assert "designarena" in model.capability_basis["design"]
    assert model.evidence["source"] == "cache" and model.evidence_stale is False
    assert model.evidence_basis("design").startswith("designarena")
    assert model.evidence_strength("design") == "direct"


def test_stale_evidence_downgrades_every_strength(tmp_path):
    client = _seed(tmp_path, offline=False)
    old = time.time() - 2 * 24 * 3600
    for path in tmp_path.iterdir():
        os.utime(path, (old, old))
    model = _config(tmp_path, client).catalog["arena-strong"]
    assert model.evidence_stale is True
    assert model.evidence_strength("design") == "weak", "direct evidence degrades, not vanishes"
    assert model.evidence_strength("summarisation") == "none", "derived evidence stops counting"


def test_a_config_override_counts_as_direct_and_never_stale(tmp_path):
    extra = {"name": "hand-graded", "provider": "host", "free": True,
             "capability": {"design": 88}}
    cfg = _config(tmp_path, _seed(tmp_path), extra)
    model = cfg.catalog["hand-graded"]
    assert model.capability["design"] == 88
    assert model.evidence_strength("design") == "direct"
    assert model.evidence_basis("design") == "config override"
    assert model.evidence_stale is False


def test_a_category_with_no_evidence_at_all_falls_back_to_general(tmp_path):
    model = ModelInfo("m", "p", "m", Prices(1.0, 2.0), capability={"general": 42.0},
                      capability_basis={"general": "aa_intelligence_index x 1.5"},
                      capability_strength={"general": "derived"})
    assert model.cap("design") == 42.0
    assert model.evidence_strength("design") == "none"
    assert "fallback: general" in model.evidence_basis("design")


# -- the discount -----------------------------------------------------------
def _m(name, strength, value=80.0):
    return ModelInfo(name, "p", name, Prices(1.0, 2.0), capability={"design": value, "general": value},
                     capability_basis={"design": "b"}, capability_strength={"design": strength})


def test_evidence_discount_shrinks_weak_evidence_toward_neutral():
    direct, derived, weak = _m("a", "direct"), _m("b", "derived"), _m("c", "weak")
    assert direct.cap("design", evidence_discount=0.5) == 80.0
    assert 50.0 < weak.cap("design", evidence_discount=0.5) < derived.cap("design", evidence_discount=0.5) < 80.0


def test_evidence_discount_of_zero_is_the_previous_behaviour():
    for strength in ("direct", "derived", "weak"):
        assert _m("m", strength).cap("design") == 80.0


def test_the_discount_never_pushes_a_weak_score_past_neutral():
    low = _m("low", "weak", value=10.0)
    assert 10.0 < low.cap("design", evidence_discount=1.0) <= 50.0
