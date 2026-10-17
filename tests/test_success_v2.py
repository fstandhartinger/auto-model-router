"""Success model v2: v1 stays the default; v2 is monotone, keeps strong models apart, shrinks measured cells, and charges truncation risk."""
import math

import pytest

from auto_router.catalog import ModelInfo, Prices
from auto_router.economics import SuccessModel, success_v2

CATEGORIES = ("coding", "math", "general", "knowledge")


def mk(name, cap, tokens=None, key=None):
    m = ModelInfo(name=name, provider="p", upstream_id=name, prices=Prices(1.0, 2.0),
                  capability={c: cap for c in CATEGORIES}, task_tokens=tokens)
    return m.with_(success_key=key) if key else m


def test_v1_defaults_are_unchanged():
    s = SuccessModel()
    assert (s.offset, s.slope, s.scale, s.ceiling, s.floor) == (20.0, 50.0, 6.0, 0.98, 0.02)
    assert s.measured_prior_n is None and s.length_penalty == 0.0
    strong, mid = mk("strong", 90), mk("mid", 70)
    assert s.p(strong, "coding", 0.2) == s.p(mid, "coding", 0.2) == 0.98   # the v1 ceiling tie that v2 removes
    assert s.p(strong, "coding", 0.2, max_tokens=100) == 0.98              # max_tokens is ignored by v1


def test_v1_measured_cell_still_overrides_the_curve():
    s = SuccessModel(measured={("m", "coding", "hard"): 0.33})
    assert s.p(mk("m", 90), "coding", 0.9) == 0.33


@pytest.mark.parametrize("category", CATEGORIES)
def test_v2_is_monotone_in_capability_and_difficulty(category):
    s = success_v2()
    caps = [20, 30, 40, 50, 60, 70, 80, 90]
    for d in (0.0, 0.2, 0.5, 0.8, 1.0):
        ps = [s.p(mk("m", c, 8000), category, d, 16000) for c in caps]
        assert all(b > a for a, b in zip(ps, ps[1:])), (category, d, ps)
    m = mk("m", 60, 8000)
    ds = [i / 10 for i in range(11)]
    ps = [s.p(m, category, d, 16000) for d in ds]
    assert all(b < a for a, b in zip(ps, ps[1:])), (category, ps)
    assert all(0.0 < p < 1.0 for p in ps)


def test_v2_keeps_strong_models_distinguishable_where_v1_saturates():
    v1, v2 = SuccessModel(), success_v2()
    strong, mid = mk("strong", 75, 8000), mk("mid", 60, 8000)
    assert v1.p(strong, "coding", 0.1) == v1.p(mid, "coding", 0.1)
    assert v2.p(strong, "coding", 0.1, 16000) > v2.p(mid, "coding", 0.1, 16000)


def test_v2_charges_truncation_risk_against_the_token_budget():
    s = success_v2()
    thrifty, verbose = mk("thrifty", 60, 4000), mk("verbose", 60, 80000)
    assert s.p(thrifty, "math", 0.5, 16000) > s.p(verbose, "math", 0.5, 16000)
    # a bigger budget removes (part of) the penalty; no budget means the model's own output limit
    assert s.p(verbose, "math", 0.5, 32000) > s.p(verbose, "math", 0.5, 4000)
    assert s.p(verbose, "math", 0.5) == s.p(verbose, "math", 0.5, verbose.max_output_tokens)
    # unknown tokens-per-task: the term is off, never a guess
    unknown = mk("unknown", 60, None)
    assert s.p(unknown, "math", 0.5, 4000) == s.p(unknown, "math", 0.5, 32000)


def test_measured_cells_shrink_toward_the_curve_instead_of_replacing_it():
    cell = {("m", "coding", "medium"): 0.5}
    m = mk("m", 60, 8000)
    curve = success_v2().curve(m, "coding", 0.5, 16000)
    s = success_v2(measured=cell)
    blended = s.p(m, "coding", 0.5, 16000)
    n_prior, n_obs = s.measured_prior_n, s.measured_n
    assert blended == pytest.approx((n_prior * curve + n_obs * 0.5) / (n_prior + n_obs))
    assert min(curve, 0.5) < blended < max(curve, 0.5)
    s.measured_prior_n = 0.0
    assert s.p(m, "coding", 0.5, 16000) == pytest.approx(0.5)      # no prior = the cell alone
    s.measured_prior_n = 1e9
    assert s.p(m, "coding", 0.5, 16000) == pytest.approx(curve, abs=1e-6)
    # cells for other models/buckets are ignored
    assert s.p(mk("other", 60, 8000), "coding", 0.5, 16000) == success_v2().p(mk("other", 60, 8000), "coding", 0.5, 16000)


def test_params_file_round_trips_and_is_documented():
    import json
    from pathlib import Path
    import auto_router
    raw = json.loads((Path(auto_router.__file__).parent / "data" / "success_v2.json").read_text())
    assert raw["version"] == "success-v2" and raw["fitted_on"]["n"] == 720
    p = raw["params"]
    s = success_v2()
    assert (s.offset, s.slope, s.scale, s.ceiling, s.length_penalty) == (p["offset"], p["slope"], p["scale"], p["ceiling"], p["length_penalty"])
    assert s.smooth_ceiling and set(s.per_category) == {"coding", "math"} and s.scale > 0 and s.length_penalty >= 0
    assert all(sl > 0 and sc > 0 for _, sl, sc in s.per_category.values()) and s.slope > 0


def test_success_model_from_config_selects_v2():
    from auto_router.router import success_model_from_config
    assert success_model_from_config({"success": {"version": 2}}).length_penalty > 0
    assert success_model_from_config({"success": {}}).length_penalty == 0.0
    assert success_model_from_config({"success": {"offset": 10}}).offset == 10


def test_fit_script_predictions_match_the_library():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    import fit_success_v2 as F
    rng = np.random.default_rng(1)
    n = 400
    data = {"cap": rng.uniform(30, 85, n), "d": rng.uniform(0, 1, n), "u": rng.uniform(-2, 3, n), "group": rng.integers(0, 3, n)}
    data["y"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.04 * (data["cap"] - 60) + 2 - 3 * data["d"] - 0.3 * data["u"])))).astype(int)
    th = F.fit(data)
    assert th["kappa"] >= 0 and min(th["g"]) >= 0 and th["lam"] >= 0 and 0.90 - 1e-9 <= th["ceiling"] <= 0.995 + 1e-9
    P = F.to_params(th)
    s = SuccessModel(offset=P["offset"], slope=P["slope"], scale=P["scale"], floor=P["floor"], ceiling=P["ceiling"],
                     per_category={k: tuple(v) for k, v in P["per_category"].items()}, length_penalty=P["length_penalty"],
                     smooth_ceiling=True, evidence_discount=0.0)
    cats = {0: "general", 1: "coding", 2: "math"}
    for i in range(0, n, 37):
        tt, mt = 8000.0 * math.exp(data["u"][i]), 8000   # u = ln(task_tokens / budget) with budget 8000
        m = mk("x", float(data["cap"][i]), tt)
        want = F.predict(th, data["cap"][i:i + 1], data["d"][i:i + 1], data["u"][i:i + 1], data["group"][i:i + 1])[0]
        assert s.curve(m, cats[int(data["group"][i])], float(data["d"][i]), mt) == pytest.approx(want, abs=1e-4)
