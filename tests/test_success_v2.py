"""V2 curve guarantees, measured bucket limits, parameter trust and v1 compatibility."""
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
def test_v2_curve_is_monotone_in_capability_and_difficulty(category):
    s = success_v2()
    caps = [20, 30, 40, 50, 60, 70, 80, 90]
    for d in (0.0, 0.2, 0.5, 0.8, 1.0):
        ps = [s.curve(mk("m", c, 8000), category, d, 16000) for c in caps]
        assert all(b > a for a, b in zip(ps, ps[1:])), (category, d, ps)
    m = mk("m", 60, 8000)
    ds = [i / 10 for i in range(11)]
    ps = [s.curve(m, category, d, 16000) for d in ds]
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


def test_curve_monotone_but_measured_probability_can_rise_at_bucket_boundary():
    import json
    from pathlib import Path
    rows = json.loads((Path(__file__).resolve().parents[1] /
                       'evaluation/antseed-irp-eval-20261017/success.measured.json').read_text())
    s = success_v2(measured={(m, c, b): p for m, c, b, p in rows})
    m = mk('claude-opus-5', 60, 8000)
    ds = sorted({i / 1000 for i in range(1001)} | {0.34 - 1e-6, 0.67 - 1e-6})
    curves = [s.curve(m, 'knowledge', d, 16000) for d in ds]
    assert all(b <= a for a, b in zip(curves, curves[1:]))
    before = s.p(m, 'knowledge', 0.34 - 1e-6, 16000)
    after = s.p(m, 'knowledge', 0.34, 16000)
    assert before == pytest.approx(0.8934701594828719, abs=1e-14)
    assert after == pytest.approx(0.9004697196356551, abs=1e-14)
    assert after > before  # documented bucket granularity; final p() is not monotone


@pytest.mark.parametrize('kind', ['absolute', 'relative', 'symlink'])
def test_params_path_cannot_escape_package(tmp_path, monkeypatch, kind):
    import json
    from pathlib import Path
    import auto_router
    from auto_router.router import success_model_from_config
    root = Path(auto_router.__file__).resolve().parent
    outside = tmp_path / 'external.json'
    raw = json.loads((root / 'data/success_v2.json').read_text())
    raw['params']['offset'] = 424242
    outside.write_text(json.dumps(raw))
    if kind == 'symlink':
        path = root / '_test_external_params.json'
        path.symlink_to(outside)
    elif kind == 'relative':
        monkeypatch.chdir(tmp_path)
        path = Path('external.json')
    else:
        path = outside
    try:
        conf = {'success': {'version': 2, 'params': str(path)}}
        with pytest.raises(ValueError, match='must stay inside auto_router'):
            success_model_from_config(conf)
        for false_opt_in in (False, 'true', 1):
            conf['success']['allow_external_params'] = false_opt_in
            with pytest.raises(ValueError, match='must stay inside auto_router'):
                success_model_from_config(conf)
        conf['success']['allow_external_params'] = True
        assert success_model_from_config(conf).offset == 424242
    finally:
        if kind == 'symlink':
            path.unlink()


def test_packaged_default_and_explicit_in_package_paths(monkeypatch):
    from pathlib import Path
    import auto_router
    from auto_router.router import success_model_from_config
    root = Path(auto_router.__file__).resolve().parent
    default = success_v2()
    monkeypatch.chdir(root.parent)
    for path in (root / 'data/success_v2.json', Path('auto_router/data/success_v2.json')):
        loaded = success_model_from_config({'success': {'version': 2, 'params': str(path)}})
        assert loaded.fit_info['file'] == str(path)  # preserve original provenance spelling
        loaded.fit_info = default.fit_info
        assert loaded == default
    link = root / '_test_internal_params.json'
    link.symlink_to(root / 'data/success_v2.json')
    try:
        assert success_v2(path=link).offset == default.offset
    finally:
        link.unlink()
