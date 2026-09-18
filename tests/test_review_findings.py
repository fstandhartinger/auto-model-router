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


# ===========================================================================
# Second review pass: the evaluation harness (experiments/, scripts/)
# ===========================================================================
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import graders, heldout, sandbox      # noqa: E402
from experiments.tasks_heldout import build_tasks      # noqa: E402

_BWRAP = pytest.mark.skipif(not Path(sandbox.BWRAP).exists(), reason="bwrap is not installed")


def _sandbox_or_skip():
    if not sandbox.preflight()["available"]:
        pytest.skip("sandbox unavailable")


CODING_TASK = next(t for t in build_tasks() if t["id"] == "coding-easy-runlength")


def _row(**kw):
    """One flat held-out ledger row, as ``heldout_run`` writes it."""
    base = dict(task_id="t", category="math", arm="router", passed=True, grader="math",
                grader_kind="exact", detail="", model="m", latency_ms=100.0, prompt_tokens=10,
                cached_tokens=0, output_tokens=10, observed_cost_usd=0.001,
                cost_basis="list", estimated_cost_usd=0.002, estimated_p_success=0.8,
                cache_status="cold", evidence_confidence=0.9, safe_fallback=None, label="live")
    base.update(kw)
    return base

CORRECT = """```python
def encode(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        n = j - i
        out.append(s[i] if n == 1 else s[i] + str(n))
        i = j
    return ''.join(out)
```"""


# -- A/high: the coding grader could be passed without implementing anything -
@_BWRAP
@pytest.mark.parametrize("attack,answer", [
    ("prints the old success marker", "```python\nprint('ALL_TESTS_PASSED')\n```"),
    ("exits zero before the tests run", "```python\nimport os\nos._exit(0)\n```"),
    ("raises SystemExit(0)", "```python\nimport sys\nsys.exit(0)\n```"),
    ("echoes the runner then exits", "```python\nprint(open('main.py').read())\n"
                                     "import os\nos._exit(0)\n```"),
    ("returns the input unchanged", "```python\ndef encode(s):\n    return s\n```"),
])
def test_the_coding_grader_cannot_be_passed_without_solving_the_task(attack, answer):
    """Finding: with the answer and the assertions concatenated into one script
    and the verdict taken from a fixed stdout marker, `print('ALL_TESTS_PASSED')`
    passed every coding task, and `os._exit(0)` passed on the exit code."""
    _sandbox_or_skip()
    passed, detail = graders.grade_coding(CODING_TASK, answer)
    assert not passed, f"{attack} was accepted: {detail}"


@_BWRAP
def test_a_correct_answer_still_passes_and_says_how_it_was_verified():
    _sandbox_or_skip()
    passed, detail = graders.grade_coding(CODING_TASK, CORRECT)
    assert passed and "nonce" in detail


@_BWRAP
def test_the_verdict_nonce_differs_between_runs():
    """A fixed marker can be learned; a per-run nonce cannot."""
    _sandbox_or_skip()
    seen = set()
    real_run = sandbox.run_python

    def capture(code, *, stdin="", **kw):
        seen.add(stdin.strip())
        return real_run(code, stdin=stdin, **kw)

    original = graders.sandbox.run_python
    graders.sandbox.run_python = capture
    try:
        graders.grade_coding(CODING_TASK, CORRECT)
        graders.grade_coding(CODING_TASK, CORRECT)
    finally:
        graders.sandbox.run_python = original
    assert len(seen) == 2 and all(len(s) == 32 for s in seen)


# -- A/high: a failed upstream call must count as a failure ------------------
def test_a_failed_route_call_counts_as_a_failure_not_an_exclusion(tmp_path):
    """Finding: every call with ok=False was given passed=None and dropped from
    the denominator, so a provider failing half its requests had those failures
    removed from its own pass rate."""
    heldout.preregister(tmp_path)
    rows = [_row(task_id="a", passed=True),
            _row(task_id="b", passed=False, detail="the route failed the call: HTTP 503"),
            _row(task_id="c", passed=None, detail="SANDBOX UNAVAILABLE: no user namespaces")]
    (tmp_path / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    data = heldout.report(tmp_path)
    entry = data["per_category"]["math"]["router"]
    assert entry["n"] == 2 and entry["passed"] == 1, "the 503 stays in the denominator"
    assert data["excluded_rows"] == 1, "only the harness failure is excluded"


# -- A/high: spend must include calls whose answer was not graded -----------
def test_spend_includes_attempts_that_were_excluded_from_grading(tmp_path):
    """Finding: cost was summed over graded rows only, so a billable call whose
    answer was truncated or ungradable vanished from the total."""
    heldout.preregister(tmp_path)
    rows = [_row(task_id="a", passed=True, observed_cost_usd=0.01),
            _row(task_id="b", passed=None, observed_cost_usd=0.02,
                 detail="TRUNCATED: hit the 4000-token output budget")]
    (tmp_path / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    entry = heldout.report(tmp_path)["per_category"]["math"]["router"]
    assert entry["n"] == 1, "only one row was graded"
    assert entry["attempted"] == 2
    assert entry["metered_cost_usd"] == pytest.approx(0.03), "both calls were billed"


# -- B/high: the experiment ledger must not carry response text verbatim ----
def test_the_experiment_ledger_scrubs_grader_detail():
    """Finding: grader details quote the model's answer, and asdict() wrote them
    to the ledger verbatim - so a response containing a key put it there."""
    from experiments import heldout_run
    secret = "sk-" + "ant-api03-" + "R" * 24
    scrubbed = heldout_run._detail(f"got 'my key is {secret}' expected '36'")
    assert secret not in scrubbed and "[REDACTED]" in scrubbed
    assert len(heldout_run._detail("x" * 5000)) <= heldout_run.DETAIL_CHARS


def test_a_provider_error_message_is_reduced_to_its_shape():
    from experiments import heldout_run
    label = heldout_run._error_label("HTTP 401: {\"message\": \"bad key sk-live-abcdef, "
                                     "while handling: Refactor the billing module\"}")
    assert label == "HTTP 401"
    assert heldout_run._error_label(None) == "unknown"


# -- A/medium: the design grader let several external references through ----
@pytest.mark.parametrize("markup", [
    '<link rel="stylesheet" href="https://cdn.example/x.css">',
    '<script src="//cdn.example/x.js"></script>',
    '<style>@import url("https://fonts.example/f.css");</style>',
    '<style>body{background:url(https://cdn.example/bg.png)}</style>',
    '<iframe src="https://example.com/widget"></iframe>',
    '<img src="https://cdn.example/logo.png">',
])
def test_the_design_grader_rejects_every_form_of_external_reference(markup):
    """Finding: only absolute `link href` and `script src` were checked, so
    @import, url(//cdn), an iframe and an external image all passed."""
    page = f"<!doctype html><html><head>{markup}</head><body><main></main></body></html>"
    passed, detail = graders.grade_design({"rules": []}, f"```html\n{page}\n```")
    assert not passed, detail
    assert "self-contained" in detail


def test_the_design_grader_still_accepts_a_genuinely_self_contained_page():
    page = ('<!doctype html><html><head><style>:root{--a:#07f}'
            'body{background:url("data:image/gif;base64,R0lGOD")}</style></head>'
            '<body><main><h1>x</h1><img src="/local.png"></main></body></html>')
    passed, detail = graders.grade_design({"rules": []}, f"```html\n{page}\n```")
    assert passed, detail


# -- C/medium: pre-registration integrity beyond the task file --------------
def test_changing_a_grader_after_registration_is_detected(tmp_path, monkeypatch):
    """Finding: only the task list was hashed, so a grader could be loosened
    after outcomes were seen without the digest changing."""
    record = heldout.preregister(tmp_path)
    assert record["code_sha256"]["graders.py"]
    monkeypatch.setitem(record["code_sha256"], "graders.py", "0" * 64)
    (tmp_path / "preregistration.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit) as exc:
        heldout.load_preregistration(tmp_path)
    assert "graders.py" in str(exc.value)


def test_a_harness_change_can_be_amended_on_the_record(tmp_path):
    record = heldout.preregister(tmp_path)
    stale = dict(record)
    stale["code_sha256"] = {**record["code_sha256"], "graders.py": "0" * 64}
    (tmp_path / "preregistration.json").write_text(json.dumps(stale))
    heldout.amend(tmp_path, "grader could be passed without solving the task")
    fresh = heldout.load_preregistration(tmp_path)
    assert len(fresh["amendments"]) == 1
    assert "without solving" in fresh["amendments"][0]["reason"]
    assert fresh["amendments"][0]["previous_code_sha256"]["graders.py"] == "0" * 64


def test_an_amendment_without_a_reason_is_refused():
    with pytest.raises(SystemExit) as exc:
        heldout.main(["amend", "--dir", "/tmp/does-not-matter"])
    assert "reason is required" in str(exc.value)


def test_the_report_names_any_unrecorded_drift(tmp_path):
    record = heldout.preregister(tmp_path)
    record["code_sha256"]["graders.py"] = "0" * 64
    (tmp_path / "preregistration.json").write_text(json.dumps(record))
    (tmp_path / "ledger.jsonl").write_text(json.dumps(_row()))
    data = heldout.report(tmp_path)
    assert "graders.py" in data["harness_drift_since_registration"]
    assert "UNRECORDED harness drift" in heldout.format_report(data)
