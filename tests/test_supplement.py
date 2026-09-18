"""The pre-registered held-out **supplement**: its task set, its own frozen
registration, valid-pair accounting and the combined report.

The supplement exists because the original 27-task ledger cannot reach ten
valid *paired* tasks in every category - four of its twelve design rows are
harness truncations. The original ledger is immutable evidence, so the
supplement is a second, separately pre-registered run that is *combined* with
it for reporting and never merged into it.

Everything here is offline. No test may call a model, and the supplement's
registration must make it impossible to run after an unrecorded change to
anything that decides an outcome - including the routing policy and the
catalog it is run against.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import graders, heldout, pairing, supplement       # noqa: E402
from experiments.tasks_supplement import build_tasks                # noqa: E402


# ---------------------------------------------------------------------------
# the supplement task set
# ---------------------------------------------------------------------------
def test_the_supplement_covers_exactly_the_six_required_categories():
    categories = {t["category"] for t in build_tasks()}
    assert categories == set(heldout.CATEGORIES)


def test_the_supplement_is_deterministic_and_rebuilds_byte_for_byte():
    import random
    first = build_tasks(random.Random(supplement.SEED))
    second = build_tasks(random.Random(supplement.SEED))
    assert [json.dumps(t, sort_keys=True) for t in first] == \
           [json.dumps(t, sort_keys=True) for t in second]


def test_supplement_task_ids_are_unique_and_never_collide_with_the_original():
    from experiments.tasks_heldout import build_tasks as original
    ids = [t["id"] for t in build_tasks()]
    assert len(ids) == len(set(ids))
    assert not (set(ids) & {t["id"] for t in original()}), \
        "a supplement id that collides with the original would silently overwrite evidence"


def test_every_supplement_task_has_a_registered_grader_and_a_difficulty():
    for task in build_tasks():
        assert task["grader"] in graders.GRADERS
        assert task["category"] in heldout.CATEGORIES
        assert task["prompt"] and task["system"]
        assert task["difficulty"] in {"easy", "medium", "hard"}


def test_each_category_has_enough_tasks_to_reach_ten_valid_pairs():
    """Counts are chosen before any outcome is seen, with headroom where the
    original run already showed the harness loses rows."""
    counts = {}
    for task in build_tasks():
        counts[task["category"]] = counts.get(task["category"], 0) + 1
    for category, needed in supplement.SUPPLEMENT_MINIMUM.items():
        assert counts[category] >= needed, (category, counts[category], needed)


def test_supplement_cache_repeats_share_one_long_prefix_and_keep_their_order():
    tasks = [t for t in build_tasks() if t["category"] == "cache_repeat"]
    assert len({t["shared_prefix"] for t in tasks}) == 1
    assert tasks[0]["repeat_of"] is None
    assert all(t["repeat_of"] == tasks[0]["id"] for t in tasks[1:])
    assert len(tasks[0]["prompt"]) > 4000


def test_no_supplement_task_carries_a_credential_shape():
    from auto_router import jev
    for task in build_tasks():
        assert not jev.scrub_report(task["prompt"] + task["system"]), task["id"]


def test_every_supplement_grader_passes_a_reference_answer_and_fails_a_wrong_one():
    """A task whose own grader cannot be satisfied would be an un-passable
    task, and a task whose grader passes anything would be no measurement at
    all. Both are checked here, offline, before the set is registered.

    The coding tasks are covered separately, because they need the sandbox.
    """
    from experiments.tasks_supplement import REFERENCE_ANSWERS
    for task in build_tasks():
        if task["category"] == "coding":
            continue
        good = REFERENCE_ANSWERS[task["id"]]
        passed, detail = graders.grade(task, good)
        assert passed, (task["id"], detail)
        bad_passed, _ = graders.grade(task, "I am not going to answer that.")
        assert not bad_passed, task["id"]


@pytest.mark.skipif(not __import__("experiments.sandbox", fromlist=["x"]).preflight()["available"],
                    reason="Bubblewrap is unavailable; coding answers are never run unisolated")
def test_every_supplement_coding_task_passes_its_own_reference_solution():
    from experiments.tasks_supplement import REFERENCE_ANSWERS
    for task in build_tasks():
        if task["category"] != "coding":
            continue
        passed, detail = graders.grade(task, REFERENCE_ANSWERS[task["id"]])
        assert passed, (task["id"], detail)


# ---------------------------------------------------------------------------
# valid-pair accounting
# ---------------------------------------------------------------------------
def _row(task_id, category, arm, passed, detail=""):
    return {"task_id": task_id, "category": category, "arm": arm,
            "passed": passed, "detail": detail, "model": "m", "grader_kind": "exact",
            "observed_cost_usd": None, "latency_ms": 1.0}


def test_a_pair_is_valid_only_when_both_arms_were_graded():
    rows = [
        _row("t1", "math", "router", True), _row("t1", "math", "control", True),
        _row("t2", "math", "router", False), _row("t2", "math", "control", True),
        _row("t3", "math", "router", None, "TRUNCATED: hit the budget"),
        _row("t3", "math", "control", True),
        _row("t4", "math", "router", True),
        _row("t4", "math", "control", None, "SANDBOX UNAVAILABLE: no bwrap"),
        _row("t5", "math", "router", True),          # comparator row missing entirely
    ]
    acc = pairing.pair_accounting(rows, "router", "control")["math"]
    assert acc["attempted_pairs"] == 5
    assert acc["valid_pairs"] == 2
    assert acc["invalid_truncated"] == 1
    assert acc["invalid_sandbox"] == 1
    assert acc["invalid_incomplete"] == 1
    assert acc["router_passed"] == 1 and acc["comparator_passed"] == 2


def test_a_failed_provider_call_is_a_valid_pair_and_a_failure_not_an_exclusion():
    rows = [_row("t1", "coding", "router", False, "the route failed the call: HTTP 503"),
            _row("t1", "coding", "control", True)]
    acc = pairing.pair_accounting(rows, "router", "control")["coding"]
    assert acc["valid_pairs"] == 1
    assert acc["router_passed"] == 0
    assert acc["invalid_truncated"] == 0


def test_pair_accounting_never_counts_a_row_twice_across_two_ledgers():
    rows = [_row("t1", "math", "router", True), _row("t1", "math", "control", True)]
    acc = pairing.pair_accounting(rows + rows, "router", "control")["math"]
    assert acc["attempted_pairs"] == 1 and acc["valid_pairs"] == 1


def test_pair_accounting_reports_every_required_category_even_when_empty():
    acc = pairing.pair_accounting([], "router", "control")
    assert set(acc) == set(heldout.CATEGORIES)
    assert all(v["valid_pairs"] == 0 for v in acc.values())


# ---------------------------------------------------------------------------
# the frozen supplement registration
# ---------------------------------------------------------------------------
def test_the_registration_freezes_policy_and_catalog_identity(tmp_path):
    config = supplement.load_frozen_config(_write_config(tmp_path))
    identity = supplement.policy_identity(config)
    assert identity["policy_name"] == "F_expected"
    assert identity["policy_source"] in ("config", "environment default")
    assert [m["name"] for m in identity["catalog"]] == ["free-one", "metered-one"]
    assert identity["catalog"][0]["free"] is True
    assert "capability" in identity["catalog"][0]
    assert identity["sha256"] == supplement.policy_identity(config)["sha256"]


def test_registration_refuses_to_run_on_unrecorded_task_drift(tmp_path):
    out = tmp_path / "supp"
    supplement.preregister(out, _write_config(tmp_path))
    (out / "tasks.jsonl").write_text('{"id": "tampered"}\n')
    with pytest.raises(SystemExit, match="digest changed"):
        supplement.load_preregistration(out)


def test_registration_refuses_to_run_on_unrecorded_policy_or_catalog_drift(tmp_path):
    out = tmp_path / "supp"
    config_path = _write_config(tmp_path)
    supplement.preregister(out, config_path)
    record = json.loads((out / "preregistration.json").read_text())
    record["policy_identity_sha256"] = "0" * 64
    (out / "preregistration.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="policy/catalog"):
        supplement.load_preregistration(out, config_path=config_path)


def test_registration_refuses_to_run_on_unrecorded_code_drift(tmp_path):
    out = tmp_path / "supp"
    supplement.preregister(out, _write_config(tmp_path))
    record = json.loads((out / "preregistration.json").read_text())
    record["code_sha256"]["graders.py"] = "0" * 64
    (out / "preregistration.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="changed since pre-registration"):
        supplement.load_preregistration(out)


def test_the_registration_names_the_original_it_supplements(tmp_path):
    out = tmp_path / "supp"
    record = supplement.preregister(out, _write_config(tmp_path))
    assert record["supplements"]["task_file_sha256"].startswith("8473101053")
    assert record["supplements"]["task_count"] == 27
    assert record["target_valid_pairs_per_category"] == 10


def test_the_registration_is_written_before_any_call_and_is_reproducible(tmp_path):
    a = supplement.preregister(tmp_path / "a", _write_config(tmp_path))
    b = supplement.preregister(tmp_path / "b", _write_config(tmp_path))
    assert a["task_file_sha256"] == b["task_file_sha256"]
    assert a["analysis_plan_sha256"] == b["analysis_plan_sha256"]
    assert not (tmp_path / "a" / "ledger.jsonl").exists()


# ---------------------------------------------------------------------------
# the budget guard
# ---------------------------------------------------------------------------
def test_the_budget_guard_fails_closed_before_the_absolute_cap(tmp_path):
    ledger = tmp_path / "calls.jsonl"
    ledger.write_text(json.dumps({"cost_usd": 0.5, "list_cost_usd": 0.2}) + "\n")
    guard = supplement.BudgetGuard(prior_usd=21.2611, cap_usd=30.0, ledgers=[ledger])
    assert guard.total() == pytest.approx(21.7611)
    assert guard.remaining() == pytest.approx(8.2389)
    guard.assert_headroom(1.0)
    with pytest.raises(supplement.BudgetExceeded):
        guard.assert_headroom(9.0)


def test_the_budget_guard_counts_the_higher_of_billed_and_list_cost(tmp_path):
    ledger = tmp_path / "calls.jsonl"
    ledger.write_text("\n".join([json.dumps({"cost_usd": 0.0, "list_cost_usd": 0.3}),
                                 json.dumps({"cost_usd": 0.4, "list_cost_usd": 0.1})]) + "\n")
    guard = supplement.BudgetGuard(prior_usd=0.0, cap_usd=30.0, ledgers=[ledger])
    assert guard.total() == pytest.approx(0.7)


def test_the_budget_guard_ignores_a_missing_ledger_but_not_a_broken_line(tmp_path):
    ledger = tmp_path / "calls.jsonl"
    ledger.write_text('{"cost_usd": 0.1, "list_cost_usd": 0.0}\nnot json\n')
    guard = supplement.BudgetGuard(prior_usd=0.0, cap_usd=1.0,
                                  ledgers=[ledger, tmp_path / "absent.jsonl"])
    assert guard.total() == pytest.approx(0.1)
    assert guard.unparsable == 1


# ---------------------------------------------------------------------------
# the combined report
# ---------------------------------------------------------------------------
def _ledger(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_the_combined_report_cites_both_registrations(tmp_path):
    original, supp = _two_runs(tmp_path)
    data = supplement.combined_report(original, supp)
    assert data["registrations"][0]["role"] == "original"
    assert data["registrations"][1]["role"] == "supplement"
    assert len(data["registrations"]) == 2


def test_the_combined_report_counts_valid_pairs_per_category_and_says_which_fall_short(tmp_path):
    original, supp = _two_runs(tmp_path)
    data = supplement.combined_report(original, supp)
    math = data["per_category"]["math"]["router_vs_control"]
    assert math["valid_pairs"] == 2
    assert "math" in data["categories_below_ten_valid_pairs"]["router_vs_control"]


def test_the_combined_report_never_claims_quality_below_the_registered_floor(tmp_path):
    original, supp = _two_runs(tmp_path)
    data = supplement.combined_report(original, supp)
    assert data["quality_claim_supported"] == []
    text = supplement.format_combined(data)
    assert "No quality claim" in text


def test_the_combined_report_keeps_the_two_ledgers_distinguishable(tmp_path):
    original, supp = _two_runs(tmp_path)
    data = supplement.combined_report(original, supp)
    assert data["rows_by_source"] == {"original": 4, "supplement": 4}


def test_the_combined_report_separates_free_routes_from_metered_ones(tmp_path):
    original, supp = _two_runs(tmp_path)
    data = supplement.combined_report(original, supp)
    arms = data["arms"]
    assert arms["router"]["cost_measurable"] is False
    assert arms["control-metered"]["cost_measurable"] is True
    assert arms["control-metered"]["measured_usd"] == pytest.approx(0.02)
    assert arms["router"]["is_real_normal_policy"] is True
    assert arms["control"]["is_real_normal_policy"] is False


def test_the_combined_report_refuses_to_call_a_static_control_the_normal_policy(tmp_path):
    original, supp = _two_runs(tmp_path)
    data = supplement.combined_report(original, supp)
    text = supplement.format_combined(data)
    assert "one fixed route for everything" in text
    assert data["arms"]["control"]["description"] != data["arms"]["router"]["description"]


def test_the_combined_report_reports_truncated_rows_rather_than_hiding_them(tmp_path):
    original, supp = _two_runs(tmp_path)
    data = supplement.combined_report(original, supp)
    assert data["per_category"]["design"]["router_vs_control"]["invalid_truncated"] == 1
    assert "TRUNCATED" in supplement.format_combined(data)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "providers:\n"
        "  p:\n"
        "    base_url: https://example.invalid/v1\n"
        "    api_key_env: DOES_NOT_EXIST\n"
        "models:\n"
        "- name: free-one\n"
        "  provider: p\n"
        "  upstream_id: vendor/free-one\n"
        "  free: true\n"
        "- name: metered-one\n"
        "  provider: p\n"
        "  upstream_id: vendor/metered-one\n"
        "  prices:\n"
        "    input: 1.0\n"
        "    output: 2.0\n")
    return path


def _two_runs(tmp_path: Path) -> tuple[Path, Path]:
    original, supp = tmp_path / "orig", tmp_path / "supp"
    _ledger(original / "ledger.jsonl", [
        _row("o1", "math", "router", True), _row("o1", "math", "control", True),
        _row("o2", "design", "router", None, "TRUNCATED: hit the 12000-token output budget"),
        _row("o2", "design", "control", True),
    ])
    (original / "preregistration.json").write_text(json.dumps({
        "registered_at": "2026-09-18T06:42:03Z", "task_file_sha256": "84731010531b266f" + "0" * 48,
        "task_count": 27, "amendments": [], "task_file": "tasks.jsonl"}))
    (original / "tasks.jsonl").write_text("")
    metered = dict(_row("s1", "math", "control-metered", True), observed_cost_usd=0.02)
    _ledger(supp / "ledger.jsonl", [
        _row("s1", "math", "router", True), _row("s1", "math", "control", True), metered,
        _row("s2", "design", "router", True),
    ])
    (supp / "preregistration.json").write_text(json.dumps({
        "registered_at": "2026-09-18T10:40:00Z", "task_file_sha256": "ab" * 32,
        "task_count": 60, "amendments": [], "task_file": "tasks.jsonl",
        "policy_identity_sha256": "cd" * 32,
        "supplements": {"task_file_sha256": "84731010531b266f" + "0" * 48, "task_count": 27}}))
    (supp / "tasks.jsonl").write_text("")
    return original, supp


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------
def test_the_serialised_router_keeps_one_router_and_serialises_its_state_changes():
    """Workers share one router - a per-worker router would be a different
    system from the one being measured - and only its state transitions are
    held under the lock, so provider calls still overlap."""
    import threading
    import time as _time

    class Fake:
        name = "F_expected"

        def __init__(self):
            self.inside = 0
            self.max_inside = 0
            self.calls = 0

        def route(self, *a, **kw):
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
            _time.sleep(0.01)
            self.inside -= 1
            self.calls += 1
            return "chosen"

        def commit(self, *a, **kw):
            return None

        def observe(self, *a, **kw):
            return None

    real = Fake()
    guarded = supplement.SerialisedRouter(real)
    assert guarded.name == "F_expected"          # forwards everything else
    threads = [threading.Thread(target=guarded.route) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert real.calls == 8
    assert real.max_inside == 1, "two workers were inside the router's state at once"


def test_each_invocation_keeps_its_own_run_meta_file(tmp_path):
    """Two arms run as two invocations against the same directory. A single
    overwritten meta file would lose the first arm's settings."""
    meta = {"started_at": "2026-09-18T10:30:00Z", "arms": ["router"]}
    path = tmp_path / "run-meta-20260918T103000Z-router.json"
    supplement._write_meta(tmp_path, path, meta)
    supplement._write_meta(tmp_path, tmp_path / "run-meta-b.json",
                           {"started_at": "2026-09-18T11:00:00Z", "arms": ["control"]})
    assert json.loads(path.read_text())["arms"] == ["router"]
    assert json.loads((tmp_path / "run-meta.json").read_text())["arms"] == ["control"]


# ---------------------------------------------------------------------------
# repairs after the independent review (2026-09-18)
# ---------------------------------------------------------------------------
def test_the_registration_also_freezes_the_product_code_that_chooses_routes(tmp_path):
    """Hashing only the experiment files leaves the thing under test unhashed.

    ``run_live`` imports ``Router`` and ``load_config``; those decide which
    route an arm uses. A reviewer pointed out that they could change while every
    registered digest still matched.
    """
    out = tmp_path / "supp"
    record = supplement.preregister(out, _write_config(tmp_path))
    assert "auto_router/router.py" in record["product_sha256"]
    assert "auto_router/policies.py" in record["product_sha256"]
    assert "auto_router/config.py" in record["product_sha256"]
    assert record["repo_git_sha"]


def test_registration_refuses_to_run_when_the_product_code_changed(tmp_path):
    out = tmp_path / "supp"
    supplement.preregister(out, _write_config(tmp_path))
    record = json.loads((out / "preregistration.json").read_text())
    record["product_sha256"]["auto_router/policies.py"] = "0" * 64
    (out / "preregistration.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="auto_router/policies.py"):
        supplement.load_preregistration(out)


def test_registration_refuses_to_run_when_the_config_file_changed(tmp_path):
    """The config digest was recorded and then never checked - a reviewer's finding."""
    config = _write_config(tmp_path)
    out = tmp_path / "supp"
    supplement.preregister(out, config)
    config.write_text(config.read_text() + "  vision: true\n")
    with pytest.raises(SystemExit, match="config file"):
        supplement.load_preregistration(out, config_path=config)


def test_duplicate_observations_are_reported_not_silently_collapsed():
    """``last write wins`` is the only sane rule for concatenating two ledgers,
    but a rerun that replaces an unfavourable row must be visible."""
    rows = [_row("t1", "math", "router", False), _row("t1", "math", "router", True),
            _row("t1", "math", "control", True)]
    acc = pairing.pair_accounting(rows, "router", "control")["math"]
    assert acc["valid_pairs"] == 1
    assert acc["duplicate_observations"] == 1
    assert acc["duplicate_task_ids"] == ["t1"]


def test_a_category_of_warm_prefix_repeats_is_flagged_as_not_independent():
    assert pairing.INDEPENDENT_SAMPLES["cache_repeat"] is False
    assert pairing.INDEPENDENT_SAMPLES["coding"] is True


def test_the_report_says_the_sample_floor_is_met_without_calling_that_a_quality_claim(tmp_path):
    original, supp = _two_runs(tmp_path)
    _ledger(supp / "ledger.jsonl", [_row(f"m{i}", "math", arm, arm == "control")
                                    for i in range(12) for arm in ("router", "control")])
    data = supplement.combined_report(original, supp)
    math = data["per_category"]["math"]["router_vs_control"]
    assert math["valid_pairs"] >= 10
    assert "math" in data["categories_meeting_the_sample_floor"]["router_vs_control"]
    # the router lost every discordant pair, so "at least as often" is NOT supported
    assert "math" not in data["quality_claim_supported"]
    assert math["router_at_least_as_good"] is False
    text = supplement.format_combined(data)
    assert "at least as often" in text


def test_the_report_breaks_a_category_down_by_the_declared_difficulty(tmp_path):
    original, supp = _two_runs(tmp_path)
    data = supplement.combined_report(original, supp,
                                      task_dirs=[Path(original), Path(supp)])
    assert "by_difficulty" in data["per_category"]["design"]["router_vs_control"]


def test_the_report_shows_the_supplement_alone_beside_the_pooled_figure(tmp_path):
    """The original registration was amended after outcomes were seen; a reader
    has to be able to see the un-amended registration's rows on their own."""
    original, supp = _two_runs(tmp_path)
    data = supplement.combined_report(original, supp)
    assert data["per_category"]["math"]["router_vs_control"]["supplement_only"]["valid_pairs"] == 1
    assert data["per_category"]["math"]["router_vs_control"]["valid_pairs"] == 2


def test_the_report_records_the_invocations_that_produced_the_rows(tmp_path):
    original, supp = _two_runs(tmp_path)
    (supp / "run-meta-20260918T110000Z-control.json").write_text(json.dumps(
        {"started_at": "2026-09-18T11:00:00Z", "arms": ["control"], "workers": 6,
         "control_model": "gpt-5.6-sol", "control_label": "control-metered",
         "run_budget_usd": 2.5}))
    data = supplement.combined_report(original, supp)
    assert data["invocations"][0]["control_model"] == "gpt-5.6-sol"
    assert data["invocations"][0]["workers"] == 6


def test_a_task_with_no_rows_at_all_is_still_counted_as_an_attempted_pair():
    """The attempted universe is the registered task list, not the rows that
    happen to exist. Deriving it from rows lets a task that was never run
    disappear instead of being reported as incomplete."""
    rows = [_row("t1", "math", "router", True), _row("t1", "math", "control", True)]
    manifest = [{"id": "t1", "category": "math"}, {"id": "t2", "category": "math"}]
    acc = pairing.pair_accounting(rows, "router", "control", manifest=manifest)["math"]
    assert acc["attempted_pairs"] == 2
    assert acc["valid_pairs"] == 1
    assert acc["invalid_incomplete"] == 1
    assert "t2" in acc["invalid_task_ids"]


def test_pair_accounting_reports_the_sensitivity_of_treating_truncation_as_failure():
    """The registered rule excludes a truncated pair. That rule was written
    after a truncation had already been seen, so the report has to show what
    the numbers look like under the opposite, harsher rule as well."""
    rows = [_row("t1", "design", "router", True), _row("t1", "design", "control", True),
            _row("t2", "design", "router", None, "TRUNCATED: hit the budget"),
            _row("t2", "design", "control", True)]
    acc = pairing.pair_accounting(rows, "router", "control")["design"]
    strict = acc["if_truncation_counted_as_failure"]
    assert strict["valid_pairs"] == 2
    assert strict["router_passed"] == 1
    assert strict["comparator_passed"] == 2


def test_the_paired_difference_carries_an_interval_not_only_a_p_value():
    entry = {"valid_pairs": 20, "router_only_passed": 1, "comparator_only_passed": 5}
    low, high = pairing.paired_difference_ci(entry)
    point = (1 - 5) / 20
    assert low < point < high
    assert -1.0 <= low <= high <= 1.0


def test_the_budget_guard_refuses_outright_when_a_ledger_line_is_unreadable(tmp_path):
    """A half-written cost row was counted as zero, which made headroom look
    larger than it was - the opposite of failing closed."""
    ledger = tmp_path / "calls.jsonl"
    ledger.write_text('{"cost_usd": 0.1, "list_cost_usd": 0.0}\nnot json\n')
    guard = supplement.BudgetGuard(prior_usd=0.0, cap_usd=30.0, ledgers=[ledger])
    with pytest.raises(supplement.BudgetExceeded, match="unreadable"):
        guard.assert_headroom(0.01)


def test_the_report_prices_an_arm_from_its_calls_not_only_from_its_retained_rows(tmp_path):
    """Discarding a row from the analysis ledger does not refund the call."""
    original, supp = _two_runs(tmp_path)
    (supp / "calls.jsonl").write_text("\n".join(
        json.dumps({"ts": 1, "model": "gpt-5.6-sol", "tag": "heldout:control:s1",
                    "ok": True, "cost_usd": c, "list_cost_usd": 0.0}) for c in (0.02, 0.05)) + "\n")
    data = supplement.combined_report(original, supp)
    spend = data["measured_spend"]
    assert spend["retained_rows_usd"] == pytest.approx(0.02)
    assert spend["all_calls_usd"] == pytest.approx(0.07)
    assert spend["discarded_or_extra_usd"] == pytest.approx(0.05)


def test_the_report_says_which_registration_froze_a_policy_identity(tmp_path):
    original, supp = _two_runs(tmp_path)
    data = supplement.combined_report(original, supp)
    assert data["policy_identity_frozen"] == {"original": False, "supplement": True}
    assert "not frozen" in supplement.format_combined(data)


def test_the_report_attributes_every_exclusion_to_its_arm(tmp_path):
    original, supp = _two_runs(tmp_path)
    data = supplement.combined_report(original, supp)
    assert data["exclusions_by_arm"]["router"]["design"] == 1


# ---------------------------------------------------------------------------
# repairs after the second independent review pass
# ---------------------------------------------------------------------------
def test_the_budget_guard_reserves_what_it_authorises_so_workers_cannot_race():
    """Six workers each saw the same remaining balance and all proceeded. The
    guard has to reserve what it hands out until the call is on the ledger."""
    guard = supplement.BudgetGuard(prior_usd=0.0, cap_usd=1.0, ledgers=[])
    guard.assert_headroom(0.6)
    with pytest.raises(supplement.BudgetExceeded):
        guard.assert_headroom(0.6)
    guard.release(0.6)
    guard.assert_headroom(0.6)


def test_a_reservation_is_released_even_when_the_call_raises(tmp_path):
    guard = supplement.BudgetGuard(prior_usd=0.0, cap_usd=1.0, ledgers=[])
    with pytest.raises(ValueError):
        with guard.reserve(0.9):
            raise ValueError("provider blew up")
    guard.assert_headroom(0.9)


def test_no_discordant_pairs_does_not_report_an_interval_of_exactly_zero():
    """[0, 0] from twelve concordant pairs reads as proven equality. It is not:
    it is the absence of any observed disagreement."""
    entry = {"valid_pairs": 12, "router_only_passed": 0, "comparator_only_passed": 0}
    low, high = pairing.paired_difference_ci(entry)
    assert low < 0 < high
    assert high > 0.2, "twelve pairs cannot exclude a twenty-point difference"


def test_the_paired_interval_still_brackets_the_point_estimate():
    entry = {"valid_pairs": 20, "router_only_passed": 1, "comparator_only_passed": 5}
    low, high = pairing.paired_difference_ci(entry)
    assert low <= (1 - 5) / 20 <= high


def test_a_tie_is_not_called_evidence_of_at_least_as_good_without_a_margin(tmp_path):
    """No non-inferiority margin was pre-registered, so no number of concordant
    pairs establishes 'at least as often'."""
    original, supp = _two_runs(tmp_path)
    _ledger(supp / "ledger.jsonl", [_row(f"c{i}", "coding", arm, True)
                                    for i in range(12) for arm in ("router", "control")])
    data = supplement.combined_report(original, supp)
    coding = data["per_category"]["coding"]["router_vs_control"]
    assert coding["valid_pairs"] == 12
    assert coding["router_at_least_as_good"] is True
    assert data["non_inferiority_margin"] is None
    assert data["quality_claim_supported"] == []
    assert "no non-inferiority margin" in supplement.format_combined(data).lower()


def test_rows_outside_the_registered_manifest_are_reported_not_analysed():
    rows = [_row("t1", "math", "router", True), _row("t1", "math", "control", True),
            _row("intruder", "math", "router", True), _row("intruder", "math", "control", True)]
    manifest = [{"id": "t1", "category": "math"}]
    acc = pairing.pair_accounting(rows, "router", "control", manifest=manifest)["math"]
    assert acc["attempted_pairs"] == 1
    assert acc["valid_pairs"] == 1
    assert acc["unregistered_task_ids"] == ["intruder"]


def test_duplicate_counting_counts_observations_and_names_the_arm():
    rows = [_row("t1", "math", "router", False), _row("t1", "math", "router", True),
            _row("t1", "math", "router", True), _row("t1", "math", "control", True)]
    acc = pairing.pair_accounting(rows, "router", "control")["math"]
    assert acc["duplicate_observations"] == 2      # three rows for one key = two collapsed
    assert acc["duplicate_keys"] == [["t1", "router"]]


def test_the_report_attributes_every_call_to_an_arm_including_discarded_ones(tmp_path):
    original, supp = _two_runs(tmp_path)
    (supp / "calls.jsonl").write_text("\n".join([
        json.dumps({"ts": 1, "model": "gpt-5.6-sol", "tag": "heldout:control-metered:s1",
                    "ok": True, "cost_usd": 0.02, "list_cost_usd": 0.0}),
        json.dumps({"ts": 2, "model": "gpt-5.6-sol", "tag": "heldout:control-metered:s9",
                    "ok": True, "cost_usd": 0.05, "list_cost_usd": 0.0})]) + "\n")
    data = supplement.combined_report(original, supp)
    arm = data["arms"]["control-metered"]
    assert arm["measured_usd"] == pytest.approx(0.02)
    assert arm["all_calls_usd"] == pytest.approx(0.07)


def test_the_report_names_the_protocol_deviation_when_an_arm_ran_separately(tmp_path):
    original, supp = _two_runs(tmp_path)
    (supp / "run-meta-a.json").write_text(json.dumps(
        {"started_at": "2026-09-18T10:23:34Z", "finished_at": "2026-09-18T10:46:10Z",
         "arms": ["router", "control"], "workers": 6, "control_model": "kimi-k3",
         "control_label": "control"}))
    (supp / "run-meta-b.json").write_text(json.dumps(
        {"started_at": "2026-09-18T10:49:45Z", "finished_at": "2026-09-18T10:51:26Z",
         "arms": ["control"], "workers": 6, "control_model": "gpt-5.6-sol",
         "control_label": "control-metered"}))
    data = supplement.combined_report(original, supp)
    assert any("separate invocation" in d for d in data["protocol_deviations"])
    assert "control-metered" in " ".join(data["protocol_deviations"])
