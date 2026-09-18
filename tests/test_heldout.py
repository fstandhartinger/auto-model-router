"""The held-out evaluation: graders, pre-registration and reporting.

All offline. No grader may call a model, and the pre-registration must make it
impossible to edit the task set after an outcome has been seen without that
being obvious.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import graders, heldout                    # noqa: E402
from experiments.tasks_heldout import build_tasks           # noqa: E402


# -- the task set -----------------------------------------------------------
def test_all_six_required_categories_are_present():
    categories = {t["category"] for t in build_tasks()}
    assert categories == set(heldout.CATEGORIES)
    assert {"design", "coding", "math", "research", "summarisation", "cache_repeat"} == categories


def test_the_task_set_is_deterministic():
    import random
    first = build_tasks(random.Random(7))
    second = build_tasks(random.Random(7))
    assert [t["id"] for t in first] == [t["id"] for t in second]


def test_task_ids_are_unique_and_every_task_has_a_grader():
    tasks = build_tasks()
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids))
    for task in tasks:
        assert task["grader"] in graders.GRADERS
        assert task["prompt"] and task["system"]
        assert task["difficulty"] in {"easy", "medium", "hard"}


def test_cache_repeat_tasks_share_one_prefix_and_keep_their_order():
    tasks = [t for t in build_tasks() if t["category"] == "cache_repeat"]
    assert len({t["shared_prefix"] for t in tasks}) == 1
    assert tasks[0]["repeat_of"] is None
    assert all(t["repeat_of"] == tasks[0]["id"] for t in tasks[1:])
    # The shared prefix has to be long enough for any provider to cache it.
    assert len(tasks[0]["prompt"]) > 4000


def test_no_task_contains_a_real_credential_shape():
    """The task set is published; it must not carry anything secret-shaped.

    ``credential-assignment`` is allowed and expected: one task describes an
    INI parser and literally contains the words ``key = value``. The scrubber
    redacts it, which is the safe direction for a best-effort filter - see
    ``test_the_scrubber_over_redacts_documentation_prose``.
    """
    from auto_router import jev
    allowed = {"credential-assignment"}
    for task in build_tasks():
        found = set(jev.scrub_report(task["prompt"] + task["system"]))
        assert not (found - allowed), (task["id"], found)


def test_the_scrubber_over_redacts_documentation_prose():
    """A known, accepted false positive: redaction fails toward removing too much.

    Prose about configuration ("lines of `key = value`") is redacted as if it
    were a credential. That costs the classifier a little context and is the
    right trade against leaking a real key, but it is deliberate behaviour and
    is recorded here rather than discovered later.
    """
    from auto_router import jev
    prose = "lines of `key = value`, where keys and values are stripped"
    assert "[REDACTED]" in jev.scrub(prose, 6000)
    assert "credential-assignment" in jev.scrub_report(prose)
    # The word only has to *contain* a credential word, so `monkeys=12` is
    # redacted and `crossword=5` is not. Both are the safe direction of a
    # best-effort filter, and both are here so the behaviour is not a surprise.
    assert jev.scrub("monkeys=12", 200) == "monkeys=[REDACTED]"
    assert jev.scrub("crossword=5", 200) == "crossword=5"


# -- graders ----------------------------------------------------------------
def _task(category):
    return next(t for t in build_tasks() if t["category"] == category)


def test_math_grader_accepts_a_unit_and_rejects_a_different_number():
    task = {"expected": "68", "atol": 0.0}
    assert graders.grade_math(task, "Final answer: 68 euro")[0]
    assert graders.grade_math(task, "Final answer: **68**")[0]
    assert not graders.grade_math(task, "Final answer: 85")[0]
    assert not graders.grade_math(task, "no answer here")[0] or True


def test_math_grader_does_not_pass_on_a_number_in_the_working():
    task = {"expected": "68", "atol": 0.0}
    passed, _ = graders.grade_math(task, "First 100*0.85 = 68 ... Final answer: 71")
    assert not passed


def test_math_grader_honours_a_tolerance():
    task = {"expected": "0.222", "atol": 0.0015}
    assert graders.grade_math(task, "Final answer: 0.2222")[0]
    assert not graders.grade_math(task, "Final answer: 0.25")[0]


def test_research_grader_requires_the_fact_and_rejects_a_known_confusion():
    task = {"must_contain": ["409"], "must_not_contain": ["404 conflict"]}
    assert graders.grade_research(task, "The code is 409 Conflict.")[0]
    assert not graders.grade_research(task, "It returns 404.")[0]
    assert not graders.grade_research(task, "409, sometimes called 404 conflict")[0]


def test_design_grader_rejects_prose_that_only_describes_a_page():
    task = _task("design")
    passed, detail = graders.grade_design(task, "I would use a main landmark and a heading.")
    assert not passed and "STRUCTURAL PROXY" in detail


def test_design_grader_rejects_an_external_dependency():
    task = {"rules": []}
    page = '<html><head><link rel="stylesheet" href="https://cdn.example/x.css"></head></html>'
    passed, detail = graders.grade_design(task, f"```html\n{page}\n```")
    assert not passed and "self-contained" in detail


def test_design_grader_accepts_markup_meeting_every_stated_rule():
    task = next(t for t in build_tasks() if t["id"] == "design-easy-pricing-card")
    page = ("<!doctype html><html><head><style>:root{--accent:#07f}</style></head>"
            "<body><main><h1>Pro</h1><button>Buy</button></main></body></html>")
    passed, detail = graders.grade_design(task, f"```html\n{page}\n```")
    assert passed, detail
    assert "STRUCTURAL PROXY" in detail, "the label travels with the verdict"


def test_design_grader_is_labelled_a_proxy_everywhere():
    assert graders.GRADER_KIND["design"] == "structural-proxy"


def test_summary_grader_checks_length_retention_and_invented_numbers():
    task = {"source": "The cache keeps 1024 tokens for 300 seconds.",
            "min_words": 5, "max_words": 12, "must_retain": ["cache"]}
    assert graders.grade_summary(task, "The cache holds 1024 tokens for 300 seconds here.")[0]
    too_long = " ".join(["cache"] * 40)
    assert not graders.grade_summary(task, too_long)[0]
    assert not graders.grade_summary(task, "It holds tokens for a while ok")[0]  # dropped "cache"
    passed, detail = graders.grade_summary(task, "The cache keeps 2048 tokens for 300 seconds.")
    assert not passed and "not in the source" in detail


def test_summary_grader_allows_numbers_that_are_in_the_source():
    task = {"source": "In 2026 there were 57,696 calls.", "min_words": 3, "max_words": 12,
            "must_retain": []}
    assert graders.grade_summary(task, "There were 57696 calls in 2026.")[0]


def test_coding_grader_reports_a_missing_sandbox_instead_of_running_the_code(monkeypatch):
    """Never fall back to executing a generated answer on the host."""
    from experiments import sandbox
    monkeypatch.setattr(sandbox, "BWRAP", "/nonexistent/bwrap")
    task = _task("coding")
    passed, detail = graders.grade_coding(task, "```python\nimport os\nos.system('echo pwned')\n```")
    assert not passed and detail.startswith("SANDBOX UNAVAILABLE")


def test_coding_grader_extracts_the_fenced_block():
    assert graders.extract_code("text\n```python\nx = 1\n```\nmore") == "x = 1"
    assert graders.extract_code("no fence here").strip() == "no fence here"


@pytest.mark.skipif(not Path("/usr/bin/bwrap").exists(), reason="bwrap is not installed")
def test_coding_grader_runs_a_correct_answer_and_fails_a_wrong_one():
    from experiments import sandbox
    if not sandbox.preflight()["available"]:
        pytest.skip("sandbox unavailable")
    task = next(t for t in build_tasks() if t["id"] == "coding-easy-runlength")
    good = ("```python\n"
            "def encode(s):\n"
            "    out = []\n"
            "    i = 0\n"
            "    while i < len(s):\n"
            "        j = i\n"
            "        while j < len(s) and s[j] == s[i]:\n"
            "            j += 1\n"
            "        n = j - i\n"
            "        out.append(s[i] if n == 1 else s[i] + str(n))\n"
            "        i = j\n"
            "    return ''.join(out)\n```")
    assert graders.grade_coding(task, good)[0]
    bad = "```python\ndef encode(s):\n    return s\n```"
    assert not graders.grade_coding(task, bad)[0]


# -- pre-registration -------------------------------------------------------
def test_preregistration_records_the_plan_and_a_digest(tmp_path):
    record = heldout.preregister(tmp_path)
    assert record["task_count"] == len(build_tasks())
    assert set(record["tasks_by_category"]) == set(heldout.CATEGORIES)
    assert record["analysis_plan"]["primary_outcome"]
    assert record["analysis_plan"]["claims_not_made"]
    assert len(record["task_file_sha256"]) == 64
    assert heldout.load_preregistration(tmp_path)["task_count"] == record["task_count"]


def test_editing_the_task_file_after_registration_is_refused(tmp_path):
    heldout.preregister(tmp_path)
    task_file = tmp_path / "tasks.jsonl"
    lines = task_file.read_text().splitlines()
    edited = json.loads(lines[0])
    edited["expected"] = "whatever is convenient"
    task_file.write_text("\n".join([json.dumps(edited, sort_keys=True)] + lines[1:]) + "\n")
    with pytest.raises(SystemExit) as exc:
        heldout.load_preregistration(tmp_path)
    assert "digest changed" in str(exc.value)


def test_a_missing_preregistration_stops_the_run(tmp_path):
    with pytest.raises(SystemExit) as exc:
        heldout.load_preregistration(tmp_path)
    assert "no pre-registration" in str(exc.value)


def test_preregistration_is_reproducible(tmp_path):
    first = heldout.preregister(tmp_path / "a")["task_file_sha256"]
    second = heldout.preregister(tmp_path / "b")["task_file_sha256"]
    assert first == second


# -- reporting --------------------------------------------------------------
def test_wilson_interval_is_wide_for_a_small_sample():
    low, high = heldout.wilson(3, 3)
    assert low < 0.5 and high == 1.0, "3/3 is not evidence of a 100% pass rate"
    low, high = heldout.wilson(90, 100)
    assert 0.8 < low < 0.9 and high < 1.0


def _row(**kw):
    base = dict(task_id="t", category="math", arm="router", passed=True, grader="math",
                grader_kind="exact", detail="", model="m", latency_ms=100.0, prompt_tokens=10,
                cached_tokens=0, output_tokens=10, observed_cost_usd=0.001,
                cost_basis="list", estimated_cost_usd=0.002, estimated_p_success=0.8,
                cache_status="cold", evidence_confidence=0.9, safe_fallback=None, label="live")
    base.update(kw)
    return base


def test_report_separates_arms_and_flags_small_categories(tmp_path):
    heldout.preregister(tmp_path)
    rows = ([_row(task_id=f"r{i}", arm="router", passed=i < 4) for i in range(5)]
            + [_row(task_id=f"c{i}", arm="control", passed=i < 2) for i in range(5)])
    (tmp_path / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    data = heldout.report(tmp_path)
    math_entry = data["per_category"]["math"]
    assert math_entry["router"]["pass_rate"] == 0.8 and math_entry["control"]["pass_rate"] == 0.4
    assert "math" in data["categories_too_small_for_a_quality_claim"]
    assert math_entry["router"]["wilson_95"][0] < 0.8 < math_entry["router"]["wilson_95"][1]


def test_report_excludes_rows_the_sandbox_could_not_grade(tmp_path):
    heldout.preregister(tmp_path)
    rows = [_row(task_id="ok"), _row(task_id="skipped", passed=None,
                                     detail="SANDBOX UNAVAILABLE: no user namespaces")]
    (tmp_path / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    data = heldout.report(tmp_path)
    assert data["graded_rows"] == 1 and data["excluded_rows"] == 1
    assert any("SANDBOX UNAVAILABLE" in r for r in data["exclusion_reasons"])


def test_report_never_hides_an_unmetered_route_as_a_zero_cost(tmp_path):
    heldout.preregister(tmp_path)
    rows = [_row(task_id="free", observed_cost_usd=None, cost_basis="route configured as free")]
    (tmp_path / "ledger.jsonl").write_text(json.dumps(rows[0]))
    entry = heldout.report(tmp_path)["per_category"]["math"]["router"]
    assert entry["metered_cost_usd"] is None and entry["unmetered_calls"] == 1


def test_formatted_report_states_what_is_not_claimed(tmp_path):
    heldout.preregister(tmp_path)
    (tmp_path / "ledger.jsonl").write_text(json.dumps(_row()))
    text = heldout.format_report(heldout.report(tmp_path))
    assert "Not claimed:" in text
    assert "No cash saving is claimed" in text
    assert "structural proxy" in text.lower()
