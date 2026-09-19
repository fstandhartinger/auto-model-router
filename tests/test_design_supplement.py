"""The 2026-09-19 design-floor supplement: tasks, registration rules, report.

Pins the properties the supplement's registration depends on:

1. the builder is deterministic and yields exactly the registered two tasks;
2. every rule the grader checks is demanded verbatim by its prompt, including
   the two checks ``grade_design`` applies to every design answer - the defect
   that let two historical tasks grade an undemanded CSS property and section;
3. each rule is necessary: a reference answer passes, and removing exactly one
   demanded construct fails exactly that rule;
4. the tasks are new: no id or prompt of any earlier task set is reused;
5. the cap guard, the no-repeat rule, the read-back and the combined floor
   arithmetic behave as registered.
"""

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import design_supplement as ds, graders, heldout_run   # noqa: E402
from experiments import tasks_heldout, tasks_supplement                 # noqa: E402
from experiments.tasks_design_supplement import (                       # noqa: E402
    IMPLICIT_CHECK_DEMANDS, build_tasks)

RUNS = Path(__file__).resolve().parent.parent.parent / "runs"

SHORTCUT = """```html
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Shortcuts</title>
<style>
body { font: 16px/1.4 system-ui, sans-serif; }
dl { display: grid; grid-template-columns: auto 1fr; gap: .5rem 1rem; }
@media (max-width: 30em) { dl { grid-template-columns: 1fr; } }
</style></head>
<body><main class="card"><h1>Keyboard shortcuts</h1>
<dl>
 <dt><kbd>Ctrl</kbd>+<kbd>S</kbd></dt><dd>Save</dd>
 <dt><kbd>Ctrl</kbd>+<kbd>Z</kbd></dt><dd>Undo</dd>
</dl></main></body></html>
```"""

RATING = """```html
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Rating</title>
<style>
.stars input:focus-visible + label { outline: 2px solid #1a73e8; }
</style></head>
<body><fieldset class="stars"><legend>Rate this article</legend>
 <input type="radio" id="s1" name="rating" value="1"><label for="s1">1 star</label>
 <input type="radio" id="s2" name="rating" value="2"><label for="s2">2 stars</label>
</fieldset></body></html>
```"""

REFERENCE = {"ds2-design-easy-shortcut-card": SHORTCUT, "ds2-design-easy-star-rating": RATING}

#: For every rule, the edit that removes every instance of the demanded construct and nothing else.
REMOVE = {
    "ds2-design-easy-shortcut-card": {
        "starts with a doctype": ("<!DOCTYPE html>", ""),
        "has an inline <style> block": ("<style>", "<template>"),
        "has a <main> landmark": ("<main class", "<div class"),
        "has an <h1>": ("<h1>", "<p>"),
        "uses a description list": ("<dl>\n", "<div>\n"),
        "has terms": ("<dt>", "<b>"),
        "has descriptions": ("<dd>", "<i>"),
        "marks keys with <kbd>": ("<kbd>", "<code>"),
        "has a media query": ("@media (max-width: 30em)", ".narrow"),
    },
    "ds2-design-easy-star-rating": {
        "starts with a doctype": ("<!DOCTYPE html>", ""),
        "has an inline <style> block": ("<style>", "<template>"),
        "groups the stars in a <fieldset>": ("<fieldset", "<div"),
        "the group has a <legend>": ("<legend>", "<p>"),
        "is built from radio inputs": ('type="radio"', 'type="checkbox"'),
        "labels are associated with for=": ("label for=", "label data-x="),
        "styles :focus-visible": (":focus-visible", ":hover"),
    },
}


def _failed(task, answer):
    ok, detail = graders.grade_design(task, answer)
    if ok:
        return []
    return json.loads(detail.split("failed: ", 1)[1].replace("'", '"'))


def test_builder_is_deterministic_and_matches_the_run_order():
    first, second = build_tasks(), build_tasks()
    assert first == second
    assert [t["id"] for t in first] == ["ds2-design-easy-shortcut-card",
                                        "ds2-design-easy-star-rating"]
    assert ds.RUN_ORDER == [(t["id"], arm) for t in first
                            for arm in (ds.ROUTER_ARM, ds.CONTROL_ARM)]
    assert all(t["category"] == "design" and t["grader"] == "design" for t in first)


@pytest.mark.parametrize("task", build_tasks(), ids=lambda t: t["id"])
def test_every_graded_rule_is_demanded_verbatim_by_the_prompt(task):
    for rule in task["rules"]:
        assert rule["demand"] in task["prompt"], rule["label"]
    for label, demand in IMPLICIT_CHECK_DEMANDS.items():
        assert demand in task["prompt"], label


def test_the_implicit_checks_are_exactly_the_ones_grade_design_applies():
    task = {"rules": []}
    ok, detail = graders.grade_design(task, "plain prose, no markup "
                                            '<link href="https://x/y.css">')
    assert not ok
    for label in IMPLICIT_CHECK_DEMANDS:
        assert label in detail
    assert graders.grade_design(task, "<!doctype html><p>x</p>")[0]


@pytest.mark.parametrize("task", build_tasks(), ids=lambda t: t["id"])
def test_reference_answer_passes_and_each_rule_is_necessary(task):
    reference = REFERENCE[task["id"]]
    assert graders.grade_design(task, reference) == (True, "STRUCTURAL PROXY - all rules met")
    removals = REMOVE[task["id"]]
    assert set(removals) == {r["label"] for r in task["rules"]}
    for label, (old, new) in removals.items():
        assert old in reference, label
        assert _failed(task, reference.replace(old, new)) == [label]


def test_an_external_resource_fails_the_self_contained_check():
    task = build_tasks()[0]
    answer = SHORTCUT.replace("</title>", '</title><link rel="stylesheet" href="https://cdn.x/a.css">')
    assert _failed(task, answer) == [
        "self-contained (no external stylesheet, script, font, image or frame)"]


def _prior_tasks():
    tasks = tasks_heldout.build_tasks(random.Random(0)) + tasks_supplement.build_tasks(random.Random(0))
    for path in RUNS.glob("*/tasks.jsonl"):
        if "design-supplement" in path.parent.name:
            continue
        tasks += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return tasks


def test_the_tasks_are_new():
    prior = _prior_tasks()
    ids = {t["id"] for t in prior}
    prompts = {t["prompt"] for t in prior}
    for task in build_tasks():
        assert task["id"] not in ids
        assert task["prompt"] not in prompts
        # and no earlier design prompt is contained in, or contains, the new one
        assert not any(p in task["prompt"] or task["prompt"] in p for p in prompts)


def test_registered_parameters():
    assert ds.CAP_USD <= 0.75
    assert ds.DESIGN_OUTPUT_BUDGET == heldout_run.OUTPUT_BUDGET["design"] == 12000
    assert ds.CONTROL_ARM == "control-metered" and ds.CONTROL_MODEL == "gpt-5.6-sol"
    assert "not a replacement" in ds.ANALYSIS_PLAN["role"]
    assert "immutable" in ds.ANALYSIS_PLAN["role"]
    assert any("equivalence" in c for c in ds.ANALYSIS_PLAN["claims_not_made"])


class _Price:
    def __init__(self, i, o):
        self.input, self.output = i, o


class _Model:
    def __init__(self, i, o):
        self.prices = _Price(i, o)


class _Catalog:
    def all(self):
        return [_Model(0, 0), _Model(2.0, 10.0), _Model(0.2, 0.5)]


class _Config:
    catalog = _Catalog()


def test_worst_case_is_the_priciest_route_at_full_budget_times_the_multiplier():
    task = build_tasks()[0]
    chars = len(task["system"]) + len(task["prompt"])
    expected = (chars * 2.0 + 12000 * 10.0) / 1e6 * ds.BILLING_MULTIPLIER
    assert ds.worst_case_usd(_Config(), task) == pytest.approx(expected)
    # four such calls would exceed the cap, so the guard is per call, on recorded spend
    assert expected * 2 < ds.CAP_USD < expected * 4


def test_spend_is_cap_basis_and_an_unreadable_line_is_not_a_zero(tmp_path):
    calls = tmp_path / "calls.jsonl"
    calls.write_text(json.dumps({"cost_usd": 0.2, "list_cost_usd": 0.1}) + "\n"
                     + json.dumps({"cost_usd": 0.0, "list_cost_usd": 0.05}) + "\n")
    assert ds.spent_usd(calls) == pytest.approx(0.25)
    calls.write_text(calls.read_text() + '{"cost_usd": 0.1\n')
    with pytest.raises(json.JSONDecodeError):
        ds.spent_usd(calls)


def test_call_tags_match_what_heldout_run_writes():
    assert ds.call_tag("t", "router") == "heldout:router:t"
    assert ds.call_tag("t", "control-metered") == "heldout:control:t"


# ---------------------------------------------------------------------------
# read-back and the combined report, on synthetic ledgers
# ---------------------------------------------------------------------------
def _row(task_id, arm, passed, detail="STRUCTURAL PROXY - all rules met", tokens=900):
    return {"task_id": task_id, "category": "design", "arm": arm, "passed": passed,
            "detail": detail, "model": "m", "output_tokens": tokens, "observed_cost_usd": None}


TRUNC = "TRUNCATED: hit the 12000-token output budget; excluded, not counted as a failure"


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _historical(tmp_path, valid=9, truncated=5):
    hist = tmp_path / "hist"
    hist.mkdir()
    tasks, rows = [], []
    for i in range(valid + truncated):
        tid = f"design-h{i}"
        tasks.append({"id": tid, "category": "design"})
        rows.append(_row(tid, "router", None if i >= valid else True,
                         TRUNC if i >= valid else "STRUCTURAL PROXY - all rules met"))
        rows.append(_row(tid, "control-metered", True))
    tasks.append({"id": "math-x", "category": "math"})
    rows += [dict(_row("math-x", "router", True), category="math"),
             dict(_row("math-x", "control-metered", True), category="math")]
    _write(hist / "tasks.jsonl", tasks)
    _write(hist / "ledger.jsonl", rows)
    return hist


def _supplement(tmp_path, router_rows):
    supp = tmp_path / "supp"
    supp.mkdir()
    _write(supp / "tasks.jsonl", build_tasks())
    rows = []
    for (tid, _), (passed, detail) in zip(ds.RUN_ORDER[::2], router_rows):
        rows += [_row(tid, "router", passed, detail), _row(tid, "control-metered", True)]
    _write(supp / "ledger.jsonl", rows)
    return supp


def test_both_new_router_answers_truncated_leaves_the_floor_unmet(tmp_path):
    data = ds.combined_design_report(
        _supplement(tmp_path, [(None, TRUNC), (None, TRUNC)]), _historical(tmp_path))
    assert data["pooled"]["valid_pairs"] == 9
    assert data["pooled"]["invalid_truncated"] == 7
    assert not data["floor_met"]
    assert data["answer"].startswith("floor NOT met")


def test_one_valid_new_pair_meets_the_floor_without_claiming_a_difference(tmp_path):
    data = ds.combined_design_report(
        _supplement(tmp_path, [(False, "STRUCTURAL PROXY - failed: ['x']"), (None, TRUNC)]),
        _historical(tmp_path))
    assert data["pooled"]["valid_pairs"] == 10
    assert data["supplement_only"]["valid_pairs"] == 1
    assert data["historical_only"]["valid_pairs"] == 9
    assert data["floor_met"]
    assert "no difference demonstrated" in data["answer"]
    assert "not equivalence" in data["answer"]
    assert data["pooled"]["discordant_router_only_vs_control_only"] == [0, 1]
    text = ds.format_markdown(data)
    assert "| **pooled** | 16 | 10 |" in text


def test_a_failed_call_keeps_its_pair_valid(tmp_path):
    data = ds.combined_design_report(
        _supplement(tmp_path, [(False, "the route failed the call: HTTP 503"),
                               (False, "the route failed the call: HTTP 503")]),
        _historical(tmp_path))
    assert data["pooled"]["valid_pairs"] == 11


def test_a_row_never_attempted_is_incomplete_not_dropped(tmp_path):
    supp = _supplement(tmp_path, [(True, "STRUCTURAL PROXY - all rules met")])
    data = ds.combined_design_report(supp, _historical(tmp_path))
    assert data["supplement_only"]["attempted_pairs"] == 2
    assert data["supplement_only"]["invalid_incomplete"] == 1


def test_overlapping_task_ids_are_refused(tmp_path):
    hist = _historical(tmp_path)
    _write(hist / "tasks.jsonl", [{"id": "ds2-design-easy-star-rating", "category": "design"}])
    with pytest.raises(SystemExit):
        ds.combined_design_report(_supplement(tmp_path, []), hist)


def test_readback_finds_duplicates_and_calls_without_rows(tmp_path):
    supp = _supplement(tmp_path, [(True, "ok"), (True, "ok")])
    with (supp / "ledger.jsonl").open("a") as fh:
        fh.write(json.dumps(_row("ds2-design-easy-star-rating", "router", True)) + "\n")
    calls = [{"tag": ds.call_tag(t, a), "ok": True, "cost_usd": 0.01, "list_cost_usd": 0.005}
             for t, a in ds.RUN_ORDER]
    calls.append({"tag": "heldout:router:ghost", "ok": False, "cost_usd": 0, "list_cost_usd": 0})
    _write(supp / "calls.jsonl", calls)
    out = ds.readback(supp, tmp_path / "nowhere")
    assert out["duplicate_rows"] == [("ds2-design-easy-star-rating", "router")]
    assert out["calls_without_row"] == ["heldout:router:ghost"]
    assert out["failed_calls"] == ["heldout:router:ghost"]
    assert out["cap_basis_usd"] == pytest.approx(0.04)
    assert out["missing_rows"] == [] and out["duplicate_call_tags"] == []
    assert out["historical_files_unchanged"] is False


@pytest.mark.skipif(not (RUNS / ds.HISTORICAL_DIR_NAME).exists(),
                    reason="the historical run lives only on the operator's machine")
def test_the_historical_run_is_byte_identical_to_its_pinned_digests():
    assert ds.historical_digests(RUNS / ds.HISTORICAL_DIR_NAME) == ds.HISTORICAL_FILES


def test_the_builder_reproduces_the_registered_task_file_byte_for_byte():
    """The 2026-09-19T10:46Z registration's task-file digest, taken from its preregistration.json."""
    import hashlib
    text = "".join(json.dumps(t, sort_keys=True) + "\n" for t in build_tasks())
    assert hashlib.sha256(text.encode()).hexdigest() == (
        "a60052083562494bbe6c4e9e410858302508bc28737bf702d1cee97802efe9d2")
