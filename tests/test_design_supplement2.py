"""The second design-floor supplement: credential preflight, tasks, registration rules, report.

Pins the properties its registration depends on:

1. the credential preflight reports booleans only, fails closed when the
   control's key variable (or any catalog provider's) is missing, and ``run``
   refuses before any client or HTTP request exists - in-process and as a
   separate process with a stripped environment;
2. the builder is deterministic and yields exactly the registered two tasks,
   new relative to every earlier task set;
3. every rule the grader checks is demanded verbatim by the prompt, and each
   rule is necessary (a reference passes, removing one construct fails exactly
   that rule);
4. a credential rejection with no tokens is a harness failure, never a
   quality outcome, in the runner and in the combined report;
5. read-back and combined floor arithmetic over the three sources behave as
   registered.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import design_supplement2 as ds, graders, heldout_run, llm   # noqa: E402
from experiments import tasks_heldout, tasks_supplement                        # noqa: E402
from experiments.tasks_design_supplement import build_tasks as build_ds1       # noqa: E402
from experiments.tasks_design_supplement2 import (                             # noqa: E402
    IMPLICIT_CHECK_DEMANDS, build_tasks)

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO.parent / "runs"
LIVE_CONFIG = REPO.parent / "live-config.yaml"
SENTINEL = "sk-SENTINEL-must-never-appear-0123456789"

HOURS = """```html
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Hours</title>
<style>
table { border-collapse: collapse; }
@media (prefers-color-scheme: dark) { table { color: #eee; background: #111; } }
</style></head>
<body><table><caption>Opening hours</caption>
 <tr><th scope="row">Monday</th><td><time>09:00</time>–<time>18:00</time></td></tr>
 <tr><th scope="row">Sunday</th><td>Closed</td></tr>
</table></body></html>
```"""

RECIPE = """```html
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Recipe</title>
<style>
:root { --accent: #c0392b; }
h2 { color: var(--accent); }
</style></head>
<body><article class="card"><h2>Tomato soup</h2>
 <ul class="ingredients"><li>Tomatoes</li><li>Onion</li></ul>
 <ol class="steps"><li>Chop.</li><li>Simmer.</li></ol>
</article></body></html>
```"""

REFERENCE = {"ds3-design-easy-opening-hours": HOURS, "ds3-design-easy-recipe-card": RECIPE}

REMOVE = {
    "ds3-design-easy-opening-hours": {
        "starts with a doctype": ("<!DOCTYPE html>", ""),
        "has an inline <style> block": ("<style>", "<template>"),
        "uses a <table>": ("<table>", "<div>"),
        "the table has a <caption>": ("<caption>", "<p>"),
        "row headers use scope=row": ('scope="row"', 'class="day"'),
        "times are marked with <time>": ("<time>", "<span>"),
        "has a dark-mode media query": ("prefers-color-scheme: dark", "min-width: 1px"),
    },
    "ds3-design-easy-recipe-card": {
        "starts with a doctype": ("<!DOCTYPE html>", ""),
        "has an inline <style> block": ("<style>", "<template>"),
        "wrapped in an <article>": ("<article class", "<div class"),
        "has an <h2> title": ("<h2>", "<p>"),
        "ingredients in a <ul>": ("<ul class", "<div class"),
        "steps in an <ol>": ("<ol class", "<div class"),
        "uses a CSS custom property via var()": ("var(--accent)", "#c0392b"),
    },
}


def _failed(task, answer):
    ok, detail = graders.grade_design(task, answer)
    if ok:
        return []
    return json.loads(detail.split("failed: ", 1)[1].replace("'", '"'))


# ---------------------------------------------------------------------------
# credential preflight
# ---------------------------------------------------------------------------
class _Price:
    def __init__(self, i, o):
        self.input, self.output = i, o
        self.is_free = not (i or o)


class _Model:
    def __init__(self, name, provider, i=0.0, o=0.0):
        self.name, self.provider, self.prices = name, provider, _Price(i, o)


class _Provider:
    def __init__(self, var):
        self.api_key_env = var


class _Catalog:
    def all(self):
        return [_Model("free-a", "free-host"), _Model("gpt-5.6-sol", "openrouter", 2.0, 10.0),
                _Model("cheap", "tensorx", 0.2, 0.5)]


class _Config:
    catalog = _Catalog()
    providers = {"free-host": _Provider("FREE_HOST_API_KEY"),
                 "openrouter": _Provider("OPEN_ROUTER_API_KEY"),
                 "tensorx": _Provider("TENSORX_API_KEY"),
                 "xai": _Provider("XAI_API_KEY")}      # not in the catalog: not required


FULL_ENV = {"FREE_HOST_API_KEY": SENTINEL, "OPEN_ROUTER_API_KEY": SENTINEL,
            "TENSORX_API_KEY": SENTINEL}


def test_preflight_passes_with_every_catalog_key_and_records_booleans_only():
    result = ds.credential_preflight(_Config(), FULL_ENV)
    assert result["ok"] and result["control_credential_present"]
    assert result["control_provider"] == "openrouter"
    assert set(result["providers"]) == {"free-host", "openrouter", "tensorx"}
    assert result["providers"]["openrouter"] == {"variable": "OPEN_ROUTER_API_KEY",
                                                 "present": True}
    assert SENTINEL not in json.dumps(result)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_preflight_fails_closed_without_the_control_key(value):
    env = dict(FULL_ENV)
    if value is None:
        env.pop("OPEN_ROUTER_API_KEY")
    else:
        env["OPEN_ROUTER_API_KEY"] = value
    result = ds.credential_preflight(_Config(), env)
    assert not result["ok"] and not result["control_credential_present"]
    with pytest.raises(SystemExit, match="OPEN_ROUTER_API_KEY"):
        ds.require_credentials(_Config(), None, env)


def test_preflight_fails_closed_without_a_router_provider_key():
    env = dict(FULL_ENV)
    env.pop("FREE_HOST_API_KEY")
    result = ds.credential_preflight(_Config(), env)
    assert result["control_credential_present"] and not result["ok"]


def test_run_refuses_before_any_client_or_http_when_the_control_key_is_missing(
        tmp_path, monkeypatch):
    """The runner's own check, not a test-process check: nothing may be sent."""
    import auto_router.config
    import httpx

    def boom(*a, **kw):
        raise AssertionError("an HTTP request or model call was attempted")

    monkeypatch.setattr(auto_router.config, "load_config", lambda path: _Config())
    monkeypatch.setattr(llm.Client, "__init__", boom)
    monkeypatch.setattr(llm.Client, "chat", boom)
    monkeypatch.setattr(httpx.Client, "send", boom)
    monkeypatch.setattr(ds, "verify", boom)
    env = {k: v for k, v in FULL_ENV.items() if k != "OPEN_ROUTER_API_KEY"}
    with pytest.raises(SystemExit, match="credential preflight FAILED"):
        ds.run(tmp_path, tmp_path / "cfg.yaml", tmp_path, environ=env)
    assert not (tmp_path / "calls.jsonl").exists()
    assert not (tmp_path / "ledger.jsonl").exists()
    refusals = list(tmp_path.glob("preflight-refused-*.json"))
    assert len(refusals) == 1
    record = json.loads(refusals[0].read_text())
    assert record["ok"] is False and record["control_credential_present"] is False
    assert SENTINEL not in refusals[0].read_text()


def test_run_reads_the_process_environment_by_default(tmp_path, monkeypatch):
    import auto_router.config
    monkeypatch.setattr(auto_router.config, "load_config", lambda path: _Config())
    monkeypatch.setattr(ds, "verify", lambda *a: (_ for _ in ()).throw(AssertionError("reached")))
    for k, v in FULL_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("OPEN_ROUTER_API_KEY")
    with pytest.raises(SystemExit, match="OPEN_ROUTER_API_KEY"):
        ds.run(tmp_path, tmp_path / "cfg.yaml", tmp_path)


@pytest.mark.skipif(not LIVE_CONFIG.exists(), reason="the live config is operator-only")
@pytest.mark.parametrize("with_key", [False, True])
def test_preflight_command_as_a_separate_process(with_key):
    env = {k: v for k, v in os.environ.items() if k != "OPEN_ROUTER_API_KEY"}
    if with_key:
        env["OPEN_ROUTER_API_KEY"] = SENTINEL
        from auto_router.config import load_config
        config = load_config(LIVE_CONFIG)
        for model in config.catalog.all():
            env.setdefault(config.providers[model.provider].api_key_env, SENTINEL)
    out = subprocess.run([sys.executable, "-m", "experiments.design_supplement2", "preflight",
                          "--runs", str(RUNS), "--config", str(LIVE_CONFIG)],
                         cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert SENTINEL not in out.stdout + out.stderr
    assert out.returncode == (0 if with_key else 1)
    assert ("credential preflight PASS" if with_key else "credential preflight FAIL") in out.stdout


def test_a_credential_rejection_with_no_tokens_is_a_harness_failure():
    assert ds.is_harness_failure(llm.CallResult(ok=False, error="HTTP 401: {}"))
    assert ds.is_harness_failure({"ok": False, "error": "HTTP 403: x", "prompt": 0, "output": 0})
    assert not ds.is_harness_failure(llm.CallResult(ok=False, error="HTTP 503: x"))
    assert not ds.is_harness_failure({"ok": False, "error": "HTTP 401", "prompt": 12, "output": 0})
    assert not ds.is_harness_failure(llm.CallResult(ok=True, output_tokens=5))


# ---------------------------------------------------------------------------
# tasks and prompt <-> grader correspondence
# ---------------------------------------------------------------------------
def test_builder_is_deterministic_and_matches_the_run_order():
    first, second = build_tasks(), build_tasks()
    assert first == second
    assert [t["id"] for t in first] == ["ds3-design-easy-opening-hours",
                                        "ds3-design-easy-recipe-card"]
    assert ds.RUN_ORDER == [(t["id"], arm) for t in first
                            for arm in (ds.ROUTER_ARM, ds.CONTROL_ARM)]
    assert all(t["category"] == "design" and t["grader"] == "design" for t in first)


@pytest.mark.parametrize("task", build_tasks(), ids=lambda t: t["id"])
def test_every_graded_rule_is_demanded_verbatim_by_the_prompt(task):
    for rule in task["rules"]:
        assert rule["demand"] in task["prompt"], rule["label"]
    for label, demand in IMPLICIT_CHECK_DEMANDS.items():
        assert demand in task["prompt"], label
    # and the grader checks nothing beyond the declared rules and the two implicit checks
    ok, detail = graders.grade_design(task, "no markup at all")
    assert set(json.loads(detail.split("failed: ", 1)[1].replace("'", '"'))) <= (
        {r["label"] for r in task["rules"]} | set(IMPLICIT_CHECK_DEMANDS))


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
    task = build_tasks()[1]
    answer = RECIPE.replace("</title>", '</title><link rel="stylesheet" href="https://cdn.x/a.css">')
    assert _failed(task, answer) == [
        "self-contained (no external stylesheet, script, font, image or frame)"]


def _prior_tasks():
    tasks = (tasks_heldout.build_tasks(random.Random(0))
             + tasks_supplement.build_tasks(random.Random(0)) + build_ds1())
    for path in RUNS.glob("*/tasks.jsonl"):
        tasks += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [t for t in tasks if not t["id"].startswith("ds3-")]


def test_the_tasks_are_new():
    prior = _prior_tasks()
    ids = {t["id"] for t in prior}
    prompts = {t["prompt"] for t in prior}
    for task in build_tasks():
        assert task["id"] not in ids
        assert not any(p in task["prompt"] or task["prompt"] in p for p in prompts)


def test_registered_parameters():
    assert ds.CAP_USD <= 0.75
    assert ds.DESIGN_OUTPUT_BUDGET == heldout_run.OUTPUT_BUDGET["design"] == 12000
    assert ds.CONTROL_ARM == "control-metered" and ds.CONTROL_MODEL == "gpt-5.6-sol"
    assert "not a replacement" in ds.ANALYSIS_PLAN["role"]
    assert "immutable" in ds.ANALYSIS_PLAN["role"]
    assert "no retries" in ds.ANALYSIS_PLAN["execution"]
    assert "HARNESS FAILURE" in ds.ANALYSIS_PLAN["exclusion_rule"]
    assert "OPEN_ROUTER_API_KEY" in ds.ANALYSIS_PLAN["credential_preflight"]
    assert any("equivalence" in c for c in ds.ANALYSIS_PLAN["claims_not_made"])
    # the per-call worst case at the priciest route keeps at least two calls under the cap
    worst = max(ds.ds1.worst_case_usd(_Config(), t) for t in build_tasks())
    assert 2 * worst < ds.CAP_USD


# ---------------------------------------------------------------------------
# read-back and the combined report, on synthetic ledgers
# ---------------------------------------------------------------------------
def _row(task_id, arm, passed, detail="STRUCTURAL PROXY - all rules met", tokens=900):
    return {"task_id": task_id, "category": "design", "arm": arm, "passed": passed,
            "detail": detail, "model": "m", "output_tokens": tokens, "observed_cost_usd": None}


TRUNC = "TRUNCATED: hit the 12000-token output budget; excluded, not counted as a failure"


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _runs(tmp_path, valid=9, truncated=5):
    runs = tmp_path / "runs"
    hist = runs / ds.HISTORICAL_DIR_NAME
    hist.mkdir(parents=True)
    tasks, rows = [], []
    for i in range(valid + truncated):
        tid = f"design-h{i}"
        tasks.append({"id": tid, "category": "design"})
        rows.append(_row(tid, "router", None if i >= valid else True,
                         TRUNC if i >= valid else "STRUCTURAL PROXY - all rules met"))
        rows.append(_row(tid, "control-metered", True))
    _write(hist / "tasks.jsonl", tasks)
    _write(hist / "ledger.jsonl", rows)
    s1 = runs / ds.FIRST_SUPPLEMENT_DIR_NAME
    s1.mkdir()
    _write(s1 / "tasks.jsonl", build_ds1())
    rows, calls = [], []
    for task in build_ds1():
        rows += [_row(task["id"], "router", True),
                 _row(task["id"], "control-metered", False, "the route failed the call: HTTP 401")]
        calls += [{"tag": f"heldout:router:{task['id']}", "ok": True, "prompt": 330, "output": 900},
                  {"tag": f"heldout:control:{task['id']}", "ok": False, "prompt": 0, "output": 0,
                   "error": "HTTP 401: {}"}]
    _write(s1 / "ledger.jsonl", rows)
    _write(s1 / "calls.jsonl", calls)
    return runs


def _supplement(tmp_path, outcomes, calls=None):
    """``outcomes``: per task, ((router passed, detail), (control passed, detail))."""
    supp = tmp_path / "supp"
    supp.mkdir()
    _write(supp / "tasks.jsonl", build_tasks())
    rows = []
    for (tid, _), (r, c) in zip(ds.RUN_ORDER[::2], outcomes):
        rows += [_row(tid, "router", *r), _row(tid, "control-metered", *c)]
    _write(supp / "ledger.jsonl", rows)
    _write(supp / "calls.jsonl", calls or [])
    return supp


PASS = (True, "STRUCTURAL PROXY - all rules met")
FAIL = (False, "STRUCTURAL PROXY - failed: ['x']")


def test_the_first_supplement_contributes_no_valid_pair(tmp_path):
    runs = _runs(tmp_path)
    data = ds.combined_design_report(_supplement(tmp_path, []), runs)
    assert data["first_supplement_only"]["valid_pairs"] == 0
    assert data["first_supplement_only"]["attempted_pairs"] == 2
    assert data["historical_only"]["valid_pairs"] == 9
    assert data["pooled"]["valid_pairs"] == 9
    assert not data["floor_met"] and data["answer"].startswith("floor NOT met")


def test_one_valid_new_pair_meets_the_floor_without_claiming_a_difference(tmp_path):
    runs = _runs(tmp_path)
    data = ds.combined_design_report(_supplement(tmp_path, [(FAIL, PASS), ((None, TRUNC), PASS)]),
                                     runs)
    assert data["pooled"]["valid_pairs"] == 10
    assert data["second_supplement_only"]["valid_pairs"] == 1
    assert data["floor_met"]
    assert "no difference demonstrated" in data["answer"] and "not equivalence" in data["answer"]
    assert "| **pooled** | 18 | 10 |" in ds.format_markdown(data)


def test_a_new_credential_rejection_is_excluded_not_a_control_failure(tmp_path):
    runs = _runs(tmp_path)
    tid = ds.RUN_ORDER[0][0]
    calls = [{"tag": f"heldout:router:{tid}", "ok": True, "prompt": 300, "output": 800},
             {"tag": f"heldout:control:{tid}", "ok": False, "prompt": 0, "output": 0,
              "error": "HTTP 401: {}"}]
    data = ds.combined_design_report(
        _supplement(tmp_path, [(PASS, (False, "the route failed the call: HTTP 401"))], calls),
        runs)
    assert data["second_supplement_only"]["valid_pairs"] == 0
    assert data["pooled"]["valid_pairs"] == 9


def test_an_ordinary_failed_call_keeps_its_pair_valid(tmp_path):
    runs = _runs(tmp_path)
    tid = ds.RUN_ORDER[0][0]
    calls = [{"tag": f"heldout:control:{tid}", "ok": False, "prompt": 0, "output": 0,
              "error": "HTTP 503: x"}]
    data = ds.combined_design_report(
        _supplement(tmp_path, [(PASS, (False, "the route failed the call: HTTP 503")),
                               (PASS, PASS)], calls), runs)
    assert data["second_supplement_only"]["valid_pairs"] == 2
    assert data["pooled"]["valid_pairs"] == 11


def test_overlapping_task_ids_are_refused(tmp_path):
    runs = _runs(tmp_path)
    _write(runs / ds.HISTORICAL_DIR_NAME / "tasks.jsonl",
           [{"id": "ds3-design-easy-recipe-card", "category": "design"}])
    with pytest.raises(SystemExit):
        ds.combined_design_report(_supplement(tmp_path, []), runs)


def test_readback_finds_duplicates_paid_calls_and_harness_failures(tmp_path):
    supp = _supplement(tmp_path, [(PASS, PASS), (PASS, PASS)])
    with (supp / "ledger.jsonl").open("a") as fh:
        fh.write(json.dumps(_row(ds.RUN_ORDER[3][0], "control-metered", True)) + "\n")
    calls = [{"tag": ds.ds1.call_tag(t, a), "ok": True, "cost_usd": 0.02 if "control" in a else 0,
              "list_cost_usd": 0.01 if "control" in a else 0, "prompt": 300, "output": 800}
             for t, a in ds.RUN_ORDER]
    calls.append(dict(calls[1]))
    calls.append({"tag": "heldout:router:ghost", "ok": False, "error": "HTTP 401",
                  "prompt": 0, "output": 0})
    _write(supp / "calls.jsonl", calls)
    out = ds.readback(supp, tmp_path / "nowhere")
    assert out["duplicate_rows"] == [(ds.RUN_ORDER[3][0], "control-metered")]
    assert out["duplicate_paid_calls"] == [ds.ds1.call_tag(*ds.RUN_ORDER[1])]
    assert out["harness_failures"] == ["heldout:router:ghost"]
    assert out["calls_without_row"] == ["heldout:router:ghost"]
    assert out["cap_basis_usd"] == pytest.approx(0.06)
    assert out["earlier_runs_unchanged"] == {ds.HISTORICAL_DIR_NAME: False,
                                             ds.FIRST_SUPPLEMENT_DIR_NAME: False}


def test_an_added_file_in_an_earlier_run_counts_as_a_change(tmp_path):
    runs = tmp_path / "runs"
    s1 = runs / ds.FIRST_SUPPLEMENT_DIR_NAME
    s1.mkdir(parents=True)
    assert ds._digests(s1, {}) == {}
    (s1 / "extra.txt").write_text("x")
    assert "extra.txt" in ds._digests(s1, {})


@pytest.mark.skipif(not (RUNS / ds.HISTORICAL_DIR_NAME).exists(),
                    reason="the earlier runs live only on the operator's machine")
def test_the_earlier_runs_are_byte_identical_to_their_pins():
    assert ds.pinned_state(RUNS) == {ds.HISTORICAL_DIR_NAME: True,
                                     ds.FIRST_SUPPLEMENT_DIR_NAME: True}
