"""The markdown report, and the separation question it has to answer out loud.

The risk this file guards is not a formatting bug. It is that the document a
reader reads says "no category separates the arms" when a category does, or the
reverse - so the tests build ledgers whose answer is known by construction and
check the sentence, not just the table.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import heldout, heldout_md                    # noqa: E402


def _row(task_id, category, arm, passed, detail="", cost=None, model="m"):
    return {"task_id": task_id, "category": category, "arm": arm, "passed": passed,
            "grader": category, "grader_kind": "exact", "detail": detail, "model": model,
            "latency_ms": 1.0, "prompt_tokens": 1, "cached_tokens": 0, "output_tokens": 1,
            "observed_cost_usd": cost, "cost_basis": "test", "estimated_cost_usd": None,
            "estimated_p_success": None, "cache_status": "n/a", "evidence_confidence": None,
            "safe_fallback": None, "error": None, "label": "live"}


def _run(tmp_path, rows, tasks):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "tasks.jsonl").write_text(
        "".join(json.dumps(t, sort_keys=True) + "\n" for t in tasks))
    record = {
        "registered_at": "2026-09-19T00:00:00Z", "seed": 1, "task_file": "tasks.jsonl",
        "task_file_sha256": heldout.digest(tmp_path / "tasks.jsonl"),
        "analysis_plan_sha256": heldout.plan_digest(), "code_sha256": {},
        "task_count": len(tasks),
        "tasks_by_category": {c: sum(1 for t in tasks if t["category"] == c)
                              for c in {t["category"] for t in tasks}},
        "grader_by_category": {t["category"]: "exact" for t in tasks},
        "task_ids": [t["id"] for t in tasks], "analysis_plan": heldout.ANALYSIS_PLAN,
        "note": "test",
    }
    (tmp_path / "preregistration.json").write_text(json.dumps(record))
    (tmp_path / "ledger.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def _identical_arms(n=12, category="math"):
    tasks, rows = [], []
    for i in range(n):
        task_id = f"t{i}"
        tasks.append({"id": task_id, "category": category})
        rows.append(_row(task_id, category, "router", True))
        rows.append(_row(task_id, category, "control-metered", True, cost=0.01))
    return tasks, rows


def test_overlapping_intervals_are_reported_as_no_separation(tmp_path):
    tasks, rows = _identical_arms()
    text = heldout_md.format_markdown(heldout_md.build(_run(tmp_path, rows, tasks)))
    assert "**Answer: no.**" in text
    assert "overlap" in text
    assert "not evidence that they are" in text


def test_a_real_separation_is_named_rather_than_buried(tmp_path):
    """Twelve tasks the router fails and the comparator passes: the intervals part."""
    tasks, rows = [], []
    for i in range(12):
        task_id = f"t{i}"
        tasks.append({"id": task_id, "category": "math"})
        rows.append(_row(task_id, "math", "router", False))
        rows.append(_row(task_id, "math", "control-metered", True, cost=0.01))
    data = heldout_md.build(_run(tmp_path, rows, tasks))
    assert data["separation"]["math"]["separated"] is True
    text = heldout_md.format_markdown(data)
    assert "**Answer: yes, in math.**" in text
    assert "DO NOT overlap" in text


def test_the_ten_task_floor_is_reported_per_category(tmp_path):
    tasks, rows = _identical_arms(n=6)
    text = heldout_md.format_markdown(heldout_md.build(_run(tmp_path, rows, tasks)))
    assert "no quality claim is made for: math" in text
    tasks, rows = _identical_arms(n=11)
    text = heldout_md.format_markdown(heldout_md.build(_run(tmp_path / "b", rows, tasks)))
    assert "Every category reaches the pre-registered floor" in text


def test_the_floor_is_judged_on_valid_pairs_not_on_each_arm(tmp_path):
    # Eleven tasks, each arm loses a DIFFERENT one to truncation: both arms keep
    # ten graded rows, but only nine tasks are graded on both.
    tasks, rows = _identical_arms(n=11)
    rows[0] = _row("t0", "math", "router", None, "TRUNCATED: hit the budget")
    rows[3] = _row("t1", "math", "control-metered", None, "TRUNCATED: hit the budget")
    data = heldout_md.build(_run(tmp_path, rows, tasks))
    assert data["separation"]["math"]["router"]["n"] == 10
    assert data["separation"]["math"]["comparator"]["n"] == 10
    assert data["separation"]["math"]["valid_pairs"] == 9
    text = heldout_md.format_markdown(data)
    assert "no quality claim is made for: math" in text


def test_a_duplicate_row_and_a_missing_arm_are_reported(tmp_path):
    tasks, rows = _identical_arms(n=11)
    rows.append(dict(rows[0]))
    del rows[3]
    text = heldout_md.format_markdown(heldout_md.build(_run(tmp_path, rows, tasks)))
    assert "**duplicate task/arm rows**" in text
    assert "missing arm" in text and "math: 1" in text


def test_a_category_with_one_arm_missing_is_under_the_floor(tmp_path):
    tasks, rows = _identical_arms(n=11)
    extra = [{"id": f"r{i}", "category": "research"} for i in range(11)]
    rows += [_row(f"r{i}", "research", "router", True) for i in range(11)]
    text = heldout_md.format_markdown(heldout_md.build(_run(tmp_path, rows, tasks + extra)))
    assert "no quality claim is made for: research" in text
    assert "Every category reaches" not in text


def test_a_non_overlap_below_the_floor_is_not_counted_as_separation(tmp_path):
    tasks, rows = [], []
    for i in range(9):
        tasks.append({"id": f"t{i}", "category": "math"})
        rows.append(_row(f"t{i}", "math", "router", True))
        rows.append(_row(f"t{i}", "math", "control-metered", False, cost=0.01))
    text = heldout_md.format_markdown(heldout_md.build(_run(tmp_path, rows, tasks)))
    assert "**Answer: no.**" in text
    assert "NOT counted as a separation: math" in text


def test_spend_is_summed_per_arm_and_a_free_arm_is_not_a_zero(tmp_path):
    tasks, rows = _identical_arms(n=10)
    data = heldout_md.build(_run(tmp_path, rows, tasks))
    assert heldout_md._spend_by_arm(data["report"]) == {"router": None,
                                                        "control-metered": 0.1}
    assert "every route free" in heldout_md.format_markdown(data)


def test_a_truncated_pair_shows_up_in_the_sensitivity_table(tmp_path):
    tasks, rows = _identical_arms(n=11)
    rows[0] = _row("t0", "math", "router", None,
                   "TRUNCATED: hit the 6000-token output budget")
    text = heldout_md.format_markdown(heldout_md.build(_run(tmp_path, rows, tasks)))
    assert "truncation-as-failure" in text


def test_intervals_overlap_is_symmetric_and_touching_counts_as_overlap():
    assert heldout_md.intervals_overlap([0.1, 0.4], [0.4, 0.9])
    assert heldout_md.intervals_overlap([0.4, 0.9], [0.1, 0.4])
    assert not heldout_md.intervals_overlap([0.1, 0.39], [0.4, 0.9])
