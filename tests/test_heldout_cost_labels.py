"""Whole-arm cash labels must describe all of the aggregated cells."""
import json

from experiments import heldout


def _report(tmp_path, cells):
    heldout.preregister(tmp_path)
    rows = []
    for arm, category, costs in cells:
        for i, cost in enumerate(costs):
            rows.append(dict(task_id=f"{category}-{i}", category=category, arm=arm,
                             passed=True, grader_kind="synthetic", detail="",
                             model="metered-route" if cost is not None else "free-route",
                             latency_ms=100.0, observed_cost_usd=cost))
    (tmp_path / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return heldout.report(tmp_path)


def test_mixed_arm_keeps_billed_cells_and_free_remainder(tmp_path):
    free_categories = dict(zip(heldout.CATEGORIES[1:], (11, 11, 11, 12, 11)))
    cells = [("router", "design", [0.042442] * 14)]
    cells += [("router", category, [None] * n) for category, n in free_categories.items()]
    cells += [("unpriced", "math", [None] * 11)]
    data = _report(tmp_path, cells)
    assert data["arms_with_no_measurable_cost"] == ["unpriced"]
    mixed, = data["arms_with_partially_measured_cost"]
    assert mixed == dict(arm="router", metered_cost_usd_total=0.594188,
                         metered_cells=["design"], free_cells=list(free_categories))
    assert mixed["metered_cost_usd_total"] == data["per_category"]["design"]["router"]["metered_cost_usd"]
    text = heldout.format_report(data)
    cash_lines = [line for line in text.splitlines() if line.startswith("No cash comparison is possible")]
    assert cash_lines == ["No cash comparison is possible for: unpriced (every route used was free or unpriced)."]
    assert "Mixed cash measurement for: router" in text
    assert "$0.594188" in text
    assert "free or unpriced cells: " + ", ".join(free_categories) in text


def test_wholly_unpriced_arm_retains_legacy_sentence(tmp_path):
    data = _report(tmp_path, [("router", "design", [None]), ("router", "math", [None])])
    assert data["arms_with_no_measurable_cost"] == ["router"]
    assert data["arms_with_partially_measured_cost"] == []
    assert "No cash comparison is possible for: router" in heldout.format_report(data)


def test_measured_zero_is_not_unpriced_and_missing_cells_do_not_count(tmp_path):
    data = _report(tmp_path, [("zero", "math", [0.0]), ("paid", "design", [0.1])])
    assert data["arms_with_no_measurable_cost"] == []
    assert data["arms_with_partially_measured_cost"] == []
    assert "No cash comparison is possible" not in heldout.format_report(data)
