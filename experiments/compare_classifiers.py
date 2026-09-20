"""Replay classifier outputs through the real expected-cost policy.

This does not regenerate LLM answers.  It selects a route for every task in
the router's calibration set, then looks up that exact model/task outcome and
list-price cost in the frozen matrix.  Consequently it measures whether each
classifier picked a model that solved the task, without paying for a new LLM
run.  Classifier latency is measured by ``jev_eval.py classify``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate import build_catalog, load_matrix  # noqa: E402
from auto_router.catalog import Catalog  # noqa: E402
from auto_router.config import RouterConfig  # noqa: E402
from auto_router.jev import Classification  # noqa: E402
from auto_router.router import Router  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)]


def as_classification(row: dict, source: str) -> Classification:
    return Classification(
        category=row["jev_category"], category_probs={}, difficulty=float(row["difficulty"]),
        difficulty_confidence=float(row.get("confidence") or 0),
        needs_tools=float(row.get("needs_tools") or 0), needs_vision=float(row.get("needs_vision") or 0),
        needs_long_context=float(row.get("needs_long_context") or 0),
        follow_up=float(row.get("follow_up") or 0), stakes=float(row.get("stakes") or 0.5),
        latency_s=float(row.get("latency_s") or 0), model=source, source_name=source,
        failed=bool(row.get("failed")),
    )


def evaluate(name: str, classifications: dict, tasks: list[dict], matrix: dict,
             sims: dict, classifier_cost: float) -> dict:
    cfg = RouterConfig({}, Catalog([s.info for s in sims.values()]), policy={
        "name": "F_expected", "jev_difficulty_calibration": [0.27, 0.51],
        "success": {"table": "examples/success.measured.json"},
        "classifier": {"backend": "heuristic"},
    })
    chosen, solved, target_cost = {}, 0, 0.0
    for task in tasks:
        row = classifications[task["id"]]
        cls = as_classification(row, name)
        router = Router(cfg, classifier=lambda _text, _context, c=cls: c, quota_reader=lambda: {})
        result = router.route([{"role": "user", "content": task.get("prompt") or task.get("instruction") or ""}],
                              tools=[{"name": "tool"}] if task["category"] in ("tool_use", "agentic") else None,
                              max_tokens=12000)
        route = result.model.name
        chosen[route] = chosen.get(route, 0) + 1
        matrix_name = sims[route].matrix_name
        outcome = matrix[(matrix_name, task["id"])]
        solved += int(outcome["ok"])
        target_cost += float(outcome.get("list_cost_usd") or outcome.get("cost_usd") or 0)
    n = len(tasks)
    latency = [float(r.get("latency_s") or 0) for r in classifications.values()]
    total_cost = target_cost + n * classifier_cost
    return {
        "tasks": n, "solved": solved, "solve_rate": round(solved / n, 4),
        "target_cost_usd": round(target_cost, 6),
        "classifier_cost_usd": round(n * classifier_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "cost_per_solved_turn_usd": round(total_cost / max(1, solved), 6),
        "classifier_latency_p50_s": round(statistics.median(latency), 3),
        "classifier_latency_p95_s": round(percentile(latency, 0.95), 3),
        "routes": dict(sorted(chosen.items())),
    }


def evaluate_heuristic(tasks: list[dict], matrix: dict, sims: dict) -> dict:
    cfg = RouterConfig({}, Catalog([s.info for s in sims.values()]), policy={
        "name": "F_expected", "success": {"table": "examples/success.measured.json"},
        "classifier": {"backend": "heuristic"},
    })
    chosen, solved, cost = {}, 0, 0.0
    for task in tasks:
        router = Router(cfg, quota_reader=lambda: {})
        result = router.route([{"role": "user", "content": task.get("prompt") or task.get("instruction") or ""}],
                              tools=[{"name": "tool"}] if task["category"] in ("tool_use", "agentic") else None,
                              max_tokens=12000)
        route = result.model.name
        chosen[route] = chosen.get(route, 0) + 1
        outcome = matrix[(sims[route].matrix_name, task["id"])]
        solved += int(outcome["ok"])
        cost += float(outcome.get("list_cost_usd") or outcome.get("cost_usd") or 0)
    return {"tasks": len(tasks), "solved": solved, "solve_rate": round(solved / len(tasks), 4),
            "total_cost_usd": round(cost, 6),
            "cost_per_solved_turn_usd": round(cost / max(1, solved), 6),
            "classifier_latency_p50_s": 0.0, "classifier_latency_p95_s": 0.0,
            "routes": dict(sorted(chosen.items()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--hosted", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tasks = [json.loads(line) for line in open(args.tasks)]
    rows = [json.loads(line) for line in open(args.matrix)]
    matrix = {(r["model"], r["task"]): r for r in rows}
    _, stats = load_matrix(args.matrix)
    caps = json.load(open(Path(args.matrix).with_name("capabilities.json")))
    sims = build_catalog("list", caps, stats)
    cls = lambda p: {r["task"]: r for r in map(json.loads, open(p))}
    local, hosted = cls(args.local), cls(args.hosted)
    result = {
        "method": "exact frozen per-task replay through F_expected; no new LLM answers",
        "local_laya": evaluate("local-laya", local, tasks, matrix, sims, 0.0),
        "hosted_jev": evaluate("jev", hosted, tasks, matrix, sims, 0.00004),
        "heuristic": evaluate_heuristic(tasks, matrix, sims),
    }
    opus = [matrix[("claude-opus-5-aiml", t["id"])] for t in tasks]
    cost = sum(float(r.get("list_cost_usd") or r.get("cost_usd") or 0) for r in opus)
    solved = sum(int(r["ok"]) for r in opus)
    result["always_opus"] = {
        "tasks": len(tasks), "solved": solved, "solve_rate": round(solved / len(tasks), 4),
        "total_cost_usd": round(cost, 6),
        "cost_per_solved_turn_usd": round(cost / max(1, solved), 6),
        "classifier_latency_p50_s": 0.0, "classifier_latency_p95_s": 0.0,
        "routes": {"claude-opus-5": len(tasks)},
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
