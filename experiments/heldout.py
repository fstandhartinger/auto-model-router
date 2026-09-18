"""Pre-registered held-out evaluation across six task categories.

The categories are the ones the next cycle asked for: web/UI design, coding,
maths/reasoning, factual research, summarisation, and cache-eligible repeat
tasks.

Pre-registration
----------------
``python experiments/heldout.py preregister`` writes the task set, the grader
for each task, the analysis plan and a SHA-256 of the task file, **before any
model is called**. The runner refuses to start unless a pre-registration
exists and its digest still matches the task file, so tasks cannot be quietly
edited after an outcome is seen. The digest appears in the ledger and in the
report.

Paired comparison
-----------------
Every task is run twice on the same prompt, in the same process, against the
same catalog: once through the router's policy and once through the *control*
policy, which is the ordinary "one capable model for everything" behaviour a
normal setup has. The control is a real default, not an inflated one, and it is
named in the report.

What is measured per task
-------------------------
classification, selected route, estimated cost and success probability
(before), then observed pass/fail, latency, tokens, cache status and cost
(after) - each kept in its own object, exactly as ``auto_router.decision``
defines them.

Labels
------
Every number in the report is tagged ``live`` (a real paired call made in this
run), ``replay`` (recomputed from an earlier run's stored answers) or
``estimate`` (the router's own forecast). They are never mixed into one figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import graders, sandbox           # noqa: E402
from experiments.tasks_heldout import build_tasks  # noqa: E402

DEFAULT_DIR = Path(os.environ.get("AUTO_ROUTER_HELDOUT_DIR",
                                  Path(__file__).resolve().parent.parent.parent / "runs" / "heldout"))

CATEGORIES = ("design", "coding", "math", "research", "summarisation", "cache_repeat")

ANALYSIS_PLAN = {
    "question": "Does policy-based routing solve held-out tasks at least as often as the "
                "ordinary single-model control, and at what measured cost?",
    "design": "Paired: every task is sent to both arms with an identical prompt in the same run.",
    "primary_outcome": "pass rate per category, graded by the pre-registered deterministic grader",
    "secondary_outcomes": ["observed cost per solved task (metered routes only)",
                           "observed latency", "route selected", "cache status",
                           "estimated versus observed cost"],
    "uncertainty": "Wilson 95% interval per category; no aggregate is reported without the "
                   "per-category table beside it.",
    "stopping_rule": "The run stops when the task set is exhausted or the spend cap is reached; "
                     "a partial run reports the categories it completed and names the rest.",
    "exclusions": "A row is excluded from pass rates and counted separately when the grader "
                  "reports SANDBOX UNAVAILABLE or the answer hit the output budget (TRUNCATED). "
                  "Both are harness failures, not model failures.",
    "claims_not_made": [
        "No cash saving is claimed from estimated or list-price arithmetic.",
        "No quality claim is made from a category with fewer than 10 graded tasks.",
        "The design grader is a structural proxy and is reported as such.",
        "Replay and simulation results are never combined with live results.",
    ],
}


# ---------------------------------------------------------------------------
# pre-registration
# ---------------------------------------------------------------------------
def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preregister(out_dir: Path, seed: int = 20260918) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(random.Random(seed))
    task_file = out_dir / "tasks.jsonl"
    with task_file.open("w") as fh:
        for task in tasks:
            fh.write(json.dumps(task, sort_keys=True) + "\n")
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["category"]] = counts.get(task["category"], 0) + 1
    record = {
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "task_file": task_file.name,
        "task_file_sha256": digest(task_file),
        "task_count": len(tasks),
        "tasks_by_category": counts,
        "grader_by_category": {c: graders.GRADER_KIND[
            next(t["grader"] for t in tasks if t["category"] == c)] for c in counts},
        "task_ids": [t["id"] for t in tasks],
        "analysis_plan": ANALYSIS_PLAN,
        "note": "Written before any model was called. The runner verifies this digest.",
    }
    (out_dir / "preregistration.json").write_text(json.dumps(record, indent=1))
    return record


def load_preregistration(out_dir: Path) -> dict:
    path = out_dir / "preregistration.json"
    if not path.exists():
        raise SystemExit(f"no pre-registration in {out_dir}; run `preregister` first")
    record = json.loads(path.read_text())
    task_file = out_dir / record["task_file"]
    actual = digest(task_file)
    if actual != record["task_file_sha256"]:
        raise SystemExit(
            f"task file digest changed since pre-registration\n"
            f"  registered: {record['task_file_sha256']}\n  actual:     {actual}\n"
            "Re-register deliberately if the task set really should change.")
    return record


def load_tasks(out_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (out_dir / "tasks.jsonl").read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval. Honest about small samples, unlike the normal approximation."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


@dataclass
class TaskOutcome:
    task_id: str
    category: str
    arm: str                    # "router" | "control"
    passed: bool | None
    grader: str
    grader_kind: str
    detail: str
    model: str
    latency_ms: float
    prompt_tokens: int
    cached_tokens: int
    output_tokens: int
    observed_cost_usd: float | None
    cost_basis: str
    estimated_cost_usd: float | None
    estimated_p_success: float | None
    cache_status: str
    evidence_confidence: float | None
    safe_fallback: str | None
    error: str | None = None
    label: str = "live"


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def report(out_dir: Path) -> dict:
    ledger = out_dir / "ledger.jsonl"
    if not ledger.exists():
        raise SystemExit(f"no ledger at {ledger}; run `run` first")
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    prereg = json.loads((out_dir / "preregistration.json").read_text())

    by_arm_category: dict[tuple[str, str], list[dict]] = {}
    excluded: list[dict] = []
    for row in rows:
        detail = row.get("detail") or ""
        if row.get("passed") is None or "SANDBOX UNAVAILABLE" in detail or "TRUNCATED" in detail:
            excluded.append(row)
            continue
        by_arm_category.setdefault((row["arm"], row["category"]), []).append(row)

    categories: dict[str, dict] = {}
    for category in CATEGORIES:
        entry: dict = {}
        for arm in ("router", "control"):
            rows_ = by_arm_category.get((arm, category), [])
            if not rows_:
                continue
            passed = sum(1 for r in rows_ if r["passed"])
            low, high = wilson(passed, len(rows_))
            metered = [r["observed_cost_usd"] for r in rows_
                       if isinstance(r.get("observed_cost_usd"), (int, float))]
            entry[arm] = {
                "n": len(rows_), "passed": passed,
                "pass_rate": round(passed / len(rows_), 4),
                "wilson_95": [round(low, 4), round(high, 4)],
                "grader_kind": rows_[0]["grader_kind"],
                "label": rows_[0].get("label", "live"),
                "metered_cost_usd": round(sum(metered), 6) if metered else None,
                "metered_calls": len(metered),
                "unmetered_calls": len(rows_) - len(metered),
                "median_latency_ms": round(sorted(r["latency_ms"] for r in rows_)[len(rows_) // 2], 1),
                "routes_used": sorted({r["model"] for r in rows_}),
                # Provider-reported, not inferred: the share of input tokens the
                # upstream actually billed as cache reads.
                "observed_prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in rows_),
                "observed_cached_tokens": sum(r.get("cached_tokens") or 0 for r in rows_),
                "router_cache_status": sorted({r.get("cache_status") or "n/a" for r in rows_}),
                "mean_estimated_p_success": (
                    round(sum(r["estimated_p_success"] for r in rows_
                              if isinstance(r.get("estimated_p_success"), (int, float)))
                          / max(1, sum(1 for r in rows_
                                       if isinstance(r.get("estimated_p_success"), (int, float)))), 3)
                    if any(isinstance(r.get("estimated_p_success"), (int, float)) for r in rows_)
                    else None),
            }
        if entry:
            categories[category] = entry

    small = [c for c, e in categories.items()
             if min((a["n"] for a in e.values()), default=0) < 10]
    out = {
        "preregistration_sha256": prereg["task_file_sha256"],
        "registered_at": prereg["registered_at"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "graded_rows": len(rows) - len(excluded),
        "excluded_rows": len(excluded),
        "exclusion_reasons": sorted({(r.get("detail") or "")[:60] for r in excluded}),
        "per_category": categories,
        "categories_too_small_for_a_quality_claim": sorted(small),
        "labels_present": sorted({r.get("label", "live") for r in rows}),
        "claims_not_made": ANALYSIS_PLAN["claims_not_made"],
    }
    (out_dir / "report.json").write_text(json.dumps(out, indent=1))
    return out


def format_report(data: dict) -> str:
    lines = [f"Held-out evaluation - task set {data['preregistration_sha256'][:12]} "
             f"registered {data['registered_at']}",
             f"graded {data['graded_rows']} rows, excluded {data['excluded_rows']}",
             ""]
    header = f"{'category':<14}{'arm':<9}{'n':>4}{'pass':>6}{'rate':>8}{'95% CI':>16}{'cost $':>10}  grader"
    lines += [header, "-" * len(header)]
    for category, arms in data["per_category"].items():
        for arm, e in arms.items():
            ci = f"{e['wilson_95'][0]:.2f}-{e['wilson_95'][1]:.2f}"
            cost = "n/a" if e["metered_cost_usd"] is None else f"{e['metered_cost_usd']:.4f}"
            lines.append(f"{category:<14}{arm:<9}{e['n']:>4}{e['passed']:>6}"
                         f"{e['pass_rate']:>8.2f}{ci:>16}{cost:>10}  {e['grader_kind']}")
    if data["categories_too_small_for_a_quality_claim"]:
        lines += ["", "No quality claim is made for: "
                  + ", ".join(data["categories_too_small_for_a_quality_claim"])
                  + " (fewer than 10 graded tasks)."]
    lines += ["", "Not claimed:"] + [f"  - {c}" for c in data["claims_not_made"]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preregister", "verify", "run", "report", "sandbox"])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--config", type=Path, help="router config for a live run")
    parser.add_argument("--budget", type=float, default=0.0, help="hard USD cap for a live run")
    parser.add_argument("--limit", type=int, default=0, help="stop after N tasks per arm")
    parser.add_argument("--categories", default=",".join(CATEGORIES))
    parser.add_argument("--control", help="control route name; default is the highest "
                                          "general capability in the catalog")
    args = parser.parse_args(argv)

    if args.command == "preregister":
        record = preregister(args.dir)
        print(json.dumps({k: v for k, v in record.items() if k != "task_ids"}, indent=1))
        return 0
    if args.command == "verify":
        record = load_preregistration(args.dir)
        print(f"pre-registration intact: {record['task_count']} tasks, "
              f"sha256 {record['task_file_sha256'][:16]}")
        return 0
    if args.command == "sandbox":
        print(json.dumps(sandbox.preflight(), indent=1))
        return 0
    if args.command == "report":
        print(format_report(report(args.dir)))
        return 0

    from experiments.heldout_run import run_live
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
