#!/usr/bin/env python3
"""Wave 3: ask the routers for their pick on every task BEFORE any model answers exist.

Router build, candidates and call are those of wave 2 (../wave2/router_plan.py: auto_router.irp.rank() as whichmodel calls it, six candidates
priced on three bases, max_tokens 16000). Arms (classifier x success model):
  v1-s1     s1 classifier + success model v1 (curve + measured cells that override it)  = what whichmodel serves today
  v2-s1     s1 classifier + success model v2 (auto_router/data/success_v2.json, PR #10; measured cells shrunk toward the curve)
  v1-prior  no classifier (conservative prior) + v1
  v2-prior  no classifier + v2   (exploratory; not part of any registered headline)
Rows: task x arm x basis {median, cheapest, list} x tradeoff 0..10.
usage: router_plan.py TASKS_FULL.jsonl PRICES.json OUT_PLAN.csv OUT_CATALOG.json"""
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repo root (auto_router)
sys.path.insert(0, str(HERE.parent))
import router_plan as w1  # noqa: E402
from auto_router import s1_router  # noqa: E402
from auto_router.economics import SuccessModel, success_v2  # noqa: E402
from auto_router.irp import rank  # noqa: E402
import auto_router  # noqa: E402

BASES = ("median", "cheapest", "list")
MAX_TOKENS = 16000
ARMS = ("v1-s1", "v2-s1", "v1-prior", "v2-prior")


def main(tasks_path, prices_path, out_plan, out_catalog):
    tasks = [json.loads(l) for l in open(tasks_path)]
    prices = json.load(open(prices_path))["models"]
    config, router, models, entries = w1.build(prices)
    measured = {(m, c, b): p for m, c, b, p in json.load(open(HERE.parent / "success.measured.json"))}
    v1 = SuccessModel(measured=dict(measured))
    v2 = success_v2(measured=dict(measured))
    irp_id = {w1.DEMO_NAMES.get(e["name"], e["name"]): e["irp_model_id"] for e in entries}
    s1 = w1.Memo(s1_router.LocalS1Classifier(s1_router.REPO, 2))
    from types import SimpleNamespace
    mk = lambda success, clf: SimpleNamespace(config=config, success=success, classifier=clf, _turn_request=router._turn_request, _summary=router._summary)  # noqa: E731
    views = {"v1-s1": mk(v1, s1), "v2-s1": mk(v2, s1), "v1-prior": mk(v1, None), "v2-prior": mk(v2, None)}
    rows = []
    for basis in BASES:
        cands = [{"id": k, "model": irp_id[k], "pricing": {"input": prices[k][basis]["input"], "cache_read": prices[k][basis]["cache_read"],
                                                           "output": prices[k][basis]["output"]}} for k in irp_id]
        for t in tasks:
            for arm in ARMS:
                for tradeoff in range(11):
                    body = {"request": {"messages": [{"role": "user", "content": t["prompt"]}], "max_tokens": MAX_TOKENS},
                            "routing": {"cost_quality_tradeoff": tradeoff, "candidates": cands}}
                    ranked = rank(views[arm], body)["ranked"]
                    top = ranked[0]
                    rows.append({"task": t["id"], "classifier": arm, "price_basis": basis, "tradeoff": tradeoff, "pick": top["candidate_id"],
                                 "expected_quality": round(top["expected_quality"], 6), "expected_cost_usd": f"{top['expected_cost_usd']:.9f}",
                                 "ranking": ">".join(e["candidate_id"] for e in ranked)})
    with open(out_plan, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    p2 = Path(auto_router.__file__).parent / "data" / "success_v2.json"
    cls = {}
    for t in tasks:  # the s1 classification each router saw (same Memo, deterministic)
        c = s1((t["prompt"]), router._summary([{"role": "user", "content": t["prompt"]}], None))
        req = router._turn_request(c, 500, MAX_TOKENS, 0.0, False, [{"role": "user", "content": t["prompt"]}])
        cls[t["id"]] = {"category": req.category, "difficulty": round(req.difficulty, 5), "s1_category": c.category}
    cat = {"success_measured_rows": len(measured),
           "models": {m.name: {"capability": {k: round(v, 3) for k, v in sorted(m.capability.items())}, "max_output_tokens": m.max_output_tokens,
                               "task_tokens": m.task_tokens} for m in models},
           "success_v2_file_sha256": hashlib.sha256(p2.read_bytes()).hexdigest(), "success_v2_params": json.load(open(p2))["params"],
           "s1_classification_per_task": cls,
           "s1_revision": s1_router.REVISION, "s1_files_sha256": s1_router.FILES, "max_tokens_in_request": MAX_TOKENS,
           "note": "router build of waves 1-2 (demo-catalog.json, success.measured.json) with the success models above; picks in router-plan.csv are the record"}
    json.dump(cat, open(out_catalog, "w"), indent=1, sort_keys=True)
    print(len(rows), "plan rows")


if __name__ == "__main__":
    main(*sys.argv[1:5])
