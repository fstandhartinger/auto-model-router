#!/usr/bin/env python3
"""Wave 2: ask the router for its pick on every task BEFORE any model answers exist. Same router build and call as
wave 1 (../router_plan.py), differences: max_tokens 16000, and three price bases (median seller = primary, cheapest seller,
OpenRouter list price). Rows: task x classifier {s1, prior} x basis x tradeoff 0..10.
usage: router_plan.py TASKS_FULL.jsonl PRICES.json OUT_PLAN.csv OUT_CATALOG.json"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import router_plan as w1  # noqa: E402
from auto_router import s1_router  # noqa: E402
from auto_router.irp import rank  # noqa: E402

BASES = ("median", "cheapest", "list")
MAX_TOKENS = 16000


def main(tasks_path, prices_path, out_plan, out_catalog):
    tasks = [json.loads(l) for l in open(tasks_path)]
    prices = json.load(open(prices_path))["models"]
    config, router, models, entries = w1.build(prices)
    irp_id = {w1.DEMO_NAMES.get(e["name"], e["name"]): e["irp_model_id"] for e in entries}
    s1 = w1.Memo(s1_router.LocalS1Classifier(s1_router.REPO, 2))
    views = {"s1": w1.view(config, router, s1), "prior": w1.view(config, router, None)}
    rows = []
    for basis in BASES:
        cands = [{"id": k, "model": irp_id[k], "pricing": {"input": prices[k][basis]["input"], "cache_read": prices[k][basis]["cache_read"],
                                                           "output": prices[k][basis]["output"]}} for k in irp_id]
        for t in tasks:
            for cname, v in views.items():
                for tradeoff in range(11):
                    body = {"request": {"messages": [{"role": "user", "content": t["prompt"]}], "max_tokens": MAX_TOKENS},
                            "routing": {"cost_quality_tradeoff": tradeoff, "candidates": cands}}
                    ranked = rank(v, body)["ranked"]
                    top = ranked[0]
                    rows.append({"task": t["id"], "classifier": cname, "price_basis": basis, "tradeoff": tradeoff, "pick": top["candidate_id"],
                                 "expected_quality": round(top["expected_quality"], 6), "expected_cost_usd": f"{top['expected_cost_usd']:.9f}",
                                 "ranking": ">".join(e["candidate_id"] for e in ranked)})
    with open(out_plan, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cat = {"success_measured_rows": len(json.load(open(HERE.parent / "success.measured.json"))),
           "models": {m.name: {"capability": {k: round(v, 3) for k, v in sorted(m.capability.items())}, "max_output_tokens": m.max_output_tokens} for m in models},
           "s1_revision": s1_router.REVISION, "s1_files_sha256": s1_router.FILES, "max_tokens_in_request": MAX_TOKENS,
           "note": "same router build as wave 1 (../router_plan.py, demo-catalog.json, success.measured.json); picks in router-plan.csv are the record"}
    json.dump(cat, open(out_catalog, "w"), indent=1, sort_keys=True)
    print(len(rows), "plan rows")


if __name__ == "__main__":
    main(*sys.argv[1:5])
