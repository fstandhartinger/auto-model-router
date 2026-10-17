#!/usr/bin/env python3
"""Ask the router for its pick on every task, before any model answers exist.

Builds the router the way auto-router-demo does (app/engine.py: catalog.json entries -> ModelInfo from Benchmark
Heaven via bench_id, SuccessModel from success.measured.json) and calls auto_router.irp.rank() exactly as the
whichmodel service does (app/irp.py), with candidates priced at AntSeed prices, for
  classifier in {s1, prior}  x  price basis in {cheapest, median}  x  cost_quality_tradeoff 0..10  x  60 tasks.
Only the top-ranked candidate is recorded. kimi-k3 is not in the demo catalog, so it gets the entry below.
usage: router_plan.py TASKS.jsonl PRICES.json OUT_PLAN.csv OUT_CATALOG.json
"""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from auto_router import s1_router
from auto_router.bench import BenchmarkClient
from auto_router.catalog import Catalog
from auto_router.config import RouterConfig, build_model
from auto_router.economics import SuccessModel
from auto_router.irp import rank
from auto_router.router import Router

HERE = Path(__file__).resolve().parent
#: demo catalog name -> our model key
DEMO_NAMES = {"claude-opus-5": "claude-opus-5", "gpt-5.6-sol": "gpt-5.6-sol", "gpt-5.6-luna": "gpt-5.6-luna",
              "glm-5.3-flash": "glm-5.3-flash", "dsv4.1-flash": "deepseek-v4.1-flash"}
KIMI = {"name": "kimi-k3", "bench_id": "kimi-k3::max", "bench_offer": {"platform": "OpenRouter"}, "latency_s": 6.0,
        "cache": {"ttl_seconds": 300, "hit_rate": 0.9}, "irp_model_id": "moonshotai/kimi-k3"}


class Memo:
    """Same classifier, computed once per request (it is deterministic); fails loudly instead of degrading."""

    def __init__(self, inner):
        self.inner, self.cache, self.calls = inner, {}, []

    def __call__(self, request, context=""):
        key = (request, context)
        if key not in self.cache:
            result = self.inner(request, context)
            assert not result.failed, "s1 classifier fell back; refusing to record a prior-only pick as s1"
            self.cache[key] = result
        return self.cache[key]


def build(prices):
    demo = {m["name"]: m for m in json.load(open(HERE / "demo-catalog.json"))["models"]}
    entries = []
    for name in DEMO_NAMES:
        e = dict(demo[name])
        e.pop("demo", None)
        entries.append({k: v for k, v in e.items() if v is not None})
    entries.append(dict(KIMI))
    bench = BenchmarkClient()
    models = [build_model(e, {}, bench) for e in entries]
    assert not bench.errors, bench.errors
    config = RouterConfig(providers={}, catalog=Catalog(models), subscriptions={}, policy={"name": "F_expected"},
                          raw={"models": entries})
    measured = {(m, c, b): p for m, c, b, p in json.load(open(HERE / "success.measured.json"))}
    router = Router(config, success=SuccessModel(measured=measured), classifier=None, quota_reader=lambda: {})
    return config, router, models, entries


def view(config, router, classifier):
    return SimpleNamespace(config=config, success=router.success, classifier=classifier,
                           _turn_request=router._turn_request, _summary=router._summary)


def main(tasks_path, prices_path, out_plan, out_catalog):
    tasks = [json.loads(l) for l in open(tasks_path)]
    prices = json.load(open(prices_path))["models"]
    config, router, models, entries = build(prices)
    irp_id = {DEMO_NAMES.get(e["name"], e["name"]): e["irp_model_id"] for e in entries}
    key_of = {v: k for k, v in irp_id.items()}
    s1 = Memo(s1_router.LocalS1Classifier(s1_router.REPO, 2))
    views = {"s1": view(config, router, s1), "prior": view(config, router, None)}
    rows = []
    for basis in ("cheapest", "median"):
        cands = [{"id": k, "model": irp_id[k], "pricing": {
            "input": prices[k][basis]["input"], "cache_read": prices[k][basis]["cache_read"], "output": prices[k][basis]["output"]}}
            for k in irp_id]
        for t in tasks:
            for cname, v in views.items():
                for tradeoff in range(11):
                    body = {"request": {"messages": [{"role": "user", "content": t["prompt"]}], "max_tokens": 8000},
                            "routing": {"cost_quality_tradeoff": tradeoff, "candidates": cands}}
                    ranked = rank(v, body)["ranked"]
                    top = ranked[0]
                    rows.append({"task": t["id"], "classifier": cname, "price_basis": basis, "tradeoff": tradeoff,
                                 "pick": top["candidate_id"], "expected_quality": round(top["expected_quality"], 6),
                                 "expected_cost_usd": f"{top['expected_cost_usd']:.9f}",
                                 "ranking": ">".join(e["candidate_id"] for e in ranked)})
    with open(out_plan, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cat = {"success_measured_rows": len(json.load(open(HERE / "success.measured.json"))), "models": {m.name: {"capability": {k: round(v, 3) for k, v in sorted(m.capability.items())},
                                                                      "max_output_tokens": m.max_output_tokens} for m in models},
           "s1_revision": s1_router.REVISION, "s1_files_sha256": s1_router.FILES,
           "note": "capabilities fetched from benchmarkheaven.com on the run date; picks in router-picks.csv are the record"}
    json.dump(cat, open(out_catalog, "w"), indent=1, sort_keys=True)
    print(len(rows), "plan rows")


if __name__ == "__main__":
    main(*sys.argv[1:5])
