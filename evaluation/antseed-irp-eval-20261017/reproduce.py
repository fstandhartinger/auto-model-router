#!/usr/bin/env python3
"""Recompute analysis.json, router-picks.csv and summary.md from the published CSVs. No API calls, no model.

  python reproduce.py            # writes the three files next to this script
  python reproduce.py --check    # recomputes into memory and fails unless the files on disk are byte-identical

Inputs: tasks-public.jsonl (categories), results.csv (every model's graded answer), router-plan.csv (the router's pick
per setting, recorded before any answer existed), prices.json. Analysis plan (frozen in manifest.json before the run):
see README.md "Analysis plan".
"""
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20261017
DRAWS = 10_000
BASES = ("cheapest", "median")
MODELS = ["claude-opus-5", "gpt-5.6-sol", "gpt-5.6-luna", "kimi-k3", "glm-5.3-flash", "deepseek-v4.1-flash"]
MARGIN = 0.05


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 6), round(c + h, 6)]


def load():
    tasks = [json.loads(l) for l in open(HERE / "tasks-public.jsonl")]
    ids = [t["id"] for t in tasks]
    prices = json.load(open(HERE / "prices.json"))["models"]
    res = {}
    for r in csv.DictReader(open(HERE / "results.csv")):
        res[(r["task"], r["model"])] = {"passed": int(r["passed"]), "cheapest": float(r["cost_antseed"]),
                                        "median": float(r["cost_antseed_median"])}
    plan = {}
    for r in csv.DictReader(open(HERE / "router-plan.csv")):
        plan[(r["task"], r["classifier"], r["price_basis"], int(r["tradeoff"]))] = r["pick"]
    return tasks, ids, prices, res, plan


def r6(x):
    return round(float(x), 9)


def analyse():
    tasks, ids, prices, res, plan = load()
    n = len(ids)
    cat = [t["category"] for t in tasks]
    dif = [t["difficulty"] for t in tasks]
    rng = random.Random(SEED)
    draws = np.array([[rng.randrange(n) for _ in range(n)] for _ in range(DRAWS)], dtype=np.int64)
    # cheapest model by AntSeed price: lowest cheapest-seller input+output price
    cheapest_model = min(MODELS, key=lambda m: (prices[m]["cheapest"]["input"] + prices[m]["cheapest"]["output"], m))
    out = {"n_tasks": n, "bootstrap": {"draws": DRAWS, "seed": SEED, "method": "paired, resampling tasks, percentile 95% CI"},
           "models": MODELS, "always_cheapest_model": cheapest_model}
    per_model_pass = {m: sum(res[(i, m)]["passed"] for i in ids) for m in MODELS}
    total_cost0 = {m: sum(res[(i, m)]["cheapest"] for i in ids) for m in MODELS}
    best = max(MODELS, key=lambda m: (per_model_pass[m], -total_cost0[m], m))
    out["always_best_model"] = best
    out["note_always_best"] = "highest pass rate over the 6 single models on these tasks (ties: cheaper total at the cheapest-seller price)"
    out["bases"] = {}
    for basis in BASES:
        def vec(picks):  # picks: list of model per task
            return (np.array([res[(i, m)]["passed"] for i, m in zip(ids, picks)], dtype=np.int64),
                    np.array([res[(i, m)][basis] for i, m in zip(ids, picks)], dtype=float))
        arms = {}
        for m in MODELS:
            arms[f"always-{m}"] = {"kind": "single", "picks": [m] * n}
        arms["always-best"] = {"kind": "single", "picks": [best] * n}
        arms["always-cheapest"] = {"kind": "single", "picks": [cheapest_model] * n}
        orc = []
        for i in ids:
            ok = [m for m in MODELS if res[(i, m)]["passed"]]
            orc.append(min(ok, key=lambda m: (res[(i, m)][basis], m)) if ok else cheapest_model)
        arms["oracle"] = {"kind": "oracle", "picks": orc}
        for clf in ("s1", "prior"):
            for t in range(11):
                arms[f"router-{clf}-t{t}"] = {"kind": "router", "classifier": clf, "tradeoff": t,
                                              "picks": [plan[(i, clf, basis, t)] for i in ids]}
        bp, bc = vec(arms["always-best"]["picks"])
        best_total = bc.sum()
        op, oc = vec(arms["always-claude-opus-5"]["picks"])
        table = {}
        for name, a in arms.items():
            p, c = vec(a["picks"])
            dp = (draws.shape and (p[draws].mean(axis=1) - bp[draws].mean(axis=1)))
            cr = c[draws].sum(axis=1) / bc[draws].sum(axis=1)
            entry = {"kind": a["kind"], "passed": int(p.sum()), "n": n, "pass_rate": r6(p.mean()),
                     "pass_rate_wilson95": wilson(int(p.sum()), n), "total_cost_usd": r6(c.sum()),
                     "mean_cost_per_task_usd": r6(c.mean()), "cost_ratio_vs_always_best": r6(c.sum() / best_total),
                     "cost_ratio_ci95": [r6(np.percentile(cr, 2.5)), r6(np.percentile(cr, 97.5))],
                     "pass_diff_vs_always_best": r6(p.mean() - bp.mean()),
                     "pass_diff_ci95": [r6(np.percentile(dp, 2.5)), r6(np.percentile(dp, 97.5))]}
            # added after the first look at the results (see README "Amendments"): same statistics against always-Opus
            dpo = p[draws].mean(axis=1) - op[draws].mean(axis=1)
            cro = c[draws].sum(axis=1) / oc[draws].sum(axis=1)
            entry.update({"cost_ratio_vs_always_opus": r6(c.sum() / oc.sum()),
                          "cost_ratio_vs_always_opus_ci95": [r6(np.percentile(cro, 2.5)), r6(np.percentile(cro, 97.5))],
                          "pass_diff_vs_always_opus": r6(p.mean() - op.mean()),
                          "pass_diff_vs_always_opus_ci95": [r6(np.percentile(dpo, 2.5)), r6(np.percentile(dpo, 97.5))]})
            if a["kind"] == "router":
                entry["classifier"], entry["tradeoff"] = a["classifier"], a["tradeoff"]
                entry["picks_by_model"] = {m: a["picks"].count(m) for m in MODELS if a["picks"].count(m)}
            table[name] = entry
        # headline: cheapest router setting per classifier meeting each criterion, against each reference arm
        head = {}
        for ref, sfx in (("always_best", ""), ("always_opus", "_opus")):
            dk, ck, rk = f"pass_diff_vs_{ref}", f"pass_diff_vs_{ref}_ci95" if sfx else "pass_diff_ci95", f"cost_ratio_vs_{ref}"
            ckr = f"cost_ratio_vs_{ref}_ci95" if sfx else "cost_ratio_ci95"
            dk = dk if sfx else "pass_diff_vs_always_best"
            for clf in ("s1", "prior"):
                sets = [table[f"router-{clf}-t{t}"] for t in range(11)]
                crit = {"not_significantly_worse": lambda e: e[ck][1] >= 0,
                        "point_estimate_at_least": lambda e: e[dk] >= 0,
                        f"noninferior_within_{int(MARGIN * 100)}_points": lambda e: e[ck][0] >= -MARGIN}
                for cname, f in crit.items():
                    ok = [e for e in sets if f(e)]
                    key = f"{clf}/{cname}/vs_{ref}"
                    if ok:
                        e = min(ok, key=lambda e: (e["total_cost_usd"], e["tradeoff"]))
                        head[key] = {"tradeoff": e["tradeoff"], "pass_rate": e["pass_rate"], "pass_diff": e[dk], "pass_diff_ci95": e[ck],
                                     "cost_pct": r6(100 * e[rk]), "cost_pct_ci95": [r6(100 * x) for x in e[ckr]],
                                     "picks_by_model": e["picks_by_model"]}
                    else:
                        head[key] = None
        # s1 vs prior per tradeoff (paired)
        svp = {}
        for t in range(11):
            ps, cs = vec(arms[f"router-s1-t{t}"]["picks"])
            pp, cp = vec(arms[f"router-prior-t{t}"]["picks"])
            d = ps[draws].mean(axis=1) - pp[draws].mean(axis=1)
            cr = cs[draws].sum(axis=1) / cp[draws].sum(axis=1)
            svp[str(t)] = {"pass_diff_s1_minus_prior": r6(ps.mean() - pp.mean()), "ci95": [r6(np.percentile(d, 2.5)), r6(np.percentile(d, 97.5))],
                           "cost_ratio_s1_over_prior": r6(cs.sum() / cp.sum()), "cost_ratio_ci95": [r6(np.percentile(cr, 2.5)), r6(np.percentile(cr, 97.5))]}
        # per category / difficulty for the main arms (+ headline-like router settings 0..10 for s1)
        def split(key):
            res_ = {}
            for g in sorted(set(key)):
                idx = [k for k, x in enumerate(key) if x == g]
                res_[g] = {"n": len(idx)}
                for name, a in arms.items():
                    p, c = vec(a["picks"])
                    res_[g][name] = {"pass_rate": r6(p[idx].mean()), "cost_usd": r6(c[idx].sum())}
            return res_
        out["bases"][basis] = {"arms": table, "headline": head, "s1_vs_prior": svp,
                               "by_category": split(cat), "by_difficulty": split(dif)}
    out["tasks_failed_by_all_models"] = [i for i in ids if not any(res[(i, m)]["passed"] for m in MODELS)]
    out["tasks_passed_by_all_models"] = sum(1 for i in ids if all(res[(i, m)]["passed"] for m in MODELS))
    out["single_model_pass_by_category"] = {m: {g: sum(res[(i, m)]["passed"] for i, c_ in zip(ids, cat) if c_ == g) for g in sorted(set(cat))} for m in MODELS}
    return out, ids, res, plan, prices


def picks_csv(out, ids, res, plan):
    lines = ["task,arm,tradeoff,picked_model,passed,cost_antseed,price_basis"]
    for basis in BASES:
        arms = out["bases"][basis]["arms"]
        for clf in ("s1", "prior"):
            for t in range(11):
                for i in ids:
                    m = plan[(i, clf, basis, t)]
                    lines.append(f"{i},router-{clf},{t},{m},{res[(i, m)]['passed']},{res[(i, m)][basis]:.9f},{basis}")
    return "\n".join(lines) + "\n"


def summary_md(out):
    return ("### Cheapest-seller prices (primary)\n\n" + table_md(out, "cheapest") + "\n### Median-seller prices (sensitivity)\n\n"
            + table_md(out, "median") + "\n### Per category (pass rate / total cost, cheapest-seller prices)\n\n" + category_md(out)
            + "\n### Against always-Opus, cheapest-seller prices\n\n" + opus_md(out, "cheapest"))


def category_md(out):
    by = out["bases"]["cheapest"]["by_category"]
    cats = list(by)
    L = ["| Arm | " + " | ".join(f"{c} (n={by[c]['n']})" for c in cats) + " |", "|---|" + "---|" * len(cats)]
    names = [f"always-{m}" for m in MODELS] + ["oracle"] + [f"router-s1-t{t}" for t in (0, 2, 5, 8, 10)] + [f"router-prior-t{t}" for t in (0, 2, 8)]
    for nm in names:
        L.append(f"| {nm} | " + " | ".join(f"{100 * by[c][nm]['pass_rate']:.0f}% / ${by[c][nm]['cost_usd']:.5f}" for c in cats) + " |")
    return "\n".join(L) + "\n"


def opus_md(out, basis):
    b = out["bases"][basis]
    L = ["| Arm | Pass rate | Total cost | Cost vs always-Opus (95% CI) | Pass diff vs Opus (95% CI) |", "|---|---|---|---|---|"]
    names = ["always-claude-opus-5", "always-best", "always-cheapest", "oracle"] + [f"router-s1-t{t}" for t in range(11)] + [f"router-prior-t{t}" for t in range(11)]
    for nm in names:
        e = b["arms"][nm]
        L.append(f"| {nm} | {100 * e['pass_rate']:.1f}% | ${e['total_cost_usd']:.5f} | {100 * e['cost_ratio_vs_always_opus']:.1f}% ({100 * e['cost_ratio_vs_always_opus_ci95'][0]:.1f}-{100 * e['cost_ratio_vs_always_opus_ci95'][1]:.1f}) | "
                 f"{100 * e['pass_diff_vs_always_opus']:+.1f} pts ({100 * e['pass_diff_vs_always_opus_ci95'][0]:+.1f} to {100 * e['pass_diff_vs_always_opus_ci95'][1]:+.1f}) |")
    return "\n".join(L) + "\n"


def table_md(out, basis):
    b = out["bases"][basis]
    L = ["| Arm | Passed | Pass rate (Wilson 95%) | Total cost | vs always-best | Pass diff vs best (95% CI) |", "|---|---|---|---|---|---|"]
    names = [f"always-{m}" for m in MODELS] + ["always-best", "always-cheapest", "oracle"] + [f"router-s1-t{t}" for t in range(11)] + [f"router-prior-t{t}" for t in range(11)]
    for nm in names:
        e = b["arms"][nm]
        L.append(f"| {nm} | {e['passed']}/{e['n']} | {100 * e['pass_rate']:.1f}% ({100 * e['pass_rate_wilson95'][0]:.0f}-{100 * e['pass_rate_wilson95'][1]:.0f}) | "
                 f"${e['total_cost_usd']:.5f} | {100 * e['cost_ratio_vs_always_best']:.1f}% | {100 * e['pass_diff_vs_always_best']:+.1f} pts ({100 * e['pass_diff_ci95'][0]:+.1f} to {100 * e['pass_diff_ci95'][1]:+.1f}) |")
    return "\n".join(L) + "\n"


def render():
    out, ids, res, plan, _ = analyse()
    return {"analysis.json": json.dumps(out, indent=1, sort_keys=True) + "\n", "router-picks.csv": picks_csv(out, ids, res, plan),
            "summary.md": summary_md(out)}


if __name__ == "__main__":
    files = render()
    if "--check" in sys.argv:
        bad = [k for k, v in files.items() if (HERE / k).read_text() != v]
        print("MISMATCH: " + ", ".join(bad) if bad else "OK: analysis.json, router-picks.csv, summary.md reproduced byte-identically")
        sys.exit(1 if bad else 0)
    for k, v in files.items():
        (HERE / k).write_text(v)
        print("wrote", k, hashlib.sha256(v.encode()).hexdigest()[:16])
