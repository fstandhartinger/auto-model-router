#!/usr/bin/env python3
"""Wave 2: recompute every analysis file from the published CSVs. No API calls, no model.

  python reproduce.py            # writes analysis.json, router-picks.csv, summary.md, pooled-analysis.json, pooled-summary.md
  python reproduce.py --check    # recomputes into memory and fails unless the files on disk are byte-identical

Inputs: tasks-public.jsonl (id, category, difficulty), results.csv (every model's graded answer and its cost on three price
bases), router-plan.csv (the router's pick per setting, recorded before any answer existed), prices.json, and for the
post-hoc pooled view ../tasks-public.jsonl, ../results.csv, ../router-plan.csv (wave 1). The analysis plan is frozen in
analysis-plan.md / manifest.json before the first model call.
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
W1 = HERE.parent
SEED = 20261018
DRAWS = 10_000
BASES = ("median", "cheapest", "list")          # median seller = primary basis for headlines
PRIMARY = "median"
COSTCOL = {"median": "cost_antseed_median", "cheapest": "cost_antseed", "list": "cost_openrouter_list"}
W1_COSTCOL = {"median": "cost_antseed_median", "cheapest": "cost_antseed"}
MODELS = ["claude-opus-5", "gpt-5.6-sol", "gpt-5.6-luna", "kimi-k3", "glm-5.3-flash", "deepseek-v4.1-flash"]
EQ_LOWER, EQ_POINT = -0.05, -0.02               # registered "equal quality": CI lower bound >= -5 points AND point estimate >= -2 points


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 6), round(c + h, 6)]


def r6(x):
    return round(float(x), 9)


def load(d, costcol, fixed_ids=None):
    tasks = [json.loads(l) for l in open(d / "tasks-public.jsonl")]
    res = {}
    for r in csv.DictReader(open(d / "results.csv")):
        res[(r["task"], r["model"])] = {"passed": int(r["passed"]), "finish": r["finish_reason"],
                                        **{b: float(r[c]) for b, c in costcol.items()}}
    plan = {}
    for r in csv.DictReader(open(d / "router-plan.csv")):
        plan[(r["task"], r["classifier"], r["price_basis"], int(r["tradeoff"]))] = r["pick"]
    complete = [t for t in tasks if all((t["id"], m) in res for m in MODELS)]
    return complete, res, plan, len(tasks)


def analyse(tasks, res, plan, prices, bases, primary):
    ids = [t["id"] for t in tasks]
    n = len(ids)
    cat = [t["category"] for t in tasks]
    dif = [t["difficulty"] for t in tasks]
    rng = random.Random(SEED)
    draws = np.array([[rng.randrange(n) for _ in range(n)] for _ in range(DRAWS)], dtype=np.int64)
    pass_by = {m: sum(res[(i, m)]["passed"] for i in ids) for m in MODELS}
    out = {"n_tasks": n, "bootstrap": {"draws": DRAWS, "seed": SEED, "method": "paired, resampling tasks, percentile 95% CI"},
           "models": MODELS, "primary_basis": primary, "bases": {}, "single_model_pass_by_category": {
               m: {g: sum(res[(i, m)]["passed"] for i, c_ in zip(ids, cat) if c_ == g) for g in sorted(set(cat))} for m in MODELS},
           "single_model_pass_by_difficulty": {m: {g: sum(res[(i, m)]["passed"] for i, c_ in zip(ids, dif) if c_ == g) for g in sorted(set(dif))} for m in MODELS},
           "finish_reasons": {m: {f: sum(1 for i in ids if res[(i, m)]["finish"] == f) for f in sorted({res[(j, m)]["finish"] for j in ids})} for m in MODELS},
           "tasks_failed_by_all_models": [i for i in ids if not any(res[(i, m)]["passed"] for m in MODELS)],
           "tasks_passed_by_all_models": sum(1 for i in ids if all(res[(i, m)]["passed"] for m in MODELS)),
           "tasks_per_difficulty": {g: dif.count(g) for g in sorted(set(dif))}}
    for basis in bases:
        total0 = {m: sum(res[(i, m)][basis] for i in ids) for m in MODELS}
        best = max(MODELS, key=lambda m: (pass_by[m], -total0[m], m))          # ties: cheaper total on THIS basis
        cheapest_model = min(MODELS, key=lambda m: (prices[m][basis]["input"] + prices[m][basis]["output"], m))

        def vec(picks):
            return (np.array([res[(i, m)]["passed"] for i, m in zip(ids, picks)], dtype=np.int64),
                    np.array([res[(i, m)][basis] for i, m in zip(ids, picks)], dtype=float))
        arms = {f"always-{m}": {"kind": "single", "picks": [m] * n} for m in MODELS}
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
        refs = {"best": vec(arms["always-best"]["picks"]), "opus": vec(arms["always-claude-opus-5"]["picks"])}
        table = {}
        for name, a in arms.items():
            p, c = vec(a["picks"])
            e = {"kind": a["kind"], "passed": int(p.sum()), "n": n, "pass_rate": r6(p.mean()), "pass_rate_wilson95": wilson(int(p.sum()), n),
                 "total_cost_usd": r6(c.sum()), "mean_cost_per_task_usd": r6(c.mean())}
            for rk, (rp, rc) in refs.items():
                dp = p[draws].mean(axis=1) - rp[draws].mean(axis=1)
                cr = c[draws].sum(axis=1) / rc[draws].sum(axis=1)
                e[f"cost_ratio_vs_always_{rk}"] = r6(c.sum() / rc.sum())
                e[f"cost_ratio_vs_always_{rk}_ci95"] = [r6(np.percentile(cr, 2.5)), r6(np.percentile(cr, 97.5))]
                e[f"pass_diff_vs_always_{rk}"] = r6(p.mean() - rp.mean())
                e[f"pass_diff_vs_always_{rk}_ci95"] = [r6(np.percentile(dp, 2.5)), r6(np.percentile(dp, 97.5))]
            if a["kind"] == "router":
                e["classifier"], e["tradeoff"] = a["classifier"], a["tradeoff"]
                e["picks_by_model"] = {m: a["picks"].count(m) for m in MODELS if a["picks"].count(m)}
            table[name] = e
        head = {}
        for rk in ("best", "opus"):
            for clf in ("s1", "prior"):
                sets = [table[f"router-{clf}-t{t}"] for t in range(11)]
                ok = [e for e in sets if e[f"pass_diff_vs_always_{rk}_ci95"][0] >= EQ_LOWER and e[f"pass_diff_vs_always_{rk}"] >= EQ_POINT]
                key = f"{clf}/equal_quality/vs_always_{rk}"
                if ok:
                    e = min(ok, key=lambda e: (e["total_cost_usd"], e["tradeoff"]))
                    head[key] = {"tradeoff": e["tradeoff"], "pass_rate": e["pass_rate"], "pass_diff": e[f"pass_diff_vs_always_{rk}"],
                                 "pass_diff_ci95": e[f"pass_diff_vs_always_{rk}_ci95"], "cost_pct": r6(100 * e[f"cost_ratio_vs_always_{rk}"]),
                                 "cost_pct_ci95": [r6(100 * x) for x in e[f"cost_ratio_vs_always_{rk}_ci95"]], "picks_by_model": e["picks_by_model"]}
                else:
                    head[key] = None
        svp = {}
        for t in range(11):
            ps, cs = vec(arms[f"router-s1-t{t}"]["picks"])
            pp, cp = vec(arms[f"router-prior-t{t}"]["picks"])
            d = ps[draws].mean(axis=1) - pp[draws].mean(axis=1)
            cr = cs[draws].sum(axis=1) / cp[draws].sum(axis=1)
            svp[str(t)] = {"pass_diff_s1_minus_prior": r6(ps.mean() - pp.mean()), "ci95": [r6(np.percentile(d, 2.5)), r6(np.percentile(d, 97.5))],
                           "cost_ratio_s1_over_prior": r6(cs.sum() / cp.sum()), "cost_ratio_ci95": [r6(np.percentile(cr, 2.5)), r6(np.percentile(cr, 97.5))]}

        def split(key):
            res_ = {}
            for g in sorted(set(key)):
                idx = [k for k, x in enumerate(key) if x == g]
                res_[g] = {"n": len(idx)}
                for name, a in arms.items():
                    p, c = vec(a["picks"])
                    res_[g][name] = {"pass_rate": r6(p[idx].mean()), "cost_usd": r6(c[idx].sum())}
            return res_
        out["bases"][basis] = {"always_best_model": best, "always_cheapest_model": cheapest_model, "arms": table, "headline": head,
                               "s1_vs_prior": svp, "by_category": split(cat), "by_difficulty": split(dif)}
    return out, ids


def picks_csv(out, ids, res, plan, bases):
    lines = ["task,arm,tradeoff,picked_model,passed,cost_usd,price_basis"]
    for basis in bases:
        for clf in ("s1", "prior"):
            for t in range(11):
                for i in ids:
                    m = plan[(i, clf, basis, t)]
                    lines.append(f"{i},router-{clf},{t},{m},{res[(i, m)]['passed']},{res[(i, m)][basis]:.9f},{basis}")
    return "\n".join(lines) + "\n"


NAMES = lambda: [f"always-{m}" for m in MODELS] + ["always-best", "always-cheapest", "oracle"] + [f"router-s1-t{t}" for t in range(11)] + [f"router-prior-t{t}" for t in range(11)]
PRETTY = {"median": "median seller (primary)", "cheapest": "cheapest seller", "list": "OpenRouter list price"}


def arm_table(out, basis):
    b = out["bases"][basis]
    L = ["| Arm | Passed | Pass rate (Wilson 95%) | Total cost | vs always-best (95% CI) | vs always-Opus (95% CI) | Pass diff vs best (95% CI) | Pass diff vs Opus (95% CI) |",
         "|---|---|---|---|---|---|---|---|"]
    pct = lambda e, k: f"{100 * e['cost_ratio_vs_always_' + k]:.1f}% ({100 * e['cost_ratio_vs_always_' + k + '_ci95'][0]:.1f}-{100 * e['cost_ratio_vs_always_' + k + '_ci95'][1]:.1f})"
    dif = lambda e, k: f"{100 * e['pass_diff_vs_always_' + k]:+.1f} ({100 * e['pass_diff_vs_always_' + k + '_ci95'][0]:+.1f} to {100 * e['pass_diff_vs_always_' + k + '_ci95'][1]:+.1f})"
    for nm in NAMES():
        e = b["arms"][nm]
        L.append(f"| {nm} | {e['passed']}/{e['n']} | {100 * e['pass_rate']:.1f}% ({100 * e['pass_rate_wilson95'][0]:.0f}-{100 * e['pass_rate_wilson95'][1]:.0f}) | "
                 f"${e['total_cost_usd']:.4f} | {pct(e, 'best')} | {pct(e, 'opus')} | {dif(e, 'best')} | {dif(e, 'opus')} |")
    return "\n".join(L) + "\n"


def headline_md(out):
    L = ["| Price basis | Reference | Classifier | Cheapest setting meeting 'equal quality' | Pass rate | Pass diff (95% CI) | Cost vs reference (95% CI) | Picks |", "|---|---|---|---|---|---|---|---|"]
    for basis in out["bases"]:
        b = out["bases"][basis]
        for rk, rname in (("best", f"always-best ({b['always_best_model']})"), ("opus", "always-Opus")):
            for clf in ("s1", "prior"):
                h = b["headline"][f"{clf}/equal_quality/vs_always_{rk}"]
                if h is None:
                    L.append(f"| {PRETTY[basis]} | {rname} | {clf} | none | - | - | - | - |")
                else:
                    L.append(f"| {PRETTY[basis]} | {rname} | {clf} | t{h['tradeoff']} | {100 * h['pass_rate']:.1f}% | {100 * h['pass_diff']:+.1f} ({100 * h['pass_diff_ci95'][0]:+.1f} to {100 * h['pass_diff_ci95'][1]:+.1f}) | "
                             f"{h['cost_pct']:.1f}% ({h['cost_pct_ci95'][0]:.1f}-{h['cost_pct_ci95'][1]:.1f}) | {h['picks_by_model']} |")
    return "\n".join(L) + "\n"


def svp_md(out, basis):
    s = out["bases"][basis]["s1_vs_prior"]
    L = ["| Tradeoff | s1 minus prior, pass diff pts (95% CI) | s1 cost / prior cost (95% CI) |", "|---|---|---|"]
    for t in range(11):
        e = s[str(t)]
        L.append(f"| {t} | {100 * e['pass_diff_s1_minus_prior']:+.1f} ({100 * e['ci95'][0]:+.1f} to {100 * e['ci95'][1]:+.1f}) | {e['cost_ratio_s1_over_prior']:.2f} ({e['cost_ratio_ci95'][0]:.2f}-{e['cost_ratio_ci95'][1]:.2f}) |")
    return "\n".join(L) + "\n"


def single_md(out):
    cats = sorted({c for m in MODELS for c in out["single_model_pass_by_category"][m]})
    dfs = sorted({c for m in MODELS for c in out["single_model_pass_by_difficulty"][m]})
    n = out["n_tasks"]
    L = ["| Model | Overall | " + " | ".join(cats) + " | " + " | ".join(dfs) + " | finish_reason |", "|---|---|" + "---|" * (len(cats) + len(dfs) + 1)]
    for m in MODELS:
        tot = sum(out["single_model_pass_by_category"][m].values())
        L.append(f"| {m} | {tot}/{n} ({100 * tot / n:.0f}%) | " + " | ".join(str(out["single_model_pass_by_category"][m].get(c, 0)) for c in cats) + " | " +
                 " | ".join(str(out["single_model_pass_by_difficulty"][m].get(c, 0)) for c in dfs) + " | " +
                 ", ".join(f"{k} {v}" for k, v in out["finish_reasons"][m].items()) + " |")
    return "\n".join(L) + "\n"


def summary_md(out, title, n_total):
    parts = [f"## {title}\n\nTasks analysed: {out['n_tasks']} of {n_total}. Easy/medium/hard: {out['tasks_per_difficulty']}. "
             f"Passed by all six models: {out['tasks_passed_by_all_models']}; passed by none: {len(out['tasks_failed_by_all_models'])} "
             f"({', '.join(out['tasks_failed_by_all_models']) or '-'}).\n",
             "### Headline: cheapest router setting meeting 'equal quality' (CI lower bound >= -5 points and point estimate >= -2 points)\n\n" + headline_md(out),
             "### Single models: tasks passed by category (count) and difficulty (count), finish reasons\n\n" + single_md(out)]
    for basis in out["bases"]:
        parts.append(f"### All arms, {PRETTY[basis]} prices (always-best = {out['bases'][basis]['always_best_model']})\n\n" + arm_table(out, basis))
        parts.append(f"### s1 vs prior-only, {PRETTY[basis]} prices\n\n" + svp_md(out, basis))
    return "\n".join(parts)


def render():
    prices = json.load(open(HERE / "prices.json"))["models"]
    tasks, res, plan, n_all = load(HERE, COSTCOL)
    out, ids = analyse(tasks, res, plan, prices, BASES, PRIMARY)
    files = {"analysis.json": json.dumps(out, indent=1, sort_keys=True) + "\n", "router-picks.csv": picks_csv(out, ids, res, plan, BASES),
             "summary.md": summary_md(out, "Wave 2 (60 harder tasks)", n_all)}
    # post-hoc pooled view: wave 1 (60 tasks, max_tokens 8000) + wave 2 (60 tasks, max_tokens 16000)
    t1, r1, p1, n1 = load(W1, W1_COSTCOL)
    tasks_p = t1 + tasks
    res_p = {**r1, **res}
    plan_p = {**p1, **plan}
    pb = ("median", "cheapest")
    pout, _ = analyse(tasks_p, res_p, plan_p, prices, pb, PRIMARY)
    files["pooled-analysis.json"] = json.dumps(pout, indent=1, sort_keys=True) + "\n"
    files["pooled-summary.md"] = summary_md(pout, "Pooled wave 1 + wave 2 (POST-HOC pooling, not a registered analysis)", n1 + n_all)
    return files


if __name__ == "__main__":
    files = render()
    if "--check" in sys.argv:
        bad = [k for k, v in files.items() if not (HERE / k).exists() or (HERE / k).read_text() != v]
        print("MISMATCH: " + ", ".join(bad) if bad else "OK: " + ", ".join(files) + " reproduced byte-identically")
        sys.exit(1 if bad else 0)
    for k, v in files.items():
        (HERE / k).write_text(v)
        print("wrote", k, hashlib.sha256(v.encode()).hexdigest()[:16])
