"""Cross-validation of the success models + offline router replay on held-out folds.

Folds: 10-fold grouped by TASK (seed 20261018, both waves mixed), leave-one-wave-out (2), leave-one-model-out (6, a model never seen in training).
Models: constant base rate, v1 as deployed (curve + measured override), v1 curve only, v2 variants (refit; refit without length term; with measured shrinkage).
Replay: for every held-out task the library's irp.rank() (the same call whichmodel makes) picks among the six models with the fold's model
trained WITHOUT that task; the pick's recorded outcome (pass, cost at median AntSeed seller prices) is used. Nothing here calls a model.
usage: cv.py OUT.json
"""
import csv, json, math, random, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

W = Path("/home/flori/jobs/antseed-router-v2-comarketing-20261017/workers")
AMR = W / "calib/lib"
EVAL_OLD = W / "eval/work/amr/evaluation/antseed-irp-eval-20261017"
sys.path.insert(0, str(AMR)); sys.path.insert(0, str(AMR / "experiments")); sys.path.insert(0, str(EVAL_OLD))
import fit_success_v2 as F
import router_plan as w1
from auto_router import s1_router
from auto_router.economics import SuccessModel
from auto_router.irp import rank

DATA = F.load(str(W / "calib/data/dataset.csv"))
N = len(DATA["y"])
TASKS = sorted(set(DATA["task"]))
WAVE_OF = {t: int(DATA["wave"][i]) for i, t in zip(range(N), DATA["task"])}
prices = json.load(open(EVAL_OLD / "prices.json"))["models"]
config, router, models, entries = w1.build(prices)
irp_id = {w1.DEMO_NAMES.get(e["name"], e["name"]): e["irp_model_id"] for e in entries}
MODEL_OF = {w1.DEMO_NAMES.get(m.name, m.name): m for m in models}
MEASURED = {(m, c, b): p for m, c, b, p in json.load(open(EVAL_OLD / "success.measured.json"))}
s1 = w1.Memo(s1_router.LocalS1Classifier(s1_router.REPO, 2))
PROMPT = {}
for wave, p in ((1, EVAL_OLD / "tasks-public.jsonl"), (2, W / "eval2/work/tasks-full.jsonl")):
    for l in open(p):
        t = json.loads(l); PROMPT[t["id"]] = t["prompt"]
OUT = {}
for wave, p in ((1, EVAL_OLD / "results.csv"), (2, EVAL_OLD / "wave2/results.csv")):
    for r in csv.DictReader(open(p)):
        OUT[(r["task"], r["model"])] = (int(r["passed"]), float(r["cost_antseed_median"]))

def logloss(p, y): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def brier(p, y): return float(np.mean((p - y) ** 2))

def v1_curve(cap, d):
    z = np.clip((cap - (20 + 50 * d)) / 6, -50, 50)
    return np.clip(1 / (1 + np.exp(-z)), 0.02, 0.98)

def shrink(p_curve, meas, n_prior, n_obs=6.0):
    has = ~np.isnan(meas)
    return np.where(has, (n_prior * p_curve + n_obs * np.nan_to_num(meas)) / (n_prior + n_obs), p_curve)

def success_obj(th, n_prior=None):
    P = F.to_params(th)
    return SuccessModel(offset=P["offset"], slope=P["slope"], scale=P["scale"], floor=P["floor"], ceiling=P["ceiling"],
                        per_category={k: tuple(v) for k, v in P["per_category"].items()}, length_penalty=P["length_penalty"], smooth_ceiling=True,
                        measured=dict(MEASURED) if n_prior is not None else {}, measured_prior_n=n_prior, measured_n=6.0)

VARIANTS = {  # name -> (fit kwargs or None, measured prior_n or None)
    "v2": (dict(), None),
    "v2-nolength": (dict(use_length=False), None),
    "v2-nocap": (dict(use_cap=False), None),
}
PRIOR_GRID = (3.0, 6.0, 12.0, 24.0, 48.0)

def folds():
    rng = random.Random(20261018); order = TASKS[:]; rng.shuffle(order)
    out = {"task10": [], "wave": [], "model": []}
    for k in range(10):
        held = set(order[k::10]); out["task10"].append((f"task-fold-{k}", [i for i in range(N) if DATA["task"][i] in held], [i for i in range(N) if DATA["task"][i] not in held]))
    for w in (1, 2):
        out["wave"].append((f"train-wave{3 - w}-test-wave{w}", [i for i in range(N) if DATA["wave"][i] == w], [i for i in range(N) if DATA["wave"][i] != w]))
    for m in sorted(set(DATA["model"])):
        out["model"].append((f"leave-out-{m}", [i for i in range(N) if DATA["model"][i] == m], [i for i in range(N) if DATA["model"][i] != m]))
    return out

def preds(th, idx):
    return F.predict(th, DATA["cap"][idx], DATA["d"][idx], DATA["u"][idx], DATA["group"][idx])

def choose_prior(th, tr):
    p = preds(th, tr); y = DATA["y"][tr]; m = DATA["measured"][tr]
    best = min(PRIOR_GRID, key=lambda n: logloss(shrink(p, m, n), y))
    return best

def metrics_for_fold(name, te, tr):
    y = DATA["y"][te]; res = {}
    res["const"] = (np.full(len(te), DATA["y"][tr].mean()))
    res["v1"] = DATA["v1_p"][te]
    res["v1-curve"] = v1_curve(DATA["cap"][te], DATA["d"][te])
    ths = {}
    for v, (kw, _) in VARIANTS.items():
        th = F.fit(DATA, np.array(tr), **kw); ths[v] = th
        res[v] = preds(th, te)
    n_prior = choose_prior(ths["v2"], tr)
    res["v2+measured"] = shrink(res["v2"], DATA["measured"][te], n_prior)
    return {k: dict(logloss=logloss(p, y), brier=brier(p, y)) for k, p in res.items()}, ths, n_prior

def pick(success, task, tradeoff, max_tokens):
    view = SimpleNamespace(config=config, success=success, classifier=s1, _turn_request=router._turn_request, _summary=router._summary)
    cands = [{"id": k, "model": irp_id[k], "pricing": {"input": prices[k]["median"]["input"], "cache_read": prices[k]["median"]["cache_read"], "output": prices[k]["median"]["output"]}} for k in irp_id]
    body = {"request": {"messages": [{"role": "user", "content": PROMPT[task]}], "max_tokens": max_tokens}, "routing": {"cost_quality_tradeoff": tradeoff, "candidates": cands}}
    return rank(view, body)["ranked"][0]["candidate_id"]

def replay(success_for_task, task_list, tradeoffs=(0, 1, 2, 3, 5, 7, 10)):
    """success_for_task(task) -> SuccessModel trained without it. Returns {tradeoff: (passed, cost)} over task_list."""
    res = {t: [0, 0.0] for t in tradeoffs}
    for task in task_list:
        sm = success_for_task(task); mt = 8000 if WAVE_OF[task] == 1 else 16000
        for t in tradeoffs:
            ok, cost = OUT[(task, pick(sm, task, t, mt))]
            res[t][0] += ok; res[t][1] += cost
    return res

def main(out_path):
    F_ = folds(); report = {"folds": {}, "pooled": {}}
    for kind, lst in F_.items():
        report["folds"][kind] = {}
        for name, te, tr in lst:
            m, ths, n_prior = metrics_for_fold(name, te, tr)
            report["folds"][kind][name] = {"n_test": len(te), "metrics": m, "prior_n_chosen": n_prior}
        # pooled out-of-fold log-loss per method
    # replay: for task folds and wave folds
    models_v1 = SuccessModel(measured=dict(MEASURED))
    for kind in ("task10", "wave"):
        sm_cache = {}
        task_model, task_model_m = {}, {}
        for name, te, tr in F_[kind]:
            th = F.fit(DATA, np.array(tr)); n_prior = choose_prior(th, tr)
            held = sorted(set(DATA["task"][te]))
            for t in held:
                task_model[t] = success_obj(th); task_model_m[t] = success_obj(th, n_prior)
            report["folds"][kind][name]["replay"] = {}
            for label, sm in (("v1", models_v1), ("v2", success_obj(th)), ("v2+measured", success_obj(th, n_prior))):
                r = replay(lambda t, sm=sm: sm, held)
                tot = {m_: (sum(OUT[(t, m_)][0] for t in held), sum(OUT[(t, m_)][1] for t in held)) for m_ in irp_id}
                report["folds"][kind][name]["replay"][label] = {str(t): v for t, v in r.items()}
            report["folds"][kind][name]["always"] = {m_: v for m_, v in tot.items()}
            report["folds"][kind][name]["n_tasks"] = len(held)
        alltasks = sorted(task_model)
        pooled = {}
        for label, store in (("v1", None), ("v2", task_model), ("v2+measured", task_model_m)):
            r = replay((lambda t: models_v1) if store is None else (lambda t, s=store: s[t]), alltasks)
            pooled[label] = {str(t): v for t, v in r.items()}
        pooled["always"] = {m_: (sum(OUT[(t, m_)][0] for t in alltasks), sum(OUT[(t, m_)][1] for t in alltasks)) for m_ in irp_id}
        pooled["oracle"] = [sum(1 if any(OUT[(t, m_)][0] for m_ in irp_id) else 0 for t in alltasks),
                            sum(min([OUT[(t, m_)][1] for m_ in irp_id if OUT[(t, m_)][0]] or [min(OUT[(t, m_)][1] for m_ in irp_id)]) for t in alltasks)]
        pooled["n_tasks"] = len(alltasks)
        report["pooled"][kind] = pooled
    json.dump(report, open(out_path, "w"), indent=1, sort_keys=True)

if __name__ == "__main__":
    main(sys.argv[1])
