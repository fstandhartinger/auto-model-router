"""Fit the v2 success model (auto_router.economics.success_v2) on graded (task, model) outcomes.

    python experiments/fit_success_v2.py DATASET.csv OUT.json [--kappa-min K] [--kappa-prior K0 --kappa-weight W]

DATASET.csv has one row per graded (task, model) with: task, wave, model, category (the router's own, from the s1 classifier),
difficulty (router-scaled s1 difficulty, 0..1), cap (the model's capability for that category as the router computes it),
task_tokens (Benchmark Heaven output tokens per task, 0 = unknown), max_tokens (the budget in the request), max_output (the model's own limit),
passed (0/1). Needs numpy and scipy (experiments only; the library itself has neither as a dependency).

Model: p = floor + (ceiling - floor) * sigmoid(z), z = kappa*(cap - 60)/10 + a_g - g_g*difficulty - lam*clamp(ln(task_tokens/budget)),
the v1 form with (offset, slope, scale) = (-a*s, g*s, s), s = 10/kappa, plus a token-budget term. Groups g: coding, math, other (everything else).
Constraints: kappa >= kappa_min >= 0 (more capability never lowers p), g_g >= 0 (more difficulty never raises p), lam >= 0, ceiling in [0.90, 0.995].
Group parameters are ridge-shrunk toward the 'other' group so thin groups cannot run away.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys

import numpy as np
from scipy.optimize import minimize

GROUPS = ("other", "coding", "math")
CLAMP = (-2.0, 3.0)
FLOOR = 0.02
CEIL_BOUNDS = (0.90, 0.995)
RIDGE = 2.0
LAMBDA_L2 = 0.0


def group_of(category: str) -> int:
    return {"coding": 1, "math": 2}.get(category, 0)


def length_ratio(task_tokens: np.ndarray, max_tokens: np.ndarray, max_output: np.ndarray) -> np.ndarray:
    """clamp(ln(task_tokens / min(max_output, max_tokens))); 0 where task_tokens is unknown (term switched off)."""
    budget = np.minimum(max_output, max_tokens)
    u = np.log(np.maximum(task_tokens, 1.0) / np.maximum(budget, 1.0))
    return np.where(task_tokens > 0, np.clip(u, *CLAMP), 0.0)


def load(path: str) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    f = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    return {
        "task": np.array([r["task"] for r in rows]), "model": np.array([r["model"] for r in rows]),
        "wave": np.array([int(r["wave"]) for r in rows]), "category": np.array([r["category"] for r in rows]),
        "group": np.array([group_of(r["category"]) for r in rows]), "d": f("difficulty"), "cap": f("cap"),
        "u": length_ratio(f("task_tokens"), f("max_tokens"), f("max_output")), "y": np.array([int(r["passed"]) for r in rows]),
        "measured": np.array([float(r["measured_p"]) if r["measured_p"] != "" else np.nan for r in rows]),
        "v1_p": f("v1_p"),
    }


def predict(theta: dict, cap, d, u, group, ceiling=None):
    z = (theta["kappa"] * (cap - 60.0) / 10.0 + np.asarray(theta["a"])[group] - np.asarray(theta["g"])[group] * d
         - theta["lam"] * u)
    c = theta["ceiling"] if ceiling is None else ceiling
    return FLOOR + (c - FLOOR) / (1.0 + np.exp(-np.clip(z, -50, 50)))


def fit(data: dict, idx=None, *, kappa_min: float = 0.0, kappa_prior: float | None = None, kappa_weight: float = 0.0,
        use_length: bool = True, use_cap: bool = True, ridge: float = RIDGE) -> dict:
    idx = np.arange(len(data["y"])) if idx is None else idx
    cap, d, u, grp, y = (data[k][idx] for k in ("cap", "d", "u", "group", "y"))

    def unpack(x):
        return {"a": x[0:3], "g": x[3:6], "kappa": x[6], "lam": x[7], "ceiling": 1 / (1 + math.exp(-x[8]))}

    def obj(x):
        th = unpack(x)
        if not use_cap:
            th["kappa"] = 0.0
        if not use_length:
            th["lam"] = 0.0
        p = np.clip(predict(th, cap, d, u, grp), 1e-6, 1 - 1e-6)
        nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        pen = ridge * (np.sum((th["a"][1:] - th["a"][0]) ** 2) + np.sum((th["g"][1:] - th["g"][0]) ** 2)) + LAMBDA_L2 * th["lam"] ** 2
        if kappa_prior is not None:
            pen += kappa_weight * (th["kappa"] - kappa_prior) ** 2
        return nll + pen

    lo = [-np.inf] * 3 + [0.0] * 3 + [kappa_min, 0.0, math.log(CEIL_BOUNDS[0] / (1 - CEIL_BOUNDS[0]))]
    hi = [np.inf] * 3 + [np.inf] * 3 + [10.0, 5.0, math.log(CEIL_BOUNDS[1] / (1 - CEIL_BOUNDS[1]))]
    best = None
    for k0 in (0.1, 0.5, 1.5):
        x0 = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, max(k0, kappa_min), 0.3, 3.0])
        r = minimize(obj, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)))
        if best is None or r.fun < best.fun:
            best = r
    th = unpack(best.x)
    th["kappa"] = th["kappa"] if use_cap else 0.0
    th["lam"] = th["lam"] if use_length else 0.0
    th["a"], th["g"] = [float(v) for v in th["a"]], [float(v) for v in th["g"]]
    return th


def to_params(th: dict) -> dict:
    """Convert to the (offset, slope, scale) triples SuccessModel uses. kappa = 0 is written as scale 1e6 (flat in capability)."""
    scale = 10.0 / th["kappa"] if th["kappa"] > 1e-9 else 1e6
    def triple(gi):
        # z = (cap - 60)/scale + a - g*d  =  (cap - offset - slope*d)/scale
        return [round(60.0 - th["a"][gi] * scale, 4), round(th["g"][gi] * scale, 4), round(scale, 4)]
    o, c, m = triple(0), triple(1), triple(2)
    return {"offset": o[0], "slope": o[1], "scale": o[2], "floor": FLOOR, "ceiling": round(th["ceiling"], 5), "smooth_ceiling": True,
            "per_category": {"coding": c, "math": m}, "length_penalty": round(th["lam"], 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("out")
    ap.add_argument("--kappa-min", type=float, default=0.0)
    ap.add_argument("--kappa-prior", type=float, default=None)
    ap.add_argument("--kappa-weight", type=float, default=0.0)
    ap.add_argument("--no-length", action="store_true")
    a = ap.parse_args()
    data = load(a.dataset)
    th = fit(data, kappa_min=a.kappa_min, kappa_prior=a.kappa_prior, kappa_weight=a.kappa_weight, use_length=not a.no_length)
    p = predict(th, data["cap"], data["d"], data["u"], data["group"])
    y = data["y"]
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    out = {"version": "success-v2", "params": to_params(th), "theta": th,
           "fitted_on": {"dataset": a.dataset.split("/")[-1], "dataset_sha256": hashlib.sha256(open(a.dataset, "rb").read()).hexdigest(),
                         "n": int(len(y)), "in_sample_logloss": round(ll, 5),
                         "options": {"kappa_min": a.kappa_min, "kappa_prior": a.kappa_prior, "kappa_weight": a.kappa_weight,
                                     "length_term": not a.no_length}}}
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps(out["params"], sort_keys=True), "in-sample logloss", round(ll, 4))


if __name__ == "__main__":
    sys.exit(main())
