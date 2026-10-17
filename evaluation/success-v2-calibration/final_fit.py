"""Fit v2 on all 720 outcomes, choose measured_prior_n on all data, write auto_router/data/success_v2.json."""
import json, sys, hashlib
import numpy as np
sys.path.insert(0, "/home/flori/jobs/antseed-router-v2-comarketing-20261017/workers/calib/lib/experiments")
import fit_success_v2 as F
ds, out = sys.argv[1:3]
D = F.load(ds); th = F.fit(D)
p = F.predict(th, D["cap"], D["d"], D["u"], D["group"]); y = D["y"]; m = D["measured"]; has = ~np.isnan(m)
def ll(q): q = np.clip(q, 1e-6, 1-1e-6); return float(-np.mean(y*np.log(q)+(1-y)*np.log(1-q)))
grid = {n: round(ll(np.where(has, (n*p+6*np.nan_to_num(m))/(n+6), p)), 5) for n in (3., 6., 12., 24., 48., 1e9)}
n_best = min((3., 6., 12., 24., 48.), key=lambda n: grid[n])
P = F.to_params(th); P["measured_prior_n"] = n_best; P["measured_n"] = 6.0
doc = {"version": "success-v2", "params": P, "theta": th,
       "fitted_on": {"dataset": "evaluation/antseed-irp-eval-20261017/calib/dataset.csv", "dataset_sha256": hashlib.sha256(open(ds,"rb").read()).hexdigest(),
                     "n": int(len(y)), "waves": [1, 2], "in_sample_logloss_curve": round(ll(p), 5), "measured_prior_n_grid_logloss": {str(k): v for k, v in grid.items()},
                     "note": "720 graded (task, model) outcomes of waves 1+2; see evaluation/antseed-irp-eval-20261017/calib/README.md"}}
json.dump(doc, open(out, "w"), indent=1, sort_keys=True); print(json.dumps(P, sort_keys=True)); print(grid)
