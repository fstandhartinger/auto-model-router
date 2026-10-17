"""cv.json -> cv-summary.md (fold-level and pooled tables)."""
import json, sys
import numpy as np
r = json.load(open(sys.argv[1])); out = []
KS = ["const", "v1", "v1-curve", "v2-nocap", "v2-nolength", "v2", "v2+measured"]
for kind, title in (("task10", "10-fold, grouped by task"), ("wave", "leave-one-wave-out (2 folds)"), ("model", "leave-one-model-out (6 folds; the held-out model is unseen in training)")):
    f = r["folds"][kind]
    out += [f"### {title}", "", "Log-loss (lower is better), weighted by fold size; in brackets the best and worst fold.", "",
            "| model | log-loss | Brier | worst fold | best fold |", "|---|---|---|---|---|"]
    for k in KS:
        ll = [v["metrics"][k]["logloss"] for v in f.values()]; br = [v["metrics"][k]["brier"] for v in f.values()]; w = [v["n_test"] for v in f.values()]
        out.append(f"| {k} | {np.average(ll, weights=w):.4f} | {np.average(br, weights=w):.4f} | {max(ll):.3f} | {min(ll):.3f} |")
    out += ["", "Fold-level log-loss:", "", "| fold | n | " + " | ".join(KS) + " |", "|---|---|" + "---|" * len(KS)]
    for n, v in f.items():
        out.append(f"| {n} | {v['n_test']} | " + " | ".join(f"{v['metrics'][k]['logloss']:.3f}" for k in KS) + " |")
    out.append("")
out += ["### Offline router replay on held-out tasks (the picks of irp.rank() made by a model trained without that task; median AntSeed seller prices; outcomes as recorded)", ""]
for kind, title in (("task10", "10-fold grouped by task"), ("wave", "leave-one-wave-out")):
    p = r["pooled"][kind]; n = p["n_tasks"]
    a = p["always"]; opus = a["claude-opus-5"]
    out += [f"**{title}, {n} tasks** (cost shown as USD total and % of always-Opus)", "", "| arm | passed | cost USD | % of Opus cost |", "|---|---|---|---|"]
    for k in ("claude-opus-5", "gpt-5.6-sol", "gpt-5.6-luna", "kimi-k3", "deepseek-v4.1-flash", "glm-5.3-flash"):
        out.append(f"| always-{k} | {a[k][0]} | {a[k][1]:.3f} | {100 * a[k][1] / opus[1]:.1f} |")
    out.append(f"| oracle (cheapest passing model per task) | {p['oracle'][0]} | {p['oracle'][1]:.3f} | {100 * p['oracle'][1] / opus[1]:.1f} |")
    for lab in ("v1", "v2", "v2+measured"):
        for t in ("0", "1", "2", "3", "5", "7", "10"):
            v = p[lab][t]; out.append(f"| router {lab} t{t} | {v[0]} | {v[1]:.3f} | {100 * v[1] / opus[1]:.1f} |")
    out.append("")
    out += ["Per fold (passed / cost USD, t0 and t2):", "", "| fold | tasks | v1 t0 | v2 t0 | v2+m t0 | v1 t2 | v2 t2 | v2+m t2 | always-Opus | always-Sol | always-Luna |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for nme, v in r["folds"][kind].items():
        rp = v["replay"]; al = v["always"]
        c = lambda lab, t: f"{rp[lab][t][0]} / {rp[lab][t][1]:.3f}"
        out.append(f"| {nme} | {v['n_tasks']} | {c('v1','0')} | {c('v2','0')} | {c('v2+measured','0')} | {c('v1','2')} | {c('v2','2')} | {c('v2+measured','2')} | {al['claude-opus-5'][0]} / {al['claude-opus-5'][1]:.3f} | {al['gpt-5.6-sol'][0]} / {al['gpt-5.6-sol'][1]:.3f} | {al['gpt-5.6-luna'][0]} / {al['gpt-5.6-luna'][1]:.3f} |")
    out.append("")
open(sys.argv[2], "w").write("\n".join(out))
