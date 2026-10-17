"""Build the calibration dataset: 120 tasks (waves 1+2) x 6 models with the router's own s1 classification
(category, router-scaled difficulty) and each model's capability for that category as the router computes it."""
import csv, json, sys
from pathlib import Path
W = Path("/home/flori/jobs/antseed-router-v2-comarketing-20261017/workers")
AMR = W / "eval/work/amr"
EV = AMR / "evaluation/antseed-irp-eval-20261017"
sys.path.insert(0, str(AMR)); sys.path.insert(0, str(EV))
import router_plan as w1
from auto_router import s1_router
from types import SimpleNamespace
prices = json.load(open(EV / "prices.json"))["models"]
config, router, models, entries = w1.build(prices)
s1 = w1.Memo(s1_router.LocalS1Classifier(s1_router.REPO, 2))
tasks = {}
for wave, p in ((1, EV / "tasks-public.jsonl"), (2, W / "eval2/work/tasks-full.jsonl")):
    for l in open(p):
        t = json.loads(l); t["wave"] = wave; tasks[t["id"]] = t
res = {}
for wave, p in ((1, EV / "results.csv"), (2, EV / "wave2/results.csv")):
    for r in csv.DictReader(open(p)):
        res[(r["task"], r["model"])] = r
measured = {(a, b, c): p for a, b, c, p in json.load(open(EV / 'success.measured.json'))}
rows = []
for tid, t in tasks.items():
    msgs = [{"role": "user", "content": t["prompt"]}]
    cls = s1(t["prompt"], router._summary(msgs, None))
    req = router._turn_request(cls, 500, 16000, 0.0, False, msgs)
    for m in models:
        name = w1.DEMO_NAMES.get(m.name, m.name)
        r = res[(tid, name)]
        sm = router.success
        cap = m.cap(req.category, benchmaxxing_weight=sm.benchmaxxing_weight, evidence_discount=sm.evidence_discount)
        rows.append(dict(wave=t["wave"], task=tid, src_category=t["category"], src_difficulty=t["difficulty"], s1_category=cls.category,
                         category=req.category, difficulty=round(req.difficulty, 5), s1_difficulty=round(cls.difficulty, 5),
                         diff_conf=round(cls.difficulty_confidence, 4), stakes=round(cls.stakes, 3), model=name, cap=round(cap, 4),
                         v1_p=round(sm.p(m, req.category, req.difficulty), 5), passed=int(r["passed"]),
                         finish=r["finish_reason"], mk=m.measurement_key, task_tokens=round(m.task_tokens or 0, 1),
                         max_tokens=8000 if t["wave"] == 1 else 16000, max_output=m.max_output_tokens,
                         measured_p=measured.get((m.measurement_key, req.category, sm.bucket(req.difficulty)), "")))
with open(Path(__file__).parent.parent / "data/dataset.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), router.jev_offset if hasattr(router, "jev_offset") else None, getattr(router, "jev_scale", None))
