#!/usr/bin/env python3
"""Wave 2 manifest. `make_manifest.py frozen REPO_COMMIT DEMO_COMMIT CREDITS_REMAINING TASKS_FULL` before the first model call,
`make_manifest.py final` after the run (adds output hashes; frozen inputs must not change unless declared in AMENDED)."""
import datetime
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = ["tasks-public.jsonl", "prices.json", "router-plan.csv", "router-catalog.json", "run-order.json", "analysis-plan.md", "build_tasks.py",
          "build_prices.py", "router_plan.py", "run.py", "grade_w2.py", "validate_graders.py", "reference-solutions.json", "reproduce.py",
          "make_manifest.py", "check_disjoint.py"]
AMENDED = {}
OUTPUTS = ["results.csv", "router-picks.csv", "analysis.json", "summary.md", "pooled-analysis.json", "pooled-summary.md", "chart-wave2.png", "chart-pooled.png"]
sha = lambda n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()

if __name__ == "__main__":
    path = HERE / "manifest.json"
    if sys.argv[1] == "frozen":
        full = Path(sys.argv[5])
        m = {"study": "antseed-irp-eval-20261017/wave2", "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             "sources": {"auto-model-router_commit": sys.argv[2], "auto-router-demo_commit": sys.argv[3],
                         "wave1": "../manifest.json (same models, router build, AntSeed snapshot)"},
             "openrouter_credits_before": {"total_credits": 578.9112268, "total_usage": float(sys.argv[4])},
             "task_count": 60, "tasks_full_sha256 (not committed; rebuilt by build_tasks.py)": hashlib.sha256(full.read_bytes()).hexdigest(),
             "models": list(json.load(open(HERE / "prices.json"))["models"]), "files_sha256": {n: sha(n) for n in FROZEN}}
    else:
        m = json.loads(path.read_text())
        amended = {}
        for n, h in m["files_sha256"].items():
            if sha(n) != h:
                assert n in AMENDED, f"frozen file changed after registration: {n}"
                amended[n] = {"registered_sha256": h, "final_sha256": sha(n), "why": AMENDED[n]}
        if amended:
            m["amendments"] = amended
        m["outputs_sha256"] = {n: sha(n) for n in OUTPUTS if (HERE / n).exists()}
        m["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
