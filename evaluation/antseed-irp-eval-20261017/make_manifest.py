#!/usr/bin/env python3
"""Write manifest.json: SHA-256 of every frozen input. `make_manifest.py frozen` before the run, `... final` after it
(adds the hashes of results/outputs and the run record; the frozen section must not change)."""
import datetime
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = ["tasks-public.jsonl", "prices.json", "router-plan.csv", "router-catalog.json", "success.measured.json",
          "demo-catalog.json", "analysis-plan.md", "build_tasks.py", "build_prices.py", "router_plan.py", "run.py",
          "grade.py", "validate_graders.py", "reproduce.py", "check_disjoint.py"]
OUTPUTS = ["results.csv", "router-picks.csv", "analysis.json", "summary.md", "chart-cost-vs-quality.png"]
sha = lambda n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()

if __name__ == "__main__":
    mode = sys.argv[1]
    path = HERE / "manifest.json"
    m = json.loads(path.read_text()) if path.exists() else {}
    if mode == "frozen":
        m = {"study": "antseed-irp-eval-20261017", "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             "sources": {"auto-model-router_commit": sys.argv[2], "auto-router-demo_commit": sys.argv[3],
                         "s1_model": "decisionmodels/s1-llm-auto-router (revision and file digests in router-catalog.json)",
                         "antseed_snapshot": "evidence/antseed-models-text-20261017.json taken 2026-10-17 10:27Z, sha256 in prices.json"},
             "openrouter_credits_before_usd_remaining": float(sys.argv[4]),
             "task_count": 60, "models": list(json.load(open(HERE / "prices.json"))["models"]),
             "files_sha256": {n: sha(n) for n in FROZEN}}
    else:
        for n, h in m["files_sha256"].items():
            assert sha(n) == h, f"frozen file changed after registration: {n}"
        m["outputs_sha256"] = {n: sha(n) for n in OUTPUTS if (HERE / n).exists()}
        m["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
