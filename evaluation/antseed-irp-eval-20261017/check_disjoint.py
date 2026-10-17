#!/usr/bin/env python3
"""Check by content that tasks-public.jsonl shares no item with the tasks that calibrated the success model.

Compares normalised 60-character prompt probes against: experiments/tasks*.py source text, the paired-study
tasks (evaluation/paired-router-ab-20260929/tasks-public.jsonl), GSM8K test rows 0-199, MATH-500, AIME 2025 and
the HumanEval+ ids reachable by the earlier seeded build. Needs network (HF). usage: check_disjoint.py [tasks.jsonl]
"""
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

here = Path(__file__).resolve().parent
repo = here.parents[1]
norm = lambda s: re.sub(r"\W+", "", s.lower())
tasks = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else here / "tasks-public.jsonl")]


def probes(t):
    p = t["prompt"]
    if t["grader"] == "code":
        p = re.sub(r"^.*?\n\n", "", p, count=1) if t["source"].startswith("HumanEval") else p
    p = norm(p.split("\n\nWork it out")[0].split("\n\nThink it through")[0])
    return {p[:60], p[-60:]}


corpus = norm("".join(f.read_text() for f in (repo / "experiments").glob("tasks*.py")))
corpus += norm((repo / "evaluation/paired-router-ab-20260929/tasks-public.jsonl").read_text())
d = snapshot_download("openai/gsm8k", repo_type="dataset", allow_patterns=["main/test-*.parquet"])
corpus += norm("".join(pd.read_parquet(glob.glob(f"{d}/main/test-*.parquet")[0]).question[:200]))
d = snapshot_download("HuggingFaceH4/MATH-500", repo_type="dataset")
corpus += norm("".join(pd.concat([pd.read_json(f, lines=True) for f in glob.glob(f"{d}/**/*.jsonl", recursive=True)]).problem))
d = snapshot_download("math-ai/aime25", repo_type="dataset")
for f in glob.glob(f"{d}/**/*.jsonl", recursive=True) + glob.glob(f"{d}/**/*.parquet", recursive=True):
    df = pd.read_json(f, lines=True) if f.endswith("jsonl") else pd.read_parquet(f)
    corpus += norm("".join(df.problem))
hits = [t["id"] for t in tasks if any(p in corpus for p in probes(t))]
print("tasks:", len(tasks), "overlapping:", hits)
sys.exit(1 if hits else 0)
