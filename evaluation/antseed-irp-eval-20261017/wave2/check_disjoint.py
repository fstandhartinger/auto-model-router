#!/usr/bin/env python3
"""Wave 2 is disjoint, by content, from wave 1 and from the calibration tasks: normalised 60-character probes (start and end of each
prompt) of every wave-2 task are searched in the wave-1 prompts, experiments/tasks*.py, the paired-study tasks, and the HF sets that
wave 1 excluded (GSM8K rows 0-199, MATH-500, AIME 2025). Positive control: a wave-1 prompt must be flagged.
usage: check_disjoint.py TASKS_FULL.jsonl   (needs network / HF cache)"""
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

here = Path(__file__).resolve().parent
repo = here.parents[2]
norm = lambda s: re.sub(r"\W+", "", s.lower())
tasks = [json.loads(l) for l in open(sys.argv[1])]
w1 = [json.loads(l) for l in open(here.parent / "tasks-public.jsonl")]


def probes(t):
    p = norm(re.split(r"\n\nWork it out|\n\nThink it through|\n\nFinish with|\n\nWrite a complete|\n\nComplete the class|\n\nStarter code", t["prompt"])[0])
    return {p[:60], p[-60:]}


corpus = norm("".join(f.read_text() for f in (repo / "experiments").glob("tasks*.py")))
corpus += norm((repo / "evaluation/paired-router-ab-20260929/tasks-public.jsonl").read_text())
corpus += norm("".join(t["prompt"] for t in w1))
d = snapshot_download("openai/gsm8k", repo_type="dataset", allow_patterns=["main/test-*.parquet"])
corpus += norm("".join(pd.read_parquet(glob.glob(f"{d}/main/test-*.parquet")[0]).question[:200]))
d = snapshot_download("HuggingFaceH4/MATH-500", repo_type="dataset")
corpus += norm("".join(pd.concat([pd.read_json(f, lines=True) for f in glob.glob(f"{d}/**/*.jsonl", recursive=True)]).problem))
d = snapshot_download("math-ai/aime25", repo_type="dataset")
for f in glob.glob(f"{d}/**/*.jsonl", recursive=True) + glob.glob(f"{d}/**/*.parquet", recursive=True):
    df = pd.read_json(f, lines=True) if f.endswith("jsonl") else pd.read_parquet(f)
    corpus += norm("".join(df.problem))
hits = [t["id"] for t in tasks if any(p in corpus for p in probes(t))]
control = [t["id"] for t in w1 if any(p in corpus for p in probes(t))]
print("tasks:", len(tasks), "overlapping:", hits, "| positive control flags", len(control), "of", len(w1), "wave-1 prompts")
sys.exit(1 if hits or not control else 0)
