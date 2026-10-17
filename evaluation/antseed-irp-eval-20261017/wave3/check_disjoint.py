#!/usr/bin/env python3
"""Wave 3 is disjoint, by content, from waves 1 and 2 and from the calibration tasks: normalised 60-character probes (start and end of each
prompt) of every wave-3 task are searched in the wave-1 and wave-2 prompts, experiments/tasks*.py, the paired-study tasks, and the HF sets that
wave 1 excluded (GSM8K rows 0-199, MATH-500, AIME 2025). Positive controls: a wave-1 and a wave-2 prompt must be flagged.
usage: check_disjoint.py TASKS_FULL.jsonl WAVE2_TASKS_FULL.jsonl   (needs network / HF cache; wave-2 full prompts are rebuilt by ../wave2/build_tasks.py)"""
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
w2 = [json.loads(l) for l in open(sys.argv[2])]


def probes(t):
    p = norm(re.split(r"\n\nWork it out|\n\nThink it through|\n\nFinish with|\n\nWrite a complete|\n\nComplete the class|\n\nStarter code", t["prompt"])[0])
    return {p[:60], p[-60:]}


corpus = norm("".join(f.read_text() for f in (repo / "experiments").glob("tasks*.py")))
corpus += norm((repo / "evaluation/paired-router-ab-20260929/tasks-public.jsonl").read_text())
corpus += norm("".join(t["prompt"] for t in w1 + w2))
d = snapshot_download("openai/gsm8k", repo_type="dataset", allow_patterns=["main/test-*.parquet"])
corpus += norm("".join(pd.read_parquet(glob.glob(f"{d}/main/test-*.parquet")[0]).question[:200]))
d = snapshot_download("HuggingFaceH4/MATH-500", repo_type="dataset")
corpus += norm("".join(pd.concat([pd.read_json(f, lines=True) for f in glob.glob(f"{d}/**/*.jsonl", recursive=True)]).problem))
d = snapshot_download("math-ai/aime25", repo_type="dataset")
for f in glob.glob(f"{d}/**/*.jsonl", recursive=True) + glob.glob(f"{d}/**/*.parquet", recursive=True):
    df = pd.read_json(f, lines=True) if f.endswith("jsonl") else pd.read_parquet(f)
    corpus += norm("".join(df.problem))
# synthetic tasks (everyday3-*) reuse the generators' wording by design (same composition as wave 2) with fresh seeded data, so head/tail probes would
# match the template; for them disjointness means "no identical prompt" (checked below on the full normalised prompt), not "no shared wording".
hits = [t["id"] for t in tasks if not t["id"].startswith("everyday3-") and any(p in corpus for p in probes(t))]
control = [t["id"] for t in w1 + w2 if any(p in corpus for p in probes(t))]
# the corpus contains the wave-1/2 prompts, so the control proves the probes would catch a repeated prompt; also check every id/prompt pair directly
old = {norm(x["prompt"]) for x in w1 + w2}
dups = [t["id"] for t in tasks if t["id"] in {x["id"] for x in w1 + w2} or norm(t["prompt"]) in old]
print("tasks:", len(tasks), "overlapping:", hits, "| duplicate ids/prompts:", dups, "| positive control flags", len(control), "of", len(w1) + len(w2), "wave-1/2 prompts")
sys.exit(1 if hits or dups or len(control) < len(w1) + len(w2) else 0)
