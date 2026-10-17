#!/usr/bin/env python3
"""Cost vs pass-rate chart (dark background, 1920x1080, large axis labels) from analysis.json / pooled-analysis.json.
usage: make_chart.py [median|cheapest|list] OUT.png   (wave 3: single models, router v1 and v2 curves with the s1 classifier, oracle)"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
basis, out = sys.argv[1], sys.argv[2]
A = json.load(open(HERE / "analysis.json"))
n = A["n_tasks"]
arms = A["bases"][basis]["arms"]
BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
ax.set_facecolor(BG)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.tick_params(colors=FG, labelsize=22)
ax.grid(color=GRID, alpha=0.6)
pt = lambda a: (arms[a]["total_cost_usd"] * 1000 / n, 100 * arms[a]["pass_rate"])  # USD per 1,000 tasks
label = {"claude-opus-5": "Claude Opus 5", "gpt-5.6-sol": "GPT-5.6 Sol", "gpt-5.6-luna": "GPT-5.6 Luna", "kimi-k3": "Kimi K3",
         "glm-5.3-flash": "GLM-5.3 Flash", "deepseek-v4.1-flash": "DeepSeek V4.1 Flash"}
pts = {m: pt(f"always-{m}") for m in label}
placed = []
for m, l in label.items():
    x, y = pts[m]
    ax.scatter(x, y, s=360, color="#8b949e", edgecolor=FG, zorder=3)
    dy = 18
    for (px, py) in placed:  # crude de-overlap: push the label down when an earlier one is close
        if abs(py - y) < 2.0 and 0.5 < x / px < 2.0:
            dy = -34
    placed.append((x, y))
    ax.annotate(l, (x, y), xytext=(14, dy), textcoords="offset points", color=FG, fontsize=22)
for clf, color, name in (("v1-s1", "#d29922", "router v1 (today's whichmodel)"), ("v2-s1", "#3fb950", "router v2 (recalibrated)")):
    xs, ys = zip(*(pt(f"router-{clf}-t{t}") for t in range(11)))
    ax.plot(xs, ys, "-o", color=color, lw=4, ms=14, zorder=4, label=f"{name}, tradeoff 0-10")
x, y = pt("oracle")
ax.scatter(x, y, marker="*", s=900, color="#58a6ff", zorder=5, label="oracle (cheapest passing model per task)")
ax.set_xscale("log")
lo = min(v[1] for v in list(pts.values()) + [pt("oracle")] + [pt(f"router-{c}-t{t}") for t in range(11) for c in ("v1-s1", "v2-s1")])
ax.set_ylim(max(0, lo - 6), 102)
ax.set_xlabel("Cost per 1,000 tasks (USD, log scale)", color=FG, fontsize=28)
ax.set_ylabel(f"Pass rate on {n} tasks (%)", color=FG, fontsize=28)
price = {"median": "median AntSeed seller", "cheapest": "cheapest AntSeed seller", "list": "OpenRouter list price"}[basis]
title = f"Wave 3, {n} fresh tasks (pre-registered): router v1 vs v2 across 6 models"
ax.set_title(f"{title}\nprices: {price}", color=FG, fontsize=32, pad=24)
ax.legend(loc="lower right", facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=22)
fig.text(0.01, 0.01, "95% intervals about +/-10 points. Tasks: LiveCodeBench, AIME/HMMT/CMIMC/BRUMO, MMLU-Pro, synthetic. github.com/fstandhartinger/auto-model-router (PR #8, #10)",
         color="#8b949e", fontsize=15)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(out, facecolor=BG)
