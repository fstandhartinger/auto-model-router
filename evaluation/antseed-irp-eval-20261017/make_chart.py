#!/usr/bin/env python3
"""Cost vs pass rate chart (dark, 1920x1080) from analysis.json. usage: make_chart.py [cheapest|median] OUT.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
basis = sys.argv[1] if len(sys.argv) > 1 else "cheapest"
out = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "chart-cost-vs-quality.png")
A = json.load(open(HERE / "analysis.json"))["bases"][basis]["arms"]
BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
ax.set_facecolor(BG)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.tick_params(colors=FG, labelsize=18)
ax.grid(color=GRID, alpha=0.6)
pt = lambda a: (A[a]["total_cost_usd"] * 1000 / 60, 100 * A[a]["pass_rate"])  # USD per 1,000 tasks
label = {"claude-opus-5": "Claude Opus 5", "gpt-5.6-sol": "GPT-5.6 Sol", "gpt-5.6-luna": "GPT-5.6 Luna", "kimi-k3": "Kimi K3",
         "glm-5.3-flash": "GLM-5.3 Flash", "deepseek-v4.1-flash": "DeepSeek V4.1 Flash"}
off = {"claude-opus-5": (14, 10), "gpt-5.6-sol": (14, -26), "gpt-5.6-luna": (-10, 22), "kimi-k3": (-60, -34),
       "glm-5.3-flash": (12, -28), "deepseek-v4.1-flash": (-40, 22)}
for m, l in label.items():
    x, y = pt(f"always-{m}")
    ax.scatter(x, y, s=260, color="#8b949e", edgecolor=FG, zorder=3)
    ax.annotate(l, (x, y), xytext=off[m], textcoords="offset points", color=FG, fontsize=18)
for clf, color in (("s1", "#3fb950"), ("prior", "#d29922")):
    xs, ys = zip(*(pt(f"router-{clf}-t{t}") for t in range(11)))
    ax.plot(xs, ys, "-o", color=color, lw=3, ms=11, zorder=4, label=f"router ({'with s1 classifier' if clf == 's1' else 'prior only'}), tradeoff 0-10")
x, y = pt("oracle")
ax.scatter(x, y, marker="*", s=700, color="#58a6ff", zorder=5, label="oracle (cheapest passing model per task)")
ax.set_xscale("log")
ax.set_xlim(right=0.6)
ax.set_ylim(83.5, 95)
ax.set_xlabel("Cost per 1,000 tasks at AntSeed prices (USD, log scale)", color=FG, fontsize=22)
ax.set_ylabel("Pass rate on 60 public tasks (%)", color=FG, fontsize=22)
ax.set_title("Routing across 6 AntSeed models: cost vs quality" + (" (cheapest seller per model)" if basis == "cheapest" else " (median seller per model)"),
             color=FG, fontsize=28, pad=24)
ax.legend(loc="lower right", facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=18)
fig.text(0.01, 0.01, "60 public tasks (HumanEval+, MBPP+, GSM8K, MATH, AIME, MMLU-Pro, synthetic everyday); 95% intervals about +/-10 points. "
         "github.com/fstandhartinger/auto-model-router", color="#8b949e", fontsize=14)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(out, facecolor=BG)
