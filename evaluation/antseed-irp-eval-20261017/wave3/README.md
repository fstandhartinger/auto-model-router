# Wave 3: out-of-sample test of the recalibrated router (success model v2)

Wave 2 (`../wave2/README.md`) showed that the router served today (v1) could not match always-Opus quality at any setting. The diagnosis and a fix, **success model v2**, are in library PR
https://github.com/fstandhartinger/auto-model-router/pull/10 (fitted only on the 720 graded outcomes of waves 1 and 2, see `evaluation/success-v2-calibration/`).
Wave 3 tests both routers on 60 NEW tasks. Pre-registered in commit `c9be58f` (tasks, v2 parameter file hash, 7,920 recorded router picks for four arms x three price bases x tradeoff 0-10, analysis plan, headline
definitions) before the first model call. `analysis-plan.md` is the plan; `manifest.json` lists the hashes and the one declared amendment (none to inputs or grades).

## Tasks, models, method
Same composition as wave 2 (14 LiveCodeBench v6 coding, 16 competition math, 10 MMLU-Pro, 20 synthetic everyday tasks; 12 easy / 16 medium / 32 hard), disjoint from waves 1 and 2 (`check_disjoint.py`: 0 overlaps, positive control flags 120/120),
same six models via OpenRouter, default reasoning, `max_tokens` 16000, same graders, same three price bases (median AntSeed seller is the headline basis). Licence handling as in wave 2: MMLU-Pro and synthetic tasks are public, MathArena and
LiveCodeBench texts only as id + source + SHA-256 (`build_tasks.py` rebuilds the full file byte-identically). One declared deviation from wave 2's plan: only 2 AIME 2026 positions 11-15 were left, so one hard math item comes from HMMT Feb 2026 instead.
Run: 360/360 answers, 0 errors; 31 cut off at the token limit (GLM 18, DeepSeek 7, Kimi 4, Opus 2, GPT-5.6 models 0). Real spend USD 7.48 (credits before/after; cap 10), reported cost 7.27 plus OpenAI routes that report 0.

## Result (median-seller prices; 60 tasks; cost = all 60 tasks; vs always-Opus)
| Arm | Passed | Cost | % of Opus cost | Pass diff vs Opus, points (95 % CI) |
|---|---|---|---|---|
| always-Claude Opus 5 | 56/60 | $0.902 | 100 | 0 |
| always-GPT-5.6 Sol (= always-best) | 57/60 | $0.215 | 23.9 (20.1-29.5) | +1.7 (-6.7 to +10.0) |
| always-Kimi K3 | 50/60 | $0.396 | 43.9 | -10.0 (-21.7 to 0.0) |
| always-GPT-5.6 Luna | 49/60 | $0.038 | 4.2 (3.4-5.4) | -11.7 (-21.7 to -1.7) |
| always-DeepSeek V4.1 Flash | 49/60 | $0.067 | 7.4 | -11.7 (-23.3 to 0.0) |
| always-GLM-5.3 Flash | 36/60 | $0.080 | 8.8 | -33.3 (-46.7 to -20.0) |
| oracle (cheapest passing model per task) | 60/60 | $0.071 | 7.9 (4.4-13.5) | +6.7 (+1.7 to +13.3) |
| **router v2 + s1, tradeoff 0** (= all Sol) | 57/60 | $0.215 | 23.9 (20.1-29.5) | +1.7 (-6.7 to +10.0) |
| router v2 + s1, tradeoff 1-5 (Luna, a few GLM) | 49/60 | $0.038-0.041 | 4.2-4.5 | -11.7 (-21.7 to -1.7) |
| router v2 + s1, tradeoff 6 / 7 / 10 | 46 / 39 / 36 | $0.049 / 0.069 / 0.080 | 5.4 / 7.6 / 8.8 | -16.7 / -28.3 / -33.3 |
| router v1 + s1 (served today), tradeoff 0 | 47/60 | $0.176 | 19.5 (12.8-29.0) | -15.0 (-28.3 to -3.3) |
| router v1 + s1, tradeoff 2 | 46/60 | $0.072 | 8.0 | -16.7 (-30.0 to -5.0) |
| router v1 + s1, tradeoff 5 | 44/60 | $0.069 | 7.6 | -20.0 (-33.3 to -6.7) |

Full tables (all arms and tradeoffs, three price bases, per category and difficulty, paired comparisons): `summary.md`, `analysis.json`; picks and outcomes per task: `router-picks.csv`; chart `chart-wave3.png` (also `-cheapest`, `-list`).

### Registered headlines
1. **Equal quality** (CI lower bound >= -5 points and point estimate >= -2 points) against always-Opus: **none** for router-v2-s1, router-v1-s1, router-v1-prior and the exploratory v2-prior, on all three price bases. The best quality setting of v2 (tradeoff 0)
   is always-Sol: +1.7 points, but the CI (-6.7 to +10.0) is too wide to certify "equal". Against always-best (= Sol on this wave) router-v2-s1 at tradeoff 0 "meets" the rule trivially because it picks Sol for every task and is the reference.
2. **v2 versus v1, fixed settings** (v2 minus v1, paired): tradeoff 0 **+16.7 points (95 % CI +8.3 to +26.7)** at 1.22 x the cost (CI 0.92-1.75); tradeoff 2 +5.0 points (CI -8.3 to +18.3) at 0.54 x the cost (0.42-0.66);
   tradeoff 5 +8.3 points (CI -3.3 to +20.0) at 0.59 x the cost (0.49-0.71). On the cheapest-seller basis the pass-rate differences are +16.7 / +5.0 / +5.0 points and the cost ratios 1.13 / 0.74 / 0.75; on OpenRouter list prices +16.7 / +8.3 / +6.7 points and 1.28 / 0.65 / 0.66 (the picks differ slightly by basis).
3. **Best measured trade-off**: Sol-level quality at 24 % of Opus's cost (v2 tradeoff 0, which is also always-Sol), or 82 % of tasks at about 4 % of Opus's cost (v2 tradeoff 1-5, which is mostly always-Luna). Nothing in between exists in the v2 curve:
   the success model ranks models almost independently of the task, so v2 behaves like a price ladder Sol -> Luna -> GLM.

### What this does and does not show
* v2 beats v1 clearly where it matters most: at the top-quality end v1 sends tasks to GLM, DeepSeek and Kimi and loses 15 points (v2-minus-v1 +16.7). It is also 40-50 % cheaper than v1 at tradeoff 2-5 at equal or better pass rate.
* v2 does **not** reach equal quality to Opus under the registered rule, and it is **not** task-adaptive: v2 with the s1 classifier equals v2 without it at tradeoff 0-5 (paired difference 0.0 points), and is worse at tradeoff 6-7
  (-5.0 and -16.7 points, CI for 7: -28.3 to -6.7). The oracle passes 60/60 at 7.9 % of Opus's cost, so per-task routing headroom exists; this router does not capture it. The honest cause (calibration README): Benchmark Heaven capability barely predicts outcomes among these six models.
* The v2 low-cost rung is Luna, which passed 49/60 here but 55/60 and 53/60 in waves 1 and 2; over all 180 tasks: Opus 166, Sol 166, Luna 157, Kimi 151, DeepSeek 146, GLM 126 (post-hoc pooling, not registered). v2 was fitted on the first 120 of them, with the same six models, so wave 3 is
  out of sample for tasks, not for models.
* The selection of "cheapest of 11 settings" is optimistic; 95 % intervals are about +/-10 points at n = 60; LiveCodeBench tests are trusted as published (harness validated on three tasks); the sandbox has no numpy (as in wave 2).
* Prices: AntSeed median-seller prices (snapshot 17 Oct 2026); the cheapest and OpenRouter-list bases are in `summary.md`.

## Reproduce
`python3 reproduce.py --check` recomputes `analysis.json`, `router-picks.csv`, `summary.md` from the published CSVs (no API call). `python3 make_chart.py median chart-wave3.png` redraws the chart. Full run: `build_tasks.py`, `router_plan.py`, `run.py` (needs OPEN_ROUTER_API_KEY and the harness of PR #8 plus PR #10).
