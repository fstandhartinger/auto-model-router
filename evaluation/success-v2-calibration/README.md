# Success model v2: calibration on 720 graded outcomes

Why: on the harder second wave of `antseed-irp-eval-20261017` (60 tasks, 6 models) the router at `cost_quality_tradeoff` 0 picked
GLM-5.3 Flash 35 times of 60 and never Claude Opus 5, although Opus passed 56/60 and GLM 39/60. Two causes in `SuccessModel.p`:
measured cells (from a small, easy 78-task matrix) replaced the curve outright, and the curve `sigmoid((cap - (20 + 50d)) / 6)` hit the
0.98 ceiling for every modern model at the difficulties the s1 classifier outputs. Quality tied, cost broke the tie.

## Data (`dataset.csv`, 720 rows = 120 tasks x 6 models)
Waves 1 and 2 of the study (graded by exact answer or hidden tests, never edited by hand). Per task the router's own s1 classification
(`category`, router-scaled `difficulty`), per model the capability for that category as the router computes it (`cap`, Benchmark Heaven,
benchmaxxing and evidence discount applied), Benchmark Heaven `task_tokens` (typical output tokens per task), the request's `max_tokens`
(8000 in wave 1, 16000 in wave 2), the 78-task measured cell if one applies, v1's `v1_p`, and the grade. The per-task outcomes of the
78-task matrix are not in the repository (only cell counts), so they are not part of the fit; the cells enter through the shrinkage prior only.
`build_dataset.py` rebuilds the file (needs the study's harness, PR #8).

## What the data say (and do not)
* v1 is worse than predicting the base rate: log-loss 0.500 vs 0.421 for a constant (720 rows).
* Capability as built (`aa_intelligence_index x 1.5` for math / knowledge / general, `aa_coding_index` for coding) carries almost no signal across these six models:
  logistic regression coefficient about 0 (+0.04 to -0.10 per 10 capability points), and a model-blind difficulty-only curve predicts as well as any capability curve.
  Example: GPT-5.6 Luna has the lowest capability numbers (37.5 math) and passed 95 % of the math tasks; GLM-5.3 Flash has 62.7 and passed 71 %.
* Difficulty (s1) is informative: pass rate 0.99 / 0.87 / 0.76 / 0.64 for difficulty bins 0-0.2 / 0.2-0.4 / 0.4-0.6 / 0.6-0.8.
* What does separate the models is truncation: 36 of the 38 answers cut off at the token limit came from GLM, DeepSeek and Kimi, whose Benchmark Heaven output-tokens-per-task (49k-89k) is
  far above Luna's (4.5k) and Sol's (7.9k). A term `-lambda * clamp(ln(task_tokens / token_budget))` is therefore part of v2. It is available for any catalog model that has Benchmark Heaven token data and
  switches itself off when it does not.
* Measured cells (78 tasks, 4-6 per cell) do not help beyond the curve: out-of-fold log-loss is lowest when their weight is smallest in the grid (prior of 48 pseudo-observations against a cell of 6).

## v2 (`auto_router/data/success_v2.json`, `economics.success_v2()`)
Same form as v1 with refitted numbers: `p = floor + (ceiling - floor) * sigmoid((cap - offset - slope * difficulty) / scale - lambda * length)`, ceiling 0.995 (smooth: `floor + (ceiling - floor) * sigmoid`, never clipped), floor 0.02, one scale for all categories,
`coding` and `math` offsets/slopes shrunk toward the shared ones (ridge), constraints `1/scale >= 0`, `slope >= 0`, `lambda >= 0` (monotone in capability, decreasing in difficulty, tested), measured cells combined
as Beta shrinkage (curve = 48 pseudo-observations, cell = 6). `scale` is large (84 capability points): the fit finds capability a weak signal, so v2 ranks models mostly by difficulty fit and truncation risk, and
only mildly by capability. Treat the ordering of models without measured outcomes accordingly. `fit_success_v2.py` (in `experiments/`) refits it; v1 stays the default.

## Validation (`cv-summary.md`, `cv.json`; `cv.py` runs it)
No tuning on wave 3. 10-fold CV grouped by task (seed 20261018), leave-one-wave-out, leave-one-model-out; metrics log-loss and Brier; offline replay of the router's own `irp.rank()` on held-out tasks with the recorded outcomes of
the picked model. Out-of-fold log-loss (constant base rate 0.421 / v1 0.500 / v2 0.395 task folds; 0.430 / 0.500 / 0.436 leave-one-wave-out; 0.422 / 0.500 / 0.381 leave-one-model-out). v2 beats v1 everywhere;
against the constant it wins in task folds and on unseen models and ties (slightly loses) when trained on one wave and tested on the other, where the token-budget term extrapolates from 8k to 16k.
Replay on 120 tasks (task folds): v1 t0 100/120 at 19 % of always-Opus's cost; v2 t0 107/120 at 25 %, v2 t1-t3 108/120 at 4.5 % (Opus 110/120, Sol 109, Luna 108). The v2 low-cost picks are mostly Luna, i.e. they inherit
Luna's strong record in these two waves: this is in-sample knowledge about the six models, which is why wave 3 (fresh tasks, same models, frozen before any call) is the real test.
