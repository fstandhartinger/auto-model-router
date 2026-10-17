# Wave 2: the same router test on 60 harder tasks

Wave 1 (`../README.md`) had a ceiling effect: every model passed 51-55 of 60 tasks, so it could not show a cost-vs-quality trade-off. Wave 2 uses 60 new,
harder tasks with the same six models, the same router and the same method. Pre-registered in commit `4b693ca` (tasks, router picks for every arm, tradeoff and
price basis, analysis plan, headline definitions) before the first model call. Draft PR: https://github.com/fstandhartinger/auto-model-router/pull/8

## What the 60 tasks are
| Category | n | Source | Grader |
|---|---|---|---|
| coding | 14 | LiveCodeBench v6 (AtCoder / LeetCode, Jan-Apr 2025), 3 easy / 4 medium / 7 hard by the dataset's own label | hidden tests in the bubblewrap sandbox (<= 30 per problem) |
| math | 16 | AIME 2026, HMMT Nov 2025 / Feb 2026, CMIMC 2025, BRUMO 2025 (numeric answers; later problem = harder) | exact final answer |
| knowledge | 10 | MMLU-Pro: 6 SciBench / TheoremQA sourced (college level), 4 easy-category | exact letter |
| everyday | 20 | synthetic, seeded: dates, business days, time zones with DST, tables, scheduling, graphs, annuities, ledger rules, constrained JSON | exact value / JSON |

12 easy (20 %), 16 medium, 32 hard. Difficulty labels use only public metadata (LiveCodeBench label, position in the contest paper, MMLU-Pro source tag); no model of
this study was run on any task before registration. The set is disjoint from wave 1 and the calibration tasks (`check_disjoint.py`: 0 overlaps, positive control flags 60/60).

**Licences / what is published.** MMLU-Pro (MIT) and the synthetic tasks (CC0) are published in full in `tasks-public.jsonl`. The MathArena sets are CC BY-NC-SA 4.0
(problems (c) the competitions) and LiveCodeBench statements are (c) AtCoder / LeetCode, so for those the public file has only id, source link, difficulty, the
SHA-256 of the prompt (and for coding the SHA-256 of the tests, for math the numeric key). `build_tasks.py` rebuilds the full file byte-identically from the public
datasets (SHA-256 of that file is in `manifest.json`). GPQA was not usable (gated, and its licence asks for no republication).

## Setup (as wave 1 except where stated)
Six models via OpenRouter, default reasoning settings, one call per task and model, **max_tokens 16000** (wave 1: 8000), `finish_reason` recorded; truncated answers fail.
Router: `auto_router.irp.rank()` as served by whichmodel, with the s1 classifier or prior only, tradeoff 0-10. Prices on three bases, all computed from each answer's real
token counts: **median AntSeed seller (primary for all headlines)**, cheapest AntSeed seller, OpenRouter list price. Statistics: Wilson CIs, paired bootstrap over tasks
(10,000 draws, seed 20261018).

## Result (median-seller prices; cost = all 60 tasks; Opus = Claude Opus 5)
Single models: Opus 56/60, Sol 55, Luna 53, Kimi K3 48, DeepSeek V4.1 Flash 44, GLM-5.3 Flash 39. The models now separate (weakest 65 %, strongest 93 %; wave 1: 85-92 %).

| Arm | Passed | Total cost | Cost vs always-Opus (95 % CI) | Pass diff vs Opus, points (95 % CI) |
|---|---|---|---|---|
| always-claude-opus-5 | 56/60 (93%) | $0.839 | 100.0% (100.0-100.0) | +0.0 (+0.0 to +0.0) |
| always-gpt-5.6-sol | 55/60 (92%) | $0.239 | 28.4% (24.6-32.8) | -1.7 (-8.3 to +3.3) |
| always-gpt-5.6-luna | 53/60 (88%) | $0.034 | 4.1% (3.5-5.0) | -5.0 (-11.7 to +1.7) |
| always-kimi-k3 | 48/60 (80%) | $0.469 | 55.9% (43.8-75.8) | -13.3 (-23.3 to -3.3) |
| always-deepseek-v4.1-flash | 44/60 (73%) | $0.063 | 7.5% (5.9-10.0) | -20.0 (-30.0 to -10.0) |
| always-glm-5.3-flash | 39/60 (65%) | $0.061 | 7.3% (5.6-10.0) | -28.3 (-40.0 to -16.7) |
| oracle | 58/60 (97%) | $0.065 | 7.7% (4.0-14.1) | +3.3 (+0.0 to +8.3) |
| router-s1-t0 | 48/60 (80%) | $0.170 | 20.3% (12.0-32.3) | -13.3 (-23.3 to -3.3) |
| router-s1-t1 | 48/60 (80%) | $0.188 | 22.4% (13.1-36.1) | -13.3 (-25.0 to -1.7) |
| router-s1-t2 | 45/60 (75%) | $0.081 | 9.6% (5.9-15.0) | -18.3 (-30.0 to -8.3) |
| router-s1-t3 | 43/60 (72%) | $0.054 | 6.4% (5.0-8.8) | -21.7 (-33.3 to -10.0) |
| router-s1-t5 | 43/60 (72%) | $0.052 | 6.2% (4.9-8.4) | -21.7 (-33.3 to -10.0) |
| router-s1-t8 | 41/60 (68%) | $0.056 | 6.7% (5.3-8.9) | -25.0 (-36.7 to -15.0) |
| router-s1-t10 | 39/60 (65%) | $0.061 | 7.3% (5.6-10.0) | -28.3 (-40.0 to -16.7) |
| router-prior-t0 | 55/60 (92%) | $0.239 | 28.4% (24.6-32.8) | -1.7 (-8.3 to +3.3) |
| router-prior-t1 | 48/60 (80%) | $0.469 | 55.9% (43.8-75.8) | -13.3 (-23.3 to -3.3) |
| router-prior-t2 | 39/60 (65%) | $0.061 | 7.3% (5.6-10.0) | -28.3 (-40.0 to -16.7) |

Always-best = always-Opus on this wave, so "vs always-best" and "vs always-Opus" are the same numbers. Cheapest-seller and OpenRouter-list versions of every table are in
`summary.md` (list-price cost vs Opus: router-s1 t0 27 %, Sol 36 %, Luna 2.4 %; cheapest-seller: 25 %, 27 %, 1.1 %).

### Registered headlines
1. **Cost and quality of router-s1 at each tradeoff vs always-Opus / always-best:** table above and `summary.md`. s1 at t0 (its most quality-oriented setting) costs 20 % of
   always-Opus (CI 12-32 %) at 48/60 vs 56/60 (-13.3 points, CI -23.3 to -3.3). t2 costs 9.6 % (CI 5.9-15 %) at 45/60 (-18.3, CI -30 to -8.3).
2. **Equal quality** (CI lower bound >= -5 points AND point estimate >= -2 points): **none**, for router-s1 and for prior-only, against always-best, against always-Opus, on all three price
   bases. No router setting matches Opus on these harder tasks.
3. **s1 vs prior-only** (median prices): at t2-t7 s1 passes 6.7-10 points more tasks (CI lower bound 0.0 at t2-t7) at 0.85-0.88 of the prior-only cost for t3-t7 (t2: cost ratio 1.32, CI 0.82-2.25); at t0 s1 is
   worse (-11.7 points, CI -21.7 to -3.3) because prior-only sends everything to Sol; at t1 equal quality, s1 costs 0.40 of prior-only (CI 0.25-0.56); at t10 identical (both all-GLM).

What the numbers say, without spin: **the router did not find a saving at equal quality on harder tasks.** It spreads picks over GLM-5.3 Flash, DeepSeek V4.1 Flash and Kimi K3 (and Sol at t0) and
never picked Claude Opus 5 (0 of 3,960 plan rows) and picked GPT-5.6 Luna only 60 times, although Luna passed 53/60 at 4 % of Opus's cost. The best cost-quality points on this task set are single models:
Sol (55/60, 28 % of Opus's cost, -1.7 points, CI -8.3 to +3.3 - not distinguishable from Opus, but the interval is too wide to meet the registered 'equal quality' bound; note that prior-only t0 *is* always-Sol, not routing) and
Luna (53/60 at 4 %). The oracle (cheapest passing model per task) reaches 58/60 at 7.7 % of Opus's cost, so per-task routing headroom exists; this router's capability tables did not capture it here.

### Post-hoc pooled view, 120 tasks (wave 1 + wave 2; NOT a registered analysis)
Wave-1 router picks were made with max_tokens 8000 and only the median / cheapest bases exist for them, so pooling is shown for those. Pooled, median prices: Opus 110/120, Sol 109, Luna 108, Kimi 101, DeepSeek 97, GLM 90.
router-s1 t0 100/120 (-8.3 points vs Opus, CI -14.2 to -2.5) at 19.3 % of Opus's cost (CI 12.3-28.7 %); t2 98/120 at 9.5 %. No s1 setting meets 'equal quality' (also not pooled). The only router-labelled setting that does is
prior-only t0, which is always-Sol (109/120, -0.8 points, CI -4.2 to +2.5) at 31.5 % of Opus's cost (CI 26.4-39.1 %). Full tables: `pooled-summary.md`, `pooled-analysis.json`; chart: `chart-pooled.png`.
The pooling combines tasks of very different difficulty and was decided after seeing wave 2; read it as context only.

## Spend
OpenRouter credits (GET /api/v1/credits): before total_usage 509.528709362 (total_credits 578.9112268), after 517.462548287: **USD 7.93 for this wave** (cap USD 10). Guard value (max of reported cost and
list-price estimate; OpenAI routes report cost 0) USD 9.20. Reported cost by model (sum 7.77; the USD 0.17 difference to the credit delta is the repeated in-flight calls of A1): Kimi K3 3.96, Opus 3.46, DeepSeek 0.23, GLM 0.12, Sol/Luna 0 (not charged to credits).
360/360 answers, 0 errors, 0 retries; 38 answers truncated at 16,000 tokens (GLM 14, Kimi K3 11, DeepSeek 11, Opus 2, Sol 0, Luna 0).

## Defects, amendments and limits (all of them)
* **A1 (spend guard, operational):** registered stop USD 7.5; the run hit it after 327 answers, 33 pairs were left. With headroom to the USD 10 cap I resumed with the stop raised to USD 9.2 (declared in `manifest.json`;
  the order and all prompts unchanged, no result was used). All 60 tasks have all six answers, so the registered contingency (analyse complete tasks only) was not needed. One resume attempt was cut off by a shell timeout; its in-flight calls
  (about USD 0.17) were repeated.
* **A2 (grader bug found after the run, fixed and declared):** the class-style coding grader prepended `from math import *`, which shadows the builtin `pow` (3-argument modular `pow` raised TypeError). All six models failed
  `coding-lcb6-leetcode-3739` for that reason, which is how it was noticed. Fix: `from builtins import *` after the imports in `grade_w2.py`; every stored answer was regraded with the fixed grader; **only the six
  answers to that task changed (0 -> 1, all six now pass)**, no other grade changed. The pre-fix grades are kept in `results-before-A2.csv` (before: Opus 55, Sol 54, Luna 52, Kimi 47, DeepSeek 43, GLM 38). Both versions are in the manifest.
  The registered picks, keys and prices did not change.
* **Sandbox has no numpy:** one answer (Opus on `coding-lcb6-atcoder-arc192_d`) imports numpy and fails with ModuleNotFoundError; the grade stays a failure (the prompt did not forbid numpy). It costs Opus at most one task.
* **Coding tests are unvalidated except on three tasks:** LiveCodeBench ships no reference solutions; the harness is validated end to end with hand-written solutions for 3 tasks (both modes) and a stub fails everywhere. Tests are capped at 30 per
  problem (all public + evenly spaced private). AtCoder problems run as Python with a 60 s wall budget for all tests, so slow-but-correct Python can fail for every model alike.
* **Truncation at 16,000 tokens** accounts for a large part of the weaker models' failures (36 of the 38 truncated answers are GLM / DeepSeek / Kimi); their pass rates measure "correct within 16,000 tokens at default reasoning", not only knowledge.
* **Two tasks no model passed:** `math-aime_2026-15` (4 truncated, 2 wrong) and `math-cmimc_2025-7` (4 truncated, 2 answered 932 vs key 930); kept as registered. 33 tasks were passed by all six models.
* **Difficulty target missed at the top:** I aimed at weakest 30-50 % and strongest 70-85 %; actual 65 % / 93 %. The task set separates the models but the strongest three are still close (CIs about +-10 points).
* **Router limits:** the router build is wave 1's (capability tables from Benchmark Heaven, success model without a Kimi / DeepSeek V4.1 measurement); it never prefers Opus because of price. These results describe that build, not the s1 classifier alone.
* **Selection:** 'cheapest setting meeting equal quality' picks among 11 tradeoffs on the same tasks (optimistic); here it is 'none' so it does not matter.
* Task set built after wave 1 was seen: the *decision* to make a harder wave came from wave 1's ceiling, not from any wave-2 number.

## Files
`tasks-public.jsonl`, `results.csv` (every call: pass, tokens, cost on three bases, finish_reason), `results-before-A2.csv`, `router-plan.csv`, `router-picks.csv`, `prices.json`, `analysis.json`, `summary.md`, `pooled-analysis.json`, `pooled-summary.md`,
`chart-wave2.png` (median prices, 1920x1080, dark; `chart-wave2-cheapest.png`, `chart-wave2-list.png`, `chart-pooled.png`), `analysis-plan.md`, `manifest.json`, `run-order.json`, code: `build_tasks.py`, `build_prices.py`, `router_plan.py`, `run.py`, `grade_w2.py`,
`reproduce.py`, `make_chart.py`, `make_manifest.py`, `validate_graders.py`, `reference-solutions.json`, `check_disjoint.py`. The answer texts stay local.

```
python reproduce.py --check      # recomputes analysis.json, router-picks.csv, summary.md, pooled-* from the CSVs; no API calls
python build_tasks.py full.jsonl tasks-public.jsonl      # rebuilds the tasks (needs network / HF)
python check_disjoint.py full.jsonl && python validate_graders.py full.jsonl
OPEN_ROUTER_API_KEY=... python run.py full.jsonl prices.json answers.jsonl results.csv      # about USD 8-9
```
