# Wave 3 analysis plan (frozen in a commit before the first model call)

Why: wave 2 showed that no router-v1 setting reached the quality of always-Opus 5. The diagnosis (saturated success curve, measured cells overriding it) led to success model v2
(`auto_router/data/success_v2.json`, library PR #10), fitted ONLY on the 720 graded outcomes of waves 1 and 2 (`evaluation/success-v2-calibration/`). Wave 3 is the out-of-sample test:
60 NEW tasks, the same six models, the same router build, the v1 router as it is served today, and the v2 router, both with picks recorded before any model call.
No wave-3 result has been used for any fit, choice or threshold. The v2 parameter file hash is in `manifest.json` / `router-catalog.json`.

## Tasks (60; `tasks-public.jsonl`, `build_tasks.py`, seed 20261019)
Same composition and rules as wave 2 (see `../wave2/analysis-plan.md`): 14 LiveCodeBench v6 coding (3 easy / 4 medium / 7 hard by LiveCodeBench's own label), 16 competition math
(AIME 2026, HMMT Nov 2025 / Feb 2026, CMIMC 2025, BRUMO 2025; difficulty from the position in the paper), 10 MMLU-Pro (6 SciBench/TheoremQA-sourced hard, 4 easy), 20 synthetic multi-step
everyday tasks (2 easy / 8 medium / 10 hard; fresh seeded data, ids `everyday3-*`). 12 easy, 16 medium, 32 hard. Difficulty comes from public metadata only, never from running our models.
Disjoint from waves 1 and 2 and the calibration tasks: no problem, MMLU-Pro item or (contest, position) is repeated, no prompt is identical, and for the non-synthetic tasks 60-character probes of start and end
of every prompt do not occur in any earlier prompt, `experiments/tasks*.py`, the paired-study tasks, GSM8K rows 0-199, MATH-500 or AIME 2025 (`check_disjoint.py`, positive control: all 120 earlier prompts flagged).
Deviation from wave 2's math plan: wave 2 used 3 of the 5 AIME 2026 positions 11-15, so wave 3 takes 2 there and one more HMMT Feb 2026 item from positions 21-33 instead (declared in `build_tasks.py`).
Licences and publication as in wave 2: MMLU-Pro and the synthetic tasks are published in full; MathArena and LiveCodeBench texts only as id + source + SHA-256 (+ numeric key for math).
Known limit (unchanged): LiveCodeBench ships no reference solutions; the harness is validated end to end on three wave-2 coding tasks with hand-written solutions (`validate_graders.py`).

## Models and calls
Claude Opus 5, GPT-5.6 Sol, GPT-5.6 Luna, Kimi K3, GLM-5.3 Flash, DeepSeek V4.1 Flash via OpenRouter, default reasoning settings, `max_tokens` 16000, one call per (task, model), transport errors retried at most twice.
Truncated (`finish_reason=length`), empty and errored answers fail. Graders are those of wave 2 including its amendment A2 (`grade_w3.py`).

## Prices
`prices.json` is wave 2's (AntSeed snapshot of 2026-10-17, three bases). Bases: **median seller (PRIMARY for every headline)**, cheapest seller, OpenRouter list price.

## Router arms (`router-plan.csv`, 7,920 rows recorded before any answer exists; `router_plan.py`)
`auto_router.irp.rank()` as whichmodel calls it, six candidates priced on each basis, `max_tokens` 16000, tradeoff 0..10, s1 classifier (local, revision in `router-catalog.json`).
* **router-v1-s1**: success model v1 (curve + measured cells that override it) = the router as served today.
* **router-v2-s1**: success model v2 (curve refit, smooth ceiling, token-budget term, measured cells as Beta prior).
* **router-v1-prior**: no classifier, v1.   * **router-v2-prior**: no classifier, v2 (exploratory, no registered claim).
Capabilities and Benchmark Heaven token data are frozen in `router-catalog.json` (fetched on the run date).
Other arms: always-X for each model; always-best = highest pass rate over the single models on that basis (ties: cheaper total); always-Opus; always-cheapest = lowest input+output price on the basis;
oracle = cheapest passing model per task (none passes: counted as failure at always-cheapest's cost).

## Statistics (`reproduce.py`)
Pass rate with Wilson 95 % CI; total cost; cost ratio vs always-best and vs always-Opus (total over total); paired bootstrap over tasks (10,000 draws, `random.Random(20261019)`, percentile 95 % CI) for pass-rate differences and cost ratios.

## Registered headlines (median-seller basis; cheapest and list shown beside them)
1. **Equal quality of router-v2-s1.** "Equal quality" = lower bound of the 95 % CI of (router pass rate minus reference pass rate) >= -5 points AND point estimate >= -2 points (as in wave 2).
   Reported for the cheapest router-v2-s1 setting (lowest total cost, ties lowest tradeoff) that meets it against always-Opus and, separately, against always-best, or "none". Choosing the cheapest of 11 settings on the same tasks is a selection and is optimistic;
   the README says so. The same is reported for router-v1-s1 and router-v1-prior.
2. **v2 versus v1 at fixed settings** (not selected after the fact): router-v2-s1 minus router-v1-s1 at tradeoff 0, 2 and 5 (5 is the whichmodel default): paired pass-rate difference with CI and cost ratio with CI.
3. **Best measured trade-off**: the full table of router-v2-s1 settings 0..10 against always-Opus / Sol / Luna and the oracle, so a reader can pick a setting; no setting is declared best by a rule applied to wave 3 alone.
What v2 is expected to do (written before the run, from the recorded picks): at tradeoff 0 all 60 picks are GPT-5.6 Sol, i.e. router-v2-s1-t0 is identical to always-Sol; tradeoffs 1-5 pick mostly Luna with some GLM; 7-9 mostly GLM;
10 is all GLM. If so, the router-v2 results reduce to those of three single models and the classifier adds little; the report says so if it happens.

## Budget and order
HARD CAP USD 10 for the wave. Tasks in the seeded order `run-order.json` (all six models of a task before the next task); dispatch stops at USD 9.0 spent (max of OpenRouter's reported cost and the list-price estimate, 6 parallel calls),
which is above the realised cost of wave 2 (USD 7.93 real for a guard value of 9.2). If the guard stops the run, the analysis uses only the tasks for which all six models answered. Credits are read from GET /api/v1/credits before and after
(`credits-before.json`, `credits-after.json`; other jobs share the key, so the run's own reported cost is also summed from `results.csv`).

## Never
No grade is edited by hand. No task is dropped after generation. No wave-3 result is used to change a task, key, price, pick, parameter or this plan; any change after the first call is listed as an amendment in `manifest.json` and the README.
