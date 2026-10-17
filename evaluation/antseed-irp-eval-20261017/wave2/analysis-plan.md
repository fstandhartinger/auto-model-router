# Wave 2 analysis plan (frozen in a commit before the first model call)

Why: wave 1 (`../`) had a ceiling effect (every model passed 51-55 of 60 tasks), so it cannot show a quality-vs-cost trade-off.
Wave 2 uses 60 new, harder tasks, disjoint from wave 1 and from the calibration tasks (`../check_disjoint.py` is re-run on
`tasks-full.jsonl`), the same six models and the same router. Nothing from wave 1's results was used to choose wave-2 tasks, and no
model of this study was run on any wave-2 task before this registration.

## Tasks (60; `tasks-public.jsonl`, built by `build_tasks.py`, seed 20261018)
* coding 14: LiveCodeBench v6 (Jan-Apr 2025 AtCoder / LeetCode), 3 easy / 4 medium / 7 hard by LiveCodeBench's own published label; hidden tests
  (all public + evenly spaced private ones, at most 30 per problem) run in the bubblewrap sandbox with the wave-1 nonce verdict.
* math 16: AIME 2026, HMMT Nov 2025, HMMT Feb 2026, CMIMC 2025, BRUMO 2025 (MathArena copies); numeric answers only; 3 easy / 4 medium / 9 hard,
  difficulty = the problem's position in its contest paper (early = easy, late = hard). Exact final answer.
* knowledge 10: MMLU-Pro (MIT), 6 "hard" from the SciBench / TheoremQA sourced physics / chemistry / math items, 4 "easy" from original-MMLU psychology /
  history / health / economics items; 10 options, exact letter. (MMLU-Pro has no per-item published accuracy; the source tag is the public signal.)
* everyday 20: synthetic, seeded, CC0: 2 easy / 8 medium / 10 hard multi-step tasks with exact checks (calendar and business-day arithmetic, time zones with
  DST, table aggregation, interval scheduling, critical path, shortest path, annuity balance, ledger with fee rules, constrained JSON transforms).
* Mix: 12 easy (20 %), 16 medium, 32 hard. Difficulty labels come only from public metadata (above), never from running our models.
* Licences: MMLU-Pro (MIT) and the synthetic tasks (CC0) are published in full. MathArena sets (CC BY-NC-SA 4.0, problems (c) the competitions) and
  LiveCodeBench problem statements ((c) AtCoder / LeetCode) are published as id + source + SHA-256 of the prompt (+ the numeric key for math);
  `build_tasks.py` rebuilds the full file byte-identically and `manifest.json` carries its SHA-256.
* Known limit: LiveCodeBench ships no reference solutions, so the hidden tests of most coding tasks cannot be validated against a solution; the harness is
  validated end to end on three tasks with hand-written solutions (`validate_graders.py`). Tasks no model passes are listed in the results.

## Models and calls
Same 6 models as wave 1 (claude-opus-5, gpt-5.6-sol, gpt-5.6-luna, kimi-k3, glm-5.3-flash, deepseek-v4.1-flash), via OpenRouter, default reasoning
settings, `max_tokens` 16000 (wave 1: 8000), one call per (task, model); transport errors retried at most twice. `finish_reason` is recorded for every call;
truncated (`length`), empty and errored answers fail.

## Prices (three bases; `prices.json`)
Cost of an answer = its real prompt/completion tokens x price. Bases: **median seller** over AntSeed sellers with a positive price (PRIMARY for every headline;
cheapest-seller prices on AntSeed look promotional), **cheapest seller**, and **OpenRouter list price**. AntSeed snapshot of 2026-10-17 (sha256 in prices.json).

## Router and arms
Router as in wave 1 (`auto_router.irp.rank()`, same build, with the local s1 classifier or prior only), candidates priced on each basis, `max_tokens` 16000.
Picks for every task x classifier x basis x tradeoff 0..10 are in `router-plan.csv` (3,960 rows), recorded before any answer exists.
Arms: always-X for each model; always-best = highest pass rate over the single models on this basis (ties: cheaper total); always-Opus = always-claude-opus-5;
always-cheapest = lowest input+output price on the basis; oracle = cheapest passing model per task (none passes: counted as failure at always-cheapest's cost);
router-s1-t0..10, router-prior-t0..10.

## Budget and order
HARD CAP USD 10 for the wave. The runner takes tasks in the seeded order `run-order.json` (all six models of a task before the next task) and stops dispatching at
USD 7.5 spent (max of OpenRouter's reported cost and the list-price estimate; 6 parallel calls). If the guard stops the run, the analysis uses only the tasks for which all six
models answered and says how many; no other task is dropped. Credits are read from GET /api/v1/credits before and after.

## Statistics (`reproduce.py`)
Pass rate with Wilson 95 % CI; total cost; cost ratio vs always-best and vs always-Opus (total over total); paired bootstrap over tasks (10,000 draws,
`random.Random(20261018)`, percentile 95 % CI) for the pass-rate difference and the cost ratio. s1 vs prior: paired bootstrap per tradeoff. Per category and difficulty.

## Headline definitions (registered)
1. **Cost and quality of router-s1 at every tradeoff 0..10** relative to always-Opus and to always-best (cost ratio with bootstrap CI; pass-rate difference with paired
   bootstrap CI). Primary basis: median seller.
2. **"Equal quality"** = lower bound of the 95 % CI of (router pass rate minus reference pass rate) >= -5 points AND the point estimate >= -2 points. Reported for the
   cheapest router-s1 setting (lowest total cost, ties lowest tradeoff) meeting it against always-best and, separately, against always-Opus, or "none". Same for prior-only.
   Choosing the cheapest of 11 settings on the same tasks is a selection and is optimistic; the README says so.
3. **s1 vs prior-only** at each tradeoff (pass-rate difference CI, cost ratio CI).
Headlines are stated on the median-seller basis; cheapest-seller and OpenRouter-list results are shown beside them.

## Post-hoc pooled view (NOT registered)
Wave 1 + wave 2 pooled to 120 tasks (always-best, oracle and bootstrap recomputed on the pooled tasks; wave-1 router picks were made with max_tokens 8000 and only
the median / cheapest bases exist for them, so pooling is shown for those two). Labelled post-hoc in every place it appears.

## Never
No grade is edited by hand. No task is dropped after generation. No wave-2 result is used to change a task, key, price, pick or this plan; a change after the first call
is listed as an amendment in `manifest.json` and README.
