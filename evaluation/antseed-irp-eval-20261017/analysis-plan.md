# Analysis plan (frozen before any model answer was generated)

Question: on a fixed public prompt set, how much does routing with the router save versus always using the best
model, at equal quality, and what does it cost versus always-cheapest? Prices are what an AntSeed buyer pays.

## Design
* 60 tasks (`tasks-public.jsonl`, SHA-256 in `manifest.json`): 15 coding, 15 math, 15 knowledge, 15 everyday, seeded
  sample (seed 20261017), fixed order, deterministic graders only (`grade.py`). Sources and licences are listed per task.
* 6 candidate models, all on AntSeed on 2026-10-17: claude-opus-5, gpt-5.6-sol, gpt-5.6-luna, kimi-k3, glm-5.3-flash,
  deepseek-v4.1-flash. Answers are generated through OpenRouter with default reasoning settings, `max_tokens` 8000,
  one call per (task, model); only transport errors are retried (max 2). Truncated, empty and errored answers fail.
* Router under test: `auto_router.irp.rank()` as served by whichmodel, (a) with the local s1 classifier, (b) without a
  classifier (prior only). The router sees all 6 candidates priced at AntSeed prices and `max_tokens` 8000. Its picks for
  every task, tradeoff 0..10, both classifiers and both price bases are recorded in `router-plan.csv` before any
  answer exists, and are therefore frozen by the manifest hash.
* Prices (`prices.json`): primary = cheapest AntSeed seller per model (lowest input+output price, zero-priced offers
  excluded); sensitivity = median over sellers. Cost of an answer = its real prompt/completion tokens (from
  OpenRouter usage) x AntSeed price. Cached prompt tokens use the cache price (expected to be ~0: prompts are unique).

## Arms (per task)
always-X for each model; always-best = the single model with the highest pass rate (ties: cheaper total); always-opus
is the conventional default (= always-claude-opus-5); always-cheapest = the model with the lowest cheapest-seller
input+output price; oracle = the cheapest passing model by realised cost (no model passed: counted as a failure at the
always-cheapest model's cost); router-s1-t0..10 and router-prior-t0..10 (the pick's real answer is graded and costed).

## Statistics
Pass rate with Wilson 95% CI; total and per-task cost; cost ratio vs always-best (total over total); paired bootstrap over
tasks (10,000 draws, `random.Random(20261017)`, percentile 95% CI) for the pass-rate difference router minus
always-best and for the cost ratio. s1 vs prior: paired bootstrap per tradeoff. Per-category and per-difficulty tables.

## Headline (decided now)
For each classifier, the cheapest (total cost) router setting among tradeoff 0..10 that satisfies, reported separately:
(A) not significantly worse than always-best: upper bound of the 95% CI of the pass-rate difference >= 0 (the brief's
criterion); (B) point estimate of the pass rate >= always-best; (C) non-inferior within 5 points: lower CI bound >= -0.05.
Each is reported as "X % of always-best's cost" with the bootstrap CI of the cost ratio. If no setting meets a
criterion this is stated as such. With 60 tasks (A) is a weak criterion (a router several points worse can still pass it);
the table always shows the CI so the possible deficit is visible. Choosing the cheapest of 11 settings on the same
tasks is a selection and is optimistic; this is stated in the README.

## Never
No grade is edited by hand. No task is dropped after generation. If the spend cap (USD 18 stop / 20 hard) is hit the
run reports the unfinished pairs as missing and the analysis says so.
