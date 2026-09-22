# Delegation evaluation

A **derived re-tabulation** of the planner/worker runs made on 19 September 2026
([`EXPERIMENTS.md`](EXPERIMENTS.md) §15), written on 21 September. Nothing new
was run for it. Machine-readable summary:
[`evaluation/delegation-20260921.json`](evaluation/delegation-20260921.json).

## Result

No saving was measured on these small tasks.

| system | quality | plan-model use | wall time |
|---|---:|---:|---:|
| Claude plan model alone (Sonnet 5) | 3/3 | $0.460 API-equivalent | 67.9 s |
| Claude plan model + free worker | 3/3 | $0.468 API-equivalent | 277.7 s |
| Codex alone (ChatGPT plan) | 1/1 | 15,260 plan tokens | 31 s |
| Codex + free worker | 1/1 | 16,269 plan tokens | 74 s |

- "API-equivalent" is the plan model's token counts from Claude Code's
  transcripts priced at list price; the plan itself charges nothing per token,
  and its own usage meter could not be sampled (§15.4).
- In the Claude orchestrator arm the planner delegated in t1 and t3; on t2 it
  did the work itself. The delegate arm's t2 is therefore not a delegation run.
- The Claude planner arm was 1.7 % more expensive at API-equivalent list prices
  and 4.1× slower. The Codex planner arm used 6.6 % more plan tokens and was
  2.4× slower. Quality was unchanged because every run passed its hidden check.

## Worker cost and brief overhead

The worker ran on a route configured as free; its cost is **not measured**, and
the worker CLI reported no token usage. The router's ledger for the Codex run
*estimated* 179 cold prompt tokens for the brief plus declared tools; the
worker's fixed agent prompt and actual input are unknown.

**Which model the Codex worker used is recorded inconsistently.** The router's
ledger (`codex-delegate-ledger.jsonl`) records the selected route as
`qwen3.8-27b` (fallback `kimi-k3`, classifier disabled). The run's own result
note (`codex-results.jsonl`) and §15 say the sub-task ran on Kimi K3 via
OpenCode. The artifacts do not settle which is right, so the summary records
both and names neither as the worker.

## Method and limits

The source run on 19 September 2026 used three small Python agent tasks: create a
slugifier with tests, fix an inventory module against existing tests, and create a
small word-count CLI package. Hidden scripts graded the resulting files. Claude
ran all three arms; Codex repeated the bug-fix task. The planner was told to
plan, delegate implementation, and verify the result itself.

- n=1 per task, small tasks, one free route.
- The numbers predate the delegate tool's worker tiers, parallel workers,
  per-worker copies, environment allowlist and timeout containment (21 Sep),
  and the symlink handling and link reporting of those copies (22 Sep); none of them was
  exercised by a live worker. They are covered only by tests with fake workers.
- It shows where delegation made the result worse: fixed planner prompts, MCP
  handoff, cold worker context, and final review cost more than they saved on
  small jobs. It does not test a large separable job or parallel workers, where
  moving a token-heavy middle phase away from the plan model could still help.
  No such saving is claimed.
