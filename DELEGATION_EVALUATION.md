# Delegation evaluation

Measured comparison for the planner/worker pattern. Raw summary:
[`evaluation/delegation-20260921.json`](evaluation/delegation-20260921.json).

## Result

No saving was measured on these small tasks.

| system | quality | strong-model use | wall time |
|---|---:|---:|---:|
| Claude strong model alone | 3/3 | $0.460 API-equivalent | 67.9 s |
| Claude planner + cheap worker | 3/3 | $0.468 API-equivalent | 277.7 s |
| Codex strong model alone | 1/1 | 15,260 plan tokens | 31 s |
| Codex planner + cheap worker | 1/1 | 16,269 plan tokens | 74 s |

The Claude planner arm was 1.7% more expensive at API-equivalent list prices and
4.1× slower. The Codex planner arm used 6.6% more plan tokens and was 2.4×
slower. Quality was unchanged because every run passed its hidden check.

The Codex worker was a configured free route, so its marginal metered cost was
$0. Its router record estimated 179 cold input tokens for the brief plus declared
tools. The worker CLI did not return actual token usage, so the fixed agent prompt
and exact handoff overhead are unknown; the MCP server now exposes
`brief_tokens_estimate` on every result so future runs retain this explicitly.

## Method and limits

The source run on 19 September 2026 used three small Python agent tasks: create a
slugifier with tests, fix an inventory module against existing tests, and create a
small word-count CLI package. Hidden scripts graded the resulting files. Claude
ran all three arms; Codex repeated the bug-fix task. The strong model was told to
plan, delegate implementation, and verify the result itself. The worker ran
through the router on a free model.

This is a small n=1-per-task sample. It shows where delegation made the result
worse: fixed planner prompts, MCP handoff, cold worker context, and final review
cost more than they saved on small jobs. It does not test a large enough
separable job or multiple parallel workers, where moving a large token-heavy
middle phase away from the strong model could still help. No such saving is
claimed.
