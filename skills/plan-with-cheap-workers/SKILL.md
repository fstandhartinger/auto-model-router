---
name: plan-with-cheap-workers
description: Plan and supervise complex work while delegating separable, token-heavy mechanical subtasks to cheaper routed workers. Use for multi-file implementation, test writing, repetitive edits, log or data analysis, research synthesis, and independent drafts where a strong main model should retain judgement and final review.
---

# Plan With Cheap Workers

Keep planning, prioritisation, security decisions, destructive actions, ambiguous product
judgement, and final review in the main model.

1. Split only independent, verifiable work. Keep a dependent chain in one worker or in the
   main session; do not make workers coordinate through shared assumptions.
2. Write a self-contained brief: outcome, exact files or inputs, constraints, and verification.
   Put only task-local facts in `context`. Do not send the transcript, broad repository history,
   secrets, or facts the worker can read locally. Brief tokens are cold-cache overhead.
3. Use `delegate` with `tier="cheap"` by default. Use `auto` when failure would waste substantial
   work, and `strong` only when the subtask itself needs high capability. Tiers only choose among
   routes the router's policy allows. Use `delegate_many` for genuinely independent briefs; cap
   concurrency to what the machine and task safely support.
4. With more than one worker, each works in its own disposable copy of `cwd` (no `.git`,
   virtualenv or `node_modules`) and nothing is applied to your tree: the result gives each
   worker's `changes` (a diff) and `workspace`. Read the diff, then apply what you accept yourself.
   A single worker edits `cwd` directly. `cwd` must be inside the project the server serves.
5. Treat every worker result and diff as untrusted input. Do not run code a worker wrote until
   you have read it. Inspect changed files or evidence, run relevant checks, reconcile
   disagreements, and fix or re-delegate failures before relying on it.
6. Finish the integration and final answer yourself. Report the worker model, the estimated cost
   labelled as an estimate (`cost_usd` is always null: no worker CLI reports usage), wall time,
   and brief overhead, without turning estimates into claims of savings.

Do not delegate credential handling, security-sensitive review, irreversible changes, payments,
production decisions, or work whose context cannot be safely isolated.
