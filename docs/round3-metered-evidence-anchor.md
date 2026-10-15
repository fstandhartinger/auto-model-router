# Round-3 metered-arm supplement — external evidence anchor

This file is the PLAN §2 external anchor for the round-3 metered-arm supplement (the
round-3 held-out run that gives the router a genuine cash figure). It binds, by digest,
the frozen registration, the reviewed analysis script, the executed run's raw artifacts,
the probe-budget decision, the independent review chain, and the one authorised claim.
It is additive documentation only: it changes no code, no harness, no policy, and no
registration. A reader who does not hold the secret-bearing runtime config cannot
reconstruct the run from it (that gap is stated, not papered over, by the harness's own
`experiments/evidence_verify.py`); this anchor records the public identity and the
reviewed chain so the run is auditable end to end.

- Executed at repo commit: `e2160900ae417a590be5801c4f1d8713769cf55e`
  (the registered `repo_git_sha`). Remote `main` had independently advanced to
  `2f9c1bbc85d11345ca04a9c299cdb0c10b1a4fde` (IRP v1, PR #6, additive) before the run;
  the run was executed at the registered pin, not fast-forwarded, so every registered
  digest matched exactly.
- Registration: `preregistration.json` registered 2026-10-04T06:20:16Z, predating every
  call. Verified by `heldout.py verify --dir round3-metered-runs`:
  "pre-registration intact: 70 tasks, 0 amendment(s)". The `note` in the preserved
  registration still names the superseded pre-repair cash script `a349d838`; the final
  reviewed script is `df055fcb` below. This historical note is preserved, not rewritten.

## 1. Frozen inputs (sha256)

| artifact | sha256 |
|---|---|
| preregistration.json | `3fff428a2a38938e872b973fd9018a967a6d11652bb44cd02423a5fa797c425a` |
| tasks.jsonl (task_file) | `0f7f78aaa2eaeadddf73fc54777ef24f4a50718a4b3519bcad373039e48951e6` |
| PARTITION.json | `ea55ede54fefb5fb0225541937c1740cbb47a14b45553bc6cbffe0be1a8d4a25` |
| CASH-CRITERION.md (PLAN §4.6 verbatim) | `b08f1e16763ab6071ab18a492f7b2c8ef614bc3ce460b67c97f44db9f7b4869f` |
| cash_diff.py (FINAL REVIEWED) | `df055fcba265b5a2572765f911319a6e14059412256eef5a5cf9c451eec7c91c` |
| config-projection.yaml (credential-free) | `a3fd07a123e4058a9c28fdb906eaf869cee3702354a98ab0dd9a6b67d01016e5` |
| IDENTITY-DIFF.json | `531457621f6b5316fe270674d5a7e19564a8fda1d584f27df4ceddf69ca92acf` |

The partition is exactly 14 `design` tasks → `gpt-5.6-sol` by the default-off operator rule
(`paid_rule_route: gpt-5.6-sol`, `paid_rule_min_max_tokens: 10000`, rule reason "stated
max_tokens 12000 >= 10000 -> gpt-5.6-sol") and 56 other tasks → free `kimi-k3`.

## 2. Executed run (raw artifact digests)

| artifact | sha256 |
|---|---|
| run-meta.json | `b7049f0bbe36329086291d7de4e1239c61c1468ca691b170397fb1b322ac804d` |
| calls.jsonl (140 rows: 70 router + 70 control) | `4a057b882916ed616dd6ef85897cd4d601840c6c5471772fb3b8b23abf4a946a` |
| ledger.jsonl (70 pairs) | `934654150224739d123bdb13f4406f46c6c2e4b01731cf5b63a7fb6bd59ba1b6` |
| router-decision-records.jsonl (70 live decisions) | `2d407b94a842aeb27e8ec3afa909cf5d9783c811f9527dc2934e7008464a3b1c` |
| report.json | `ff597c6f4ad9cbd3c1be46e134d2bf4dcc8b9db163e382feb69ae00512eb6244` |
| run.log | `545d5aed1506bc0bccd2c108493db7410ccd7d37be60f6ffafa3bd9f90f7f67b` |
| cash-diff-out.json | `3554978fbbf3da97a1fa34e3de417a8e59f39ae0a3194c35d1abbdd5c2999b9f` |
| provider-before-2.json (18:02:26Z) | `c315d10aac52a8449c34305664638ee21a88f601091de8ec9ab100b8080f3e3d` |
| provider-after-2.json (18:55:07Z) | `8e980c1ebe6a7dc233cf2be51d4dd31d80cf480b734601e3a46cb51b7f7721f1` |

Run: 2026-10-12, 18:04:20Z–18:51:30Z, rc 0, 70/70 pairs, 140 calls, 0 errors, 0 excluded.
No stop condition (S1–S10) fired; S9 headroom re-checked immediately before the run.
Coding answers executed inside the harness's Bubblewrap sandbox (network blocked, home not
readable, no secret env). `report.json`: `harness_drift_since_registration []`,
`amendments []`, no (category, arm) cell below 10 graded rows.

### Cash result (CD1–CD4 all PASS; bootstrap seed 20261003, 10,000 resamples, percentile)

| | |
|---|---|
| C (control, all gpt-5.6-sol) | $0.791442 |
| R (router) | $0.594188 |
| D = C − R | $0.197254 (95% CI $0.081624–$0.281017, excludes 0) |
| A (routing-attributable, 56 free-routed pairs) | $0.196794 (95% CI $0.164404–$0.228941) |
| B (14 design pairs, same model both arms) | $0.000460 |
| D = A + B | yes |

The run's raw `calls.jsonl`/`ledger.jsonl` were re-aggregated independently and
`cash_diff.py` re-run on a fresh copy: both reproduce these figures exactly.

### Receipts (reconciled to the cent)

OpenRouter `byok_usage` 64.211802235 → 65.597736235 = **$1.385934** = health $0.000304 +
run $1.385630 (residual $0.000000). Second BYOK provider balance 5.0 → 5.0 = $0. OpenRouter credits
`total_usage` delta $0.000001642 = the disclosed BYOK fee for the health-plus-run window
(not part of the 70-pair billed-inference figures). Health billed spend $0.000304 is
excluded from the comparison. Run billed inference $1.385630 < the $2.50 cap.

## 3. Route-health gate — probe-budget decision (supervisor-authorised)

The §4.1 route-health probe `max_tokens` was amended **64 → 3000** tokens before this run.
`kimi-k3` is a reasoning route: under a 64-token cap it spends the budget on hidden
reasoning and returns empty *visible* content even when healthy (the prior P6 attempt's
gate failed on it this way at HTTP 200). The run serves each task at its per-category
output budget (3000–12000; `research`/`cache_repeat` = 3000 is the minimum `kimi-k3`
serves), so 3000 is the probe's real minimum. This is a go/no-go PRECONDITION outside the
analysed data: the health rows go to `health/`, never to the run's `calls.jsonl`/
`ledger.jsonl`, and it changes neither the task set, arms, per-task budgets, grading,
registration, nor the analysis. All other gate conditions are unchanged (HTTP 200,
non-empty visible content, < 60 s each, gateway-billed basis for gpt-5.6-sol). The amended
gate PASSED 2026-10-12T18:03:44Z (kimi-k3 2/2 non-empty, gpt-5.6-sol 2/2 non-empty).

## 4. Independent review chain

- Pre-execution (R1, PLAN §7.1): the named initial R1 was BLOCKER; the subsequent convergent
  re-review of the final `df055fcb` cash script returned **CLEAN** (4 October), closing all
  earlier findings. The `df055fcb` script is the reviewed analysis executed here.
- Post-execution (R2, PLAN §7.2): an independent **non-Anthropic** reviewer
  (Codex gpt-6.1-sol, read-only, $0) returned **SUBSTANTIATED** — but every substantive
  check PASSED (raw digests, independent re-aggregation, independent bootstrap to 1e-12,
  live rule joins, health gate, provider receipts, CD1–CD4). The two findings were
  presentation-only:
  - **F1 (LOW):** `report.json:arms_with_no_measurable_cost = ["router"]` and the matching
    `run.log` sentence ("No cash comparison is possible for: router…") are false for this
    mixed arm — the router has 14 billed design calls totalling $0.594188. Root cause is a
    per-arm report-label predicate in the shipped harness that fires when *any* cell lacks
    metered cash; it does not affect any cash figure or the CD1–CD4 analysis. The harness
    source fix is a separate product change needing its own review cycle and is NOT done
    here.
  - **F2 (MEDIUM):** the registered script's generated `allowed_wording` contained a
    payment-benefit assertion ("paid for what measurably benefits from payment") and an
    unfinished pass-rate ellipsis, and `not_allowed` carried a percentage exception
    ("unless L2") looser than this task's unconditional prohibition. No percentage or
    benefit claim was actually asserted in the run's evidence.
- Closure: the supervisor bound a reporting correction addendum (sha256
  `cf000099bbd7b7222e25b0906e540c8e4a559cc6f7544b56bc6bc3c59945353c`, in
  `round3-metered-runs/R2-CORRECTION-ADDENDUM.md`) that supersedes the two presentation
  fields and the generated template wording while preserving every original hashed
  artifact. A narrow independent **non-Anthropic** offline closure review
  (Codex gpt-6.1-sol, $0) returned **CLEAN**: F1 and F2 are closed, all 30 original R3
  files are byte-identical to the pre-addendum inventory, and the addendum is the sole
  new file. Rung cleared from L2-pending-R2 for the binding corrected claim below.

## 5. Authorised claim (binding; §4.6 scope)

> On this registered 70-task set, the router paid for the 14 long-output (design) tasks
> under an explicit operator rule and sent the other 56 tasks to free kimi-k3. Router
> billed inference spend was $0.594188 versus $0.791442 for sending all 70 tasks to
> gpt-5.6-sol. The paired difference was $0.197254 (95% CI $0.081624–$0.281017). Of that
> difference, $0.196794 came from the 56 free-routed pairs (95% CI $0.164404–$0.228941);
> the remaining $0.000460 came from the 14 design pairs, which used the same model on both
> arms and is sampling variation. Descriptive pass counts: design 9/14 vs 10/14 and
> summarisation 11/12 vs 9/12; coding, math, research and cache_repeat were each 11/11 on
> both arms. Design grading is a structural proxy.
>
> The OpenRouter BYOK fee $0.000001642 is disclosed separately for the health-plus-run
> window and is not part of the 70-pair billed-inference figures (health billed spend
> $0.000304 is excluded from the comparison).

**Not authorised** in any publication or video for this run: any percentage-saving claim
(unconditional — the template's "unless L2" exception is voided); "saves money / cheaper
overall / lower total spend"; any claim that F_expected chose to pay; any payment-benefit
assertion ("paid for what measurably benefits"); any quality-superiority claim; anything
from replay or list prices. Pass-rate figures are descriptive only, never a quality claim.

Publication and the video gate remain separately gated: this anchor records that the
difference is measurable on the registered set; it is not a publication approval.
