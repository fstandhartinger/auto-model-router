# Experiments (17 Sep 2026)

All numbers below were measured for this repository: a graded task set run against eight
models, a traffic profile of one week of our own coding-agent sessions (metadata only), a
simulator that replays those sessions through the router's policy code, and a live run of
the router server. Paid API spend for everything: about $20.

Scripts: `experiments/tasks.py` (task set), `run_matrix.py` (quality matrix),
`traffic_stats.py` (traffic profile), `simulate.py` (policy replay), `jev_eval.py`
(classifier and judge), `calibrate.py` (measured success table). The task set is built
from public datasets at run time and is not redistributed.

## 1. Quality matrix

78 automatically graded tasks, five categories × three difficulty levels:

| category | easy | medium | hard | grading |
|---|---|---|---|---|
| coding | HumanEval+ (6) | CodeContests rated 1300–1700 (6) | CodeContests ≥ 2100 (6) | hidden tests; a reference solution must pass them |
| math | GSM8K (6) | MATH-500 level 5 (6) | AIME 2025 (6) | exact final answer |
| tool use | simulated order system: 1 action (6) | refund + message (6) | conditional swap across orders with stock rules (6) | final database state vs reference solver |
| long context | 1 needle in ~12k tokens (4) | two-hop lookup in ~40k (4) | aggregation over ~80k (4) | exact answer |
| agentic coding | fix a seeded bug in a file (4) | solve a CodeContests medium problem with a tool loop (4) | same, hard (4) | hidden tests after the loop |

Output limit 16k tokens per call; agent loops up to 30 steps and ~8 minutes.

Solved tasks per model:

| cell | Qwen3.8 27B | DeepSeek V4 Flash | GLM-5.3 Flash | GPT-5.6 Luna | Kimi K3 | GLM-5.3 | GPT-5.6 Sol | Claude Opus 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coding easy | 6/6 | 6/6 | 6/6 | 5/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| coding medium | 3/6 | 3/6 | 5/6 | 3/6 | 6/6 | 3/6 | 6/6 | 6/6 |
| coding hard | 0/6 | 2/6 | 1/6 | 3/6 | 3/6 | 1/6 | 4/6 | 2/6 |
| math easy | 5/6 | 5/6 | 6/6 | 5/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| math medium | 5/6 | 6/6 | 6/6 | 6/6 | 5/6 | 6/6 | 6/6 | 6/6 |
| math hard | 5/6 | 3/6 | 4/6 | 5/6 | 5/6 | 5/6 | 6/6 | 6/6 |
| tool use easy/medium | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| tool use hard | 6/6 | 4/6 | 6/6 | 6/6 | 6/6 | 5/6 | 6/6 | 6/6 |
| long context easy | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 3/4 | 4/4 | 2/4 |
| long context medium | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| long context hard | 3/4 | 1/4 | 3/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| agentic easy | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| agentic medium | 4/4 | 3/4 | 3/4 | 4/4 | 4/4 | 2/4 | 4/4 | 4/4 |
| agentic hard | 0/4 | 1/4 | 0/4 | 3/4 | 2/4 | 0/4 | 3/4 | 3/4 |
| **total** | **61** | **58** | **64** | **68** | **71** | **61** | **75** | **71** |
| list-price cost of the run | $1.26* | $0.19* | $0.24 | $0.25 | $4.80* | $2.28 | $1.98 | $9.08 |

\* ran on a free tier; list cost estimated from tokens. Opus 5 ran with a 6k thinking budget,
GPT-5.6 models at their default effort. Two Opus "long context easy" misses are format misses
(answer given without the requested `\boxed{}`).

What this says:

- **Easy work is solved by everything.** Tool use, easy coding, easy math and needle retrieval
  do not separate a $0.20 model from a $5 one. The differences live in hard coding, hard agent
  loops and hard math.
- **Price and capability are only loosely related.** GPT-5.6 Sol at medium effort solved the
  most (75/78) for a fifth of Opus 5's cost. GLM-5.3 costs ten times GLM-5.3 Flash and solved
  fewer tasks. GPT-5.6 Luna solved 68 for $0.25.
- **Headline benchmark indexes mis-rank specific variants.** The benchmark API rates Luna at
  medium effort far below Kimi K3 on agentic work; on our tasks Luna solved 3/4 hard agentic
  tasks. Per-variant category scores were worse still (a 27B model above Opus 5 on "coding").
  This is why the router reads headline indexes by default and accepts a measured table.

Cell sizes are small (4–6 tasks). The simulator smooths each cell towards the mean of all
models for that cell and everything below is cross-validated on a two-way task split.

## 2. Cache behaviour, measured

Prefix-cache read share on consecutive calls of the same task (calls with ≥ 1024-token prefixes):

| route | read share |
|---|---:|
| GPT-5.6 Sol, GPT-5.6 Luna (hosted APIs) | 0.98–0.99 |
| GLM-5.3 / GLM-5.3 Flash (EU host) | 0.92–0.98 |
| DeepSeek V4 Flash (free tier) | 0.78 |
| Kimi K3 (free tier) | 0.58 |
| Qwen3.8 27B (free tier) | 0.00 |
| Claude Opus 5 via an aggregator, no `cache_control` markers | 0.00 (1.00 with markers) |

From one week of our own agent logs, read share by gap since the previous call:

| client | < 60 s | 1–5 min | 5–10 min | 10–60 min |
|---|---:|---:|---:|---:|
| Claude Code (subscription) | 0.976 | 0.975 | 0.982 | 0.984 |
| OpenCode on free-tier models | 0.887 | 0.821 | 0.725 | 0.570 |

Findings: caching is a property of the route, not the model — measure it per endpoint.
Aggregators may need explicit cache markers. Claude Code on a subscription effectively keeps
a one-hour cache (reads stay at 98 % after 10–60 minute pauses), so a replay that assumes the
5-minute API default overstates its cost by 57 %; with a one-hour TTL the replay reproduces the
real list-price cost of the week within 5 % ($3,369 simulated vs $3,201 from the logs).

## 3. Traffic profile (7 days, metadata only)

| | Claude Code | Codex | OpenCode | all |
|---|---:|---:|---:|---:|
| sessions | 311 | 235 | 1,092 | 1,638 |
| API calls | 20,269 | 12,275 | 25,152 | 57,696 |
| user turns per session (mean) | 2.0 | 1.0 | 1.2 | 1.3 |
| calls per user turn (median / mean) | 15 / 33 | 25 / 52 | 8 / 19 | 10 / 26 |
| prefix tokens per call (median / p90) | 166k / 514k | 92k / 202k | 80k / 186k | 103k / 281k |
| output tokens per call (median) | 555 | 177 | 260 | 308 |
| gap between calls (median) | 10 s | 9 s | 15 s | 12 s |
| gaps longer than 5 min | 4.4 % | 0 % | 1.4 % | 2.2 % |
| input read from cache | 98.7 % | 97.3 % | 89.3 % | 95.7 % |

Consequences for routing:

1. **Almost all tokens are cached prefix re-reads inside tool loops.** Switching model inside a
   loop throws away a cache that is seconds old on a 100k+ prefix. The router therefore decides
   only at user-turn boundaries and escalates inside a loop only after repeated failing tool
   results.
2. **Sessions are short in user turns** (1.3 on average). Rules about when to *downgrade* a warm
   conversation (the original expected-value switch rule) rarely get a chance to act: in the
   replay, policy C behaved exactly like naive per-turn routing. The levers that matter are the
   starting model, escalation on failure, and plan pacing.
3. **Cache expiry between turns is rare** (2 % of gaps exceed 5 minutes), so "downgrade once the
   cache expired" is a minor effect on this traffic.

Plan usage: one percent of a weekly Claude plan quota corresponded to about $38 of list-price
usage in the same window (a lower bound; other clients on the account also count). For the
Codex plan used here, one percent was about $1.10.

## 4. Jev as classifier and judge

Classifier on the 78 task prompts (one call, ~0.65 s, ~1.4k input tokens):

- category: 78/78 plausible (agentic tasks labelled "coding", every other category exact)
- difficulty: Pearson r = 0.74 with the task level; Jev compresses the scale
  (`d_jev ≈ 0.27 + 0.51 × level`), residual noise ≈ 0.22 level units. The router maps it back
  with `policy.jev_difficulty_calibration: [0.27, 0.51]`.
- Jev's difficulty predicts which tasks a cheap model fails better than the task label does:
  AUC 0.78 (Luna), 0.82 (DeepSeek V4 Flash), 0.85 (GLM-5.3 Flash), 0.90 (Kimi K3).

Adequacy judge ("does this response fully and correctly address the request?") on 96 answers
from two free models:

| answers | AUC | flag if p < 0.3 |
|---|---:|---|
| coding (36) | 0.96 | catches 85 % of wrong answers, 0 % false flags |
| math (36) | 0.99 | catches 67 % of wrong answers, 0 % false flags |
| long-document questions (24) | poor | the judge cannot see the document: false flags |

Use the judge as an escalation signal only for self-contained requests.

## 5. Policy comparison (session replay)

`simulate.py` replays all 1,638 sessions in time order. Each call keeps its real prefix size,
output size and gap; output is scaled by each model's measured verbosity; cache reads follow the
route's measured hit rate and TTL; the Claude plan is paced live over the week (25 % background
interactive use, reserve line 65 %). Turn difficulty comes from a proxy (calls in the turn:
≤ 8 easy, ≤ 40 medium, more hard), the policy sees it with noise 0.15, success is drawn from the
matrix, a failure is noticed with probability 0.6 and may be retried. Beliefs come from one half
of the tasks, ground truth from the other half, and the two folds are averaged.

**Our deployment** (free-tier models, metered Luna/GLM/Sol/Opus APIs, Claude plan only for
Claude Code's own sessions):

| policy | solved | paid API $/week | $ per solved turn | Claude plan used (sim) | switches |
|---|---:|---:|---:|---:|---:|
| today: Claude Code on plan, Codex on plan, OpenCode on free models | 91.6 % | 0 | 0 | 127 % | 43 |
| A: Opus 5 API for everything | 92.8 % | 12,917 | 6.42 | 0 % | 0 |
| B: cheapest model predicted ≥ 80 % | 91.0 % | 5,110 | 2.59 | 0 % | 268 |
| C: B + expected-value downgrade rule | 91.0 % | 5,110 | 2.59 | 0 % | 268 |
| D: start cheap, escalate, downgrade on expiry or clear savings | 90.4 % | 3,443 | 1.76 | 0 % | 282 |
| E: D + paced plan tier | 90.2 % | 1,939 | 0.99 | 43 % | 232 |
| **F: minimise expected cost (call + failure × retry or stakes), paced plan** | **91.0 %** | **485** | **0.25** | **24 %** | 361 |

The simulator over-counts plan usage for "today" (127 % vs about 95 % in reality, mostly
retries), so read the plan column relatively.

Smaller catalogues for F (same replay):

| F with | solved | $/week | plan used |
|---|---:|---:|---:|
| everything | 91.0 % | 485 | 24 % |
| no Opus API, no GLM-5.3 | 89.9 % | 158 | 25 % |
| free tier + Luna only | 90.0 % | 147 | 25 % |
| free tier + plan only | 88.3 % | 0 | 31 % |

**Public list prices, no subscription:**

| policy | solved | $/week | $ per solved turn |
|---|---:|---:|---:|
| A: Opus 5 for everything | 92.8 % | 12,917 | 6.42 |
| B: naive cheapest capable | 88.7 % | 5,472 | 2.85 |
| C: expected-value switch | 88.7 % | 5,478 | 2.85 |
| D: cheap first, escalate | 87.7 % | 2,995 | 1.58 |
| F: expected cost, stakes 2× turn cost | 89.0 % | 1,341 | 0.70 |
| F: expected cost, stakes 10× turn cost | 90.0 % | 2,236 | — |

Sensitivity of F (our deployment):

| change | solved | $/week | plan used |
|---|---:|---:|---:|
| baseline | 91.0 % | 485 | 24 % |
| failures noticed 30 % of the time | 89.7 % | 324 | 26 % |
| failures noticed 90 % of the time | 92.0 % | 742 | 40 % |
| perfect difficulty estimate | 91.1 % | 244 | 15 % |
| difficulty noise 0.3 | 90.7 % | 749 | 24 % |
| stakes 0.3× turn cost | 88.9 % | 331 | 30 % |
| stakes 10× turn cost | 91.2 % | 750 | 28 % |
| **beliefs from benchmark headlines only, no local calibration** | **88.3 %** | 1,431 | 39 % |

What the comparison shows:

- **F wins on every catalogue and price model.** It is the only policy that matches today's
  success while cutting both paid spend and plan use sharply, and it stays ahead under every
  sensitivity setting.
- **"Start cheap and escalate" (D) is right in spirit but prices failure badly.** Always starting
  on the cheapest adequate model and escalating to the strongest one sends hard turns through two
  expensive attempts. F starts hard turns directly on the model with the best success per dollar
  and keeps easy ones on free models.
- **Calibration beats cleverness.** Replacing measured success rates with benchmark-derived
  ones costs about 3 points of success and triples spend. A 78-task calibration run cost about
  $20.
- **The cache matters through the choice of route, not through switching rules.** Luna's
  $0.02/M cached input is why it absorbs most long-prefix turns under F.

## 6. Live validation

The router server ran with the measured success table and Jev, one instance per policy, on 30
tasks (all 12 agentic tasks with a follow-up user turn, hard tool use, medium and hard coding),
routing across three free-tier models, GLM-5.3 Flash, Luna and Sol:

| policy | solved | paid $ (list) | $ per solved | calls by model |
|---|---:|---:|---:|---|
| F | 23/30 | 0.31 | 0.014 | Qwen3.8 62, Kimi K3 42, Sol 12 |
| D | 24/30 | 0.30 | 0.013 | Kimi K3 106, Sol 12 |
| C | 25/30 | 0.30 | 0.012 | Kimi K3 106, Sol 12 |

All three kept tool loops on one model (no mid-loop switches), sent hard competitive
programming to Sol and everything else to free models. At 30 tasks the difference between them
is within noise. The live run confirms that the policies behave as simulated and cost about a
cent per solved task on this mix; it cannot rank them.

---

# Cycle 2 (18 Sep 2026): evidence, separation and a pre-registered held-out set

The first cycle asked "does routing pay?". This one asks a narrower question that
the first could not answer honestly: **is the router's belief about a route
actually evidenced, and can you tell an estimate from a measurement afterwards?**

## 7. Specialisation is read from evidence, not asserted

`bench.capability_evidence()` now returns, per category, a value *and* the basis
it rests on *and* how strong that basis is (`direct`, `derived`, `weak`). Two
categories were added: `design` (web/UI) and `summarisation`.

`design` comes from the `designarena` block the benchmark API serves —
frontend and fullstack Elo with battle counts — mapped onto the 0–100 axis with
a documented affine transform (1200 Elo ≡ 50, 400 Elo ≡ 100 points), and marked
`weak` below 200 recorded battles. Read on 18 September 2026:

| route | coding | design | design basis |
|---|---:|---:|---|
| kimi-k3::max | 76.20 | **80.12** | designarena elo 1320 over 2813 battles |
| claude-opus-5::high | 76.50 | 73.88 | designarena elo 1296 over 2573 battles |
| gpt-5.6-sol::medium | 76.30 | **46.50** | designarena elo 1186 over 3866 battles |
| gpt-5.6-luna::medium | 50.70 | 50.70 | *fallback:* aa_coding_index (derived) |

Three routes within 0.3 points of each other on coding spread across 46.5–80.1
on design. That is the whole argument for a specialised route: the design rank
is not recoverable from the coding rank. It is also why the router must not
hard-code a name — the ranking is a property of data that moves.

`summarisation` has no dedicated benchmark in this feed. It is derived from
long-context, knowledge and the intelligence index, and is reported as
`derived` **every time**, never as a measurement.

Four of the seven routes carry no design data at all (`designarena: {}`), so the
fallback path is exercised in practice, not just in a test.

### The evidence discount

`policy.success.evidence_discount` (0 by default, `0.5` in the held-out config)
shrinks a capability score toward a neutral prior of 50 in proportion to how
weak its evidence is — `direct` 1.0, `derived` 0.6, `weak` 0.4, `none` 0.0 —
and a stale benchmark document demotes every strength by one step. The effect is
that a cheap route cannot win a specialised task on a number nobody measured.
`test_the_evidence_discount_makes_a_thin_cheap_route_less_attractive` pins the
behaviour in both directions.

## 8. Four kinds of statement, kept apart

`auto_router/decision.py` splits every decision into `classification` (what the
task is), `selection` (what the router decided and why), `estimated_outcome`
(what it expected) and `observed_outcome` (what happened). They are served at
`GET /v1/router/decisions` and appended to a JSONL ledger.

Two rules are enforced by tests rather than by convention:

- **A cost with no measurement basis is `null` plus the reason, never `0`.** A
  free or subscription route must not later read as a measured saving.
- **The record contains no prompt text.** Not the request, not the response, not
  tool arguments, not a credential.

## 9. Pre-registered held-out evaluation

`experiments/heldout.py preregister` writes 27 tasks across the six categories
the next cycle asked for — web/UI design, coding, maths/reasoning, factual
research, summarisation and cache-eligible repeats — plus the analysis plan,
the stopping rule, the exclusion rule and a list of claims that will *not* be
made, and takes a SHA-256 of the task file. The runner refuses to start if that
digest has changed. All of that happens before a single model is called.

Graders are deterministic and none of them calls a model, so a run can be
regraded from stored answers with identical verdicts. Each one declares what it
measures:

| category | grader | what it really is |
|---|---|---|
| coding | `executed` | hidden tests run in the Bubblewrap sandbox |
| math | `exact` | the model's own stated final answer vs the known value |
| research | `exact` | required fact present, named confusion absent |
| summarisation | `rubric` | inside the length bound, keeps the required facts, invents no number absent from the source |
| design | **`structural-proxy`** | real self-contained markup meeting the rules the prompt stated — *not* a judgement that the design is good |
| cache_repeat | `exact` | four questions over one shared 5k-character prefix |

The `structural-proxy` label travels into every ledger row and into the report,
so a design number can never be read as more than it is.

## 10. Executing generated code without Docker

The 18 September run stopped rather than execute model-written code without
isolation, because the Docker socket is deliberately unreachable. This cycle
uses Bubblewrap instead (`experiments/sandbox.py`): read-only host, private
tmpfs for `/tmp`, `$HOME` and the working directory, all namespaces unshared
including the network, all capabilities dropped, an empty environment, and
CPU/address-space/file-size limits. Verified on this host:

```
{"network": "blocked:OSError", "home_readable": false, "secret_env": [],
 "cwd": "/work", "writable_work": true}
```

`sandbox.preflight()` is called before grading. If it does not confirm real
isolation, the coding tasks are **excluded from the results with the exact
reason recorded** — the harness never falls back to running a generated answer
on the host.

## 11. The live held-out run (18 Sep 2026)

Three arms on the same pre-registered tasks, identical prompts, one run.
`router` is policy F choosing per task. `control` is one capable route for
everything, picked by the same capability data the router uses — it resolved to
**kimi-k3, which is free in this catalog**. `control-metered` is **gpt-5.6-sol**,
added on the record once the first control turned out to cost nothing, so that a
billed figure exists at all.

> **No saving was measured and none is claimed.** The router and the free
> control both spent $0.0000; there is no cash difference between them to
> report. The paid control spent real money for pass rates that are
> indistinguishable at this sample size. Every category is below the
> pre-registered ten-task floor for a quality claim, which is why the Wilson
> intervals are printed — they overlap completely.

<!-- heldout:start -->
| category | arm | n | passed | pass rate | Wilson 95 % | measured USD | grader |
|---|---|---:|---:|---:|---|---:|---|
| design | router (policy F) | 2 | 2 | 1.00 | 0.34–1.00 | $0.0000 (free route) | structural-proxy |
| design | control · kimi-k3 (free) | 2 | 2 | 1.00 | 0.34–1.00 | $0.0000 (free route) | structural-proxy |
| design | control · gpt-5.6-sol (metered) | 4 | 3 | 0.75 | 0.30–0.95 | $0.2071 | structural-proxy |
| coding | router (policy F) | 4 | 4 | 1.00 | 0.51–1.00 | $0.0000 (free route) | executed |
| coding | control · kimi-k3 (free) | 4 | 4 | 1.00 | 0.51–1.00 | $0.0000 (free route) | executed |
| coding | control · gpt-5.6-sol (metered) | 4 | 4 | 1.00 | 0.51–1.00 | $0.0231 | executed |
| math | router (policy F) | 6 | 5 | 0.83 | 0.44–0.97 | $0.0000 (free route) | exact |
| math | control · kimi-k3 (free) | 6 | 6 | 1.00 | 0.61–1.00 | $0.0000 (free route) | exact |
| math | control · gpt-5.6-sol (metered) | 6 | 6 | 1.00 | 0.61–1.00 | $0.0177 | exact |
| research | router (policy F) | 5 | 5 | 1.00 | 0.57–1.00 | $0.0000 (free route) | exact |
| research | control · kimi-k3 (free) | 5 | 5 | 1.00 | 0.57–1.00 | $0.0000 (free route) | exact |
| research | control · gpt-5.6-sol (metered) | 5 | 5 | 1.00 | 0.57–1.00 | $0.0017 | exact |
| summarisation | router (policy F) | 4 | 3 | 0.75 | 0.30–0.95 | $0.0000 (free route) | rubric |
| summarisation | control · kimi-k3 (free) | 4 | 4 | 1.00 | 0.51–1.00 | $0.0000 (free route) | rubric |
| summarisation | control · gpt-5.6-sol (metered) | 4 | 4 | 1.00 | 0.51–1.00 | $0.0132 | rubric |
| cache_repeat | router (policy F) | 4 | 4 | 1.00 | 0.51–1.00 | $0.0000 (free route) | exact |
| cache_repeat | control · kimi-k3 (free) | 4 | 4 | 1.00 | 0.51–1.00 | $0.0000 (free route) | exact |
| cache_repeat | control · gpt-5.6-sol (metered) | 4 | 4 | 1.00 | 0.51–1.00 | $0.0032 | exact |

**Measured spend over the whole set, by arm:** router (policy F) **$0.0000** · control · kimi-k3 (free) **$0.0000** · control · gpt-5.6-sol (metered) **$0.2659**

Graded rows 77, excluded 4. Task set `84731010531b266f` registered 2026-09-18T06:42:03Z, 3 recorded amendment(s), no drift.
<!-- heldout:end -->

Two things this does show:

1. **The router routes by category.** All four summarisation tasks and half the
   maths went to the cheaper `dsv4-flash`; research and design stayed entirely on
   `kimi-k3`; the cache-repeat set went to `qwen3.8-27b`; coding split three to
   one in `kimi-k3`'s favour. Those are per-category decisions taken from
   evidence, not one fixed model.
2. **Both of its failures were on the cheaper route it downgraded to** — with
   no cash saving to weigh against them, because both routes are free. At 4–6
   tasks per cell that is well inside noise. The honest reading is not that the
   router is worse; it is that this run cannot show it is better.

**The design category is only *fully* measurable on the metered arm.** The easy
design task completed on kimi-k3 on both free arms. The two harder ones did not:
`design-medium-form` and `design-hard-dashboard` ran for 305 s and 377 s and hit
the 12,000-token output budget without finishing the page. gpt-5.6-sol answered
the same prompts in roughly 2,200 tokens and 18 s. The truncated rows are
excluded as harness failures rather than scored as model failures, which is the
pre-registered rule — but the pattern is not noise, and it is the difficulty
that separates them.

A router that reads only capability scores cannot see that at all: both routes
look capable, and design-arena Elo puts kimi-k3 *above* gpt-5.6-sol on exactly
this kind of work. It shows up only as an *observed* property of the route on
harder instances.

**This is now the one thing the router does about it.** A completion the
provider itself flags as a length stop is recorded as `status: "truncated"` —
an observed failed attempt, not a success — and the non-streaming OpenAI
surface takes the same sideways safe fallback a 5xx takes, for one extra route
by default. Nothing is written back into a capability score, and the flag is
read only from the provider's own machine-readable field, never from the answer
text. See `auto_router/truncation.py` and the README section *An answer the
provider says it never finished*.

What that does **not** do is turn the observation above into a claim. These
numbers are still four excluded rows on two tasks in one run. The change makes
the router able to *see* the failure and survive it; whether falling back on a
length stop produces better answers or lower cost across a real workload is
unmeasured, and the held-out set at its current size cannot measure it. The
evidence for the behaviour is the deterministic local suite
(`tests/test_truncation.py`, 41 tests) and the loopback HTTP smoke test, not a
live comparison.

The same failure appeared somewhere else entirely, which is why it looks like a
route property rather than a task artefact. Two reasoning routes were also asked
to review this change set: each spent its *whole* output budget on reasoning and
returned zero characters, twice, at 6,000 and again at 16,000 output tokens —
four calls and about 57,000 reasoning tokens for no output at all. "Capable but
unable to finish within a budget" is not something a capability index measures,
and it costs real time and real tokens.

Reproduce:

```sh
python experiments/heldout.py preregister
python experiments/heldout.py verify
python experiments/heldout.py run --config my.local.yaml --budget 4.00
python experiments/heldout.py run --config my.local.yaml --budget 4.00 \
    --control <a-metered-route> --control-label control-metered --arms control
python experiments/heldout.py report
```

## 12. The pre-registered supplement: ten valid pairs in every category (18 Sep 2026)

The run in §11 could not answer its own question, for one arithmetic reason: the
unit of evidence in a paired comparison is the **pair**, a pair needs *both*
rows graded, and four of the twelve design rows were harness truncations. Every
category sat under the pre-registered floor of ten.

The 27-task ledger is finished evidence and was not touched. A **second,
separately pre-registered supplement** of 60 tasks was registered before any
call — `runs/heldout-supplement-<ts>/preregistration.json`, task digest
`b55743c99d41fac9…`, policy/catalog identity digest `56ca0ebd84fd9fe6…` — and
the two are *combined for reporting* while staying distinguishable.

The supplement registration freezes something the first one did not: **what the
routing arm actually is.** `policy_identity` records the resolved policy name
(`F_expected`, the router's own default, because the config sets no
`policy.name`), its settings, and for every route the provider, upstream id,
prices, per-category capability and staleness. The runner refuses to start if
any of that, the task file, the analysis plan, the config file, the experiment
code or `auto_router/*.py` has moved without an amendment.

### The result

<!-- supplement:start -->
| category | valid pairs | router | control (free) | control-metered | paired diff vs control, conservative 95 % | sign p |
|---|---:|---:|---:|---:|---:|---:|
| design | 22 / 23 | 19 / 20 | 20 | 21 | −0.045 (−0.228…+0.217) | 1.000 |
| coding | 12 | 12 | 12 | 12 | 0.000 (−0.265…+0.265) | n/a |
| math | 13 | 11 | 13 | 13 | −0.154 (−0.455…+0.311) | 0.500 |
| research | 12 | 12 | 12 | 12 | 0.000 (−0.265…+0.265) | n/a |
| summarisation | 11 / 12 | 7 / 8 | 9 | 11 | −0.182 (−0.683…+0.423) | 0.625 |
| cache_repeat | 12 | 12 | 12 | 12 | 0.000 (−0.265…+0.265) | n/a |

Every category now reaches **ten valid pairs against both comparators**. The
intervals are printed because they are wide: even a category where the two arms
agreed on all twelve pairs still admits a difference of a quarter in either
direction. "No disagreement observed" is not "equal".

**No category supports a quality claim, and that is structural.** "At least as
often" is a non-inferiority statement, and **no non-inferiority margin was
pre-registered**. Choosing one now, with the numbers in hand, is exactly the
move the pre-registration exists to prevent. The report therefore reports an
empty supported list by construction and says why, instead of promoting a tie
into a result.
<!-- supplement:end -->

**What that buys, stated exactly.** The routing policy is **never ahead** in any
category. It ties in coding, research and cache_repeat — identically, on every
pair — and it is **behind** in design, maths and summarisation, losing 1, 2 and
3 discordant pairs respectively. None of those deficits is significant at this
size (sign test p = 0.25–1.00), and a large p-value here is *absence of
evidence against equality*, not evidence of equality. The one thing this does
establish is a ceiling: a routing advantage large enough to matter at these
sample sizes is not there.

**And still no saving.** The routing arm and the free control both spent
**$0.0000**, because the policy keeps choosing free routes — which is the
correct decision and is also why there is no cash contrast to weigh the
deficits against. The metered comparator spent **$0.9457** on retained rows
(**$1.1095** counting the calls whose rows were later discarded and re-run) for
pass rates that are equal or one pair better. That is a real billed-adjacent
figure — the gateway's reported upstream inference cost, not an invoice.

**The routing is real, and visible per category:** `qwen3.8-27b` for every
cache-repeat, `dsv4-flash` for all summarisation and part of maths and coding,
`kimi-k3` for design and research. Both of the maths losses and three of the
four summarisation losses are on `dsv4-flash`, the cheaper route the policy
downgraded to. The pattern from §11 held at four times the sample size.

### The honest caveats, in the report itself

The combined report prints nine of them next to the numbers rather than in a
footnote. The three that change how the table should be read:

- **The supplement is an adaptive sample.** Its size and per-category mix were
  chosen after the original outcomes were known — that is what "bring every
  category to ten valid pairs" requires. No rule, grader, prompt or arm moved;
  the counts did.
- **19 of 22 design tasks are compact single components**, because the original
  larger pages are exactly what the free routes could not finish. Design is
  therefore reported split by the registered difficulty tier, and the split is
  the interesting part: easy 12/13 vs 13/13, medium 7/7 vs 7/7, **hard (full
  pages) 0/2 vs 0/2 with two of four pairs truncated.** The failure mode is
  still there; it is not averaged away.
- **`cache_repeat` is eight questions over one warm prefix**, so its twelve
  pairs are twelve observations and nothing like twelve independent tasks. It
  is flagged `independent_samples: false` everywhere it appears.

Because the original run's exclusion rule ("a truncated answer is excluded, not
failed") was written *after* a truncation had been seen, every category also
carries the same numbers under the opposite rule. The conclusion does not move:
design 20/26 vs 21/26, summarisation 8/12 vs 9/12 — the routing policy is level
or behind either way.

### What a reader can check, and what only the operator can

Start with the part that is easy to read past: **the run directories are not in
this repository.** `experiments/runs/` is in `.gitignore`, and the two runs
behind every number above — `runs/heldout` and
`runs/heldout-supplement-20260918T1021Z` — live on the operator's machine. What
is published is the code that produced them, the numbers, and the digests quoted
in this file and in the commit. A reader can therefore re-run the *harness*, and
can check any run directory they are given against this checkout; a reader
cannot, from a clone alone, verify the runs that produced the table above.

Given the run directory, the split is sharper still. Everything below the first
line of the recipe needs `my.local.yaml`: the runtime config, carrying provider
credentials, which is not in this repository and cannot be reconstructed from the
registration. That is not an oversight that can be patched — it is what the
strict verifier is *for*, and it refuses rather than pretending when the file is
absent:

```sh
$ python experiments/supplement.py verify --dir runs/heldout-supplement-<ts> --config anything-else
the registered config '...' is missing, so the policy/catalog identity cannot be
re-checked; refusing to run                                               # exit 1
```

So the supplement's configuration identity is **verifiable only by whoever holds
that exact file**, and no claim that it was independently verified should be
made on a reader's behalf. What a reader *can* re-derive, on any machine, with
no config at all, is the run's public identity:

```sh
python experiments/evidence_verify.py --dir runs/heldout-supplement-<ts>
```

That checks the task file, the registered task ids and per-category counts, the
analysis plan, the frozen policy/catalog identity against the digest it carries,
and every registered experiment- and product-code digest against the checkout —
and then prints, every time, that none of it proves the secret-bearing runtime
config ever existed, hashed as recorded, or produced that identity. Offline
verification of a frozen redacted identity is not proof of the original.

Two things it turns up that were not visible before, both about the human-readable
analysis plan rather than any collected row:

- In the **supplement** registration the plan *text* is the version first
  registered, while the enforced `analysis_plan_sha256` is the amended one —
  `amend` refreshes the digest and leaves the prose alone. The trail links them
  (the text hashes to the `previous_analysis_plan_sha256` every amendment
  records), so it verifies, with that stated as a limitation.
- In the **original 27-task** registration it does **not** verify. That run's
  first amendment recorded `previous_analysis_plan_sha256: null`, because the
  registration had no plan digest until the amendment added one. The plan text
  sitting in that file hashes to `4e2cee61…` and nothing in the record ties it to
  the enforced `0fa59644…`. `heldout.py verify` passes — it only compares the
  digest to the checkout — but the prose and the digest in that registration are
  not demonstrably the same plan, and freezing it after the fact would be
  tampering, so it is reported rather than repaired.

A run registered from now on should freeze a credential-free projection of its
config as well, which is the only thing that closes the configuration gap for a
*future* run:

```sh
python experiments/evidence_verify.py --dir runs/<new-run> --freeze-public-config my.local.yaml
```

Rotating a credential leaves that projection's digest unchanged; changing a
route, an endpoint or a policy setting does not.

Reproduce (the operator's path, needing the config):

```sh
python experiments/supplement.py preregister --dir runs/heldout-supplement-<ts> --config my.local.yaml
python experiments/supplement.py verify     --dir runs/heldout-supplement-<ts> --config my.local.yaml
python experiments/supplement.py run --dir runs/heldout-supplement-<ts> --config my.local.yaml \
    --arms router,control --workers 6 --budget 1.00 --cap 30.00 --prior-spend <recorded>
python experiments/supplement.py run --dir runs/heldout-supplement-<ts> --config my.local.yaml \
    --arms control --control <a-metered-route> --control-label control-metered \
    --workers 6 --budget 2.50 --cap 30.00 --prior-spend <recorded>
python experiments/supplement.py report --dir runs/heldout-supplement-<ts> --original runs/heldout
```

## 13. The two subscription paths, run live (18 Sep 2026)

Eight runs on one machine, each n=1. They establish that the paths *work* and
what they record; they measure no saving, rank no policy and compare no cost.
Client versions: Claude Code 2.1.270, Codex CLI 0.154.0, OpenCode 1.18.18.

### 13.1 The launcher

| # | what was asked | plan state at the time | route chosen | result |
|---|---|---|---|---|
| 1 | "What is 2 + 2? Answer with the number only." | both plans over their hard stop | free model, through its own CLI | answered `4`, exit 0, $0 |
| 2 | a hard refactor-and-test job (dry run) | same | free model | plan routes excluded, reason recorded: "weekly use 94% at or above hard stop 80%" and "weekly use 89% at or above hard stop 75%" |
| 3 | the same job, with a stand-in usage reader reporting 20 % / 25 % | plans open, shadow price x0.00 | **a plan route** (expected $2.18 vs $2.94 for the best free route) | dry run; the tier switch is the point |
| 4 | "Write a Python function merge_sorted_unique(a, b) …", forced with `--route` | Claude plan at 94 % | Claude plan, `claude -p --model opus` | answered with working code in 10.8 s, exit 0 |
| 5 | "Write a Python one-liner that counts unique words …", forced with `--route` | ChatGPT plan at 89 % | ChatGPT plan, `codex exec -m …` | answered in 7.0 s, exit 0, and the record shows `cleared_env: ["OPENAI_API_KEY"]` |

Runs 4 and 5 are the ones that prove the billing path, in two different ways.
This machine has no Anthropic API key at all, so a successful answer from
`claude -p` can only have been served by the signed-in plan. It *does* have an
OpenAI API key exported - which is exactly the trap `clear_env` exists for: the
Codex run would have been billed per token had the variable reached the child,
and the record names the variable that was emptied. Both records carry
`cost_usd: null` with the basis "subscription route: no per-token charge; the
plan's own usage limits apply".

Run 3 used a deliberately fake usage reader, and the record says so: it is a
*configuration* experiment about the pacing rule, not a measurement of a plan.

### 13.2 The gateway

A real Claude Code session against `ANTHROPIC_BASE_URL=http://127.0.0.1:8787`
with no gateway credential set, three times, one tiny prompt each:

| # | mode and catalog | what the router decided | what happened |
|---|---|---|---|
| 6 | default `passthrough_only` | a free route would have done | every turn forwarded to Anthropic unchanged, 200, recorded `not_taken`; the session answered normally |
| 7 | plan open, only plan and metered routes configured | the plan route | forwarded, 200 in 1.5–1.8 s, recorded `ok` with the subscription cost basis |
| 8 | `route_others`, free routes available | a free route | the turn really was served by the free model: 38,849 uncached input tokens, **0 cache reads**, 43 output tokens |

Run 8 is the most informative and the least flattering to the idea it tests. The
turn worked - Claude Code accepted the translated answer - but Claude Code's
prefix is large and the free host cached none of it, so a route that is free in
cash paid full price in tokens and latency on every turn. That is the measured
version of the point the replay made in §5: caching is a property of the route,
and the plan's own cache is a large part of what the plan is worth. It is also
the configuration Anthropic says it "doesn't support", which is why it is off
unless asked for.

The server log and the ledger were grepped for credential-shaped strings after
every run: nothing. `redact()` covers the log lines, and decision records carry
no headers at all.

### 13.3 Two bugs these runs found

Both were invisible to the test suite and obvious within a minute of running the
thing against a real machine:

1. **One usage reader, two plans, one cache entry.** The command-backed quota
   reader cached its result by command only. Both subscriptions were configured
   with the same reader, so the second plan was paced with the first plan's
   numbers - the ChatGPT plan was reported as 94 % full when it was at 89 %.
   Fixed by putting the plan name in the cache key, with a regression test.
2. **A plan offered on a surface it cannot serve.** With both plans open, the
   gateway chose the *ChatGPT* plan for a Claude Code turn. Nothing broke,
   because subscription traffic is forwarded unchanged either way, but the
   decision was meaningless: that plan is reachable only through its own CLI.
   Routes now declare `launch_only`, and the HTTP catalog drops them.

A third, quieter one came from the same session: a subscription route named
after the plan rather than the model missed the measured success table entirely
and was priced off the fitted curve, two tiers below where 78 graded tasks had
put it. Routes now carry a `success_key`, so a plan route and a metered route to
the same model share its measurements.

### 13.4 What these runs do not show

- **No before/after plan usage.** The usage endpoint behind the reader answered
  `rate_limit_error` throughout the experiment window, so the plan's percentage
  could not be sampled on either side of the runs. The evidence that a run was
  billed to the plan is structural (no API key exists on the machine), not a
  measured delta.
- **No saving, no quality comparison.** One run per path, different prompts per
  path, no pairing, no grader.
- **Run 3's plan headroom was simulated**, and the two real plans were near their
  weekly limits all day, so the interesting regime - a plan with room, chosen on
  its merits, for a hard job - was exercised as a decision and not as a job.

# Cycle 3 (18 Sep 2026): checking a cheap answer, and what that does to a chain

## 14. Verify-and-escalate: calibration, cost, and Scott's cascade

Prompted by a reply to the launch post: *"I'd want to see how that holds when
you chain three or four routed calls where a bad early model pick cascades
downstream. Single-hop latency flatters routers."* The router did not grade
answers at all; §4 had measured that Jev *could* grade a cheap one, and nothing
acted on it. This cycle wires it in and measures what it buys.

**Reproduce:**

```
python experiments/verify_calibrate.py answers   --config <cfg> --models qwen3.8-27b,dsv4-flash,kimi-k3,glm-5.3-flash --out runs/verify/answers.jsonl
python experiments/verify_calibrate.py judge     --answers runs/verify/answers.jsonl --out runs/verify/judged.jsonl
python experiments/verify_calibrate.py escalate  --judged runs/verify/judged.jsonl --answers runs/verify/answers.jsonl --config <cfg> --threshold coding=0.25,math=0.60 --carry both --out runs/verify/escalated.jsonl
python experiments/verify_calibrate.py report    --judged runs/verify/judged.jsonl --escalated runs/verify/escalated.jsonl --out runs/verify/report.json
python experiments/cascade.py --config <cfg> --success runs/success.json --calibration runs/verify/report.json --latency runs/verify/latency.json --out runs/verify/cascade.json
```

### 14.1 What was measured

192 answers: four cheap routes (Qwen3.8 27B, DeepSeek V4 Flash, Kimi K3, all
free; GLM-5.3 Flash, metered) on the 48 single-shot tasks of the 78-task set -
18 coding, 18 maths, 12 long-document. Each answer graded against ground truth
by the same deterministic graders as §1, then judged by Jev with a **typed
question pair** it had not been asked before: the adequacy Noul with
*task-specific* criteria, plus a Choice naming the failure (`wrong`,
`incomplete`, `off_topic`, `fine`). Both questions in one call.

46 of the 192 answers were wrong. Judge: `jev-1.13.0`, 1,395 input and 66 output
tokens on average, **median 0.69 s, p90 0.78 s**, 0 failures in 192 calls.

| answers | n | wrong | AUC | shipped threshold | catches | false alarms |
|---|---:|---:|---:|---:|---|---|
| coding | 72 | 33 | 0.948 | **0.25** | 85 % (28/33) | 10 % (4/39) |
| maths | 72 | 11 | 0.955 | **0.60** | 73 % (8/11) | **0 %** (0/61) |
| long-document | 48 | 2 | 0.538 | *not checked* | — | 96 % at 0.3 (44/46) |

The two thresholds differ because the score distributions do. An *adequate*
maths answer never scored below 0.77, so a high bar there is free; an adequate
coding answer can score anywhere, so the bar has to sit low and still costs
false alarms. **Four of the six coding false flags at 0.3 are the same task**
(HumanEval/149, `sorted_list_sum`), whose written specification contradicts its
own hidden tests - the judge reads the specification, the grader runs the
tests, and they disagree. The rate is reported as measured, not adjusted for
that.

The long-document row is the negative control and it reproduces §4 exactly: the
judge is not shown the document, so it rejects almost every adequate answer.
That is why `long_context` is skipped rather than given a threshold.

### 14.2 What an escalation buys

Every flagged answer was re-run at the threshold above, on the route **the
router itself** would escalate to (the cheapest route by its own expected-cost
ranking that is at least 4 capability points stronger - in this catalog usually
Kimi K3, sometimes GLM-5.3 Flash or GPT-5.6 Sol), and graded again. Both arms
of the carry decision were run.

| | escalations | wrong ones fixed | unnecessary | broke a good answer | extra cost | extra latency (median) |
|---|---:|---:|---:|---:|---:|---:|
| clean retry (**shipped**) | 39 | **18 of 35 (51 %)** | 4 | 0 | $1.04 ($0.027 each) | +84 s |
| carrying the failed answer | 39 | 19 of 35 (54 %) | 4 | 0 | $1.23 ($0.031 each) | +139 s |

Carrying the failed attempt fixes one more answer out of 35, costs 18 % more
and takes 66 % longer, so the default is a clean retry - a measurement, not a
preference.

**On the 192 answers as a whole:** 46 wrong before, 28 after; correctness 76.0 %
-> 85.4 %, for 39 second calls, $1.04, a judge call on every checked answer
(0.7 s), and 4 escalations that were not needed. Per *answered* turn that is
+0.7 s always and +84 s on the 20 % of turns that escalate.

### 14.3 The expected-cost model now prices the judge

Detection stops being a property of the user and becomes a property of the
route: `detect = detect_prob + (1 - detect_prob) x catch_rate`. Against that
gain the model charges the judge's own price on every checked answer, and the
second call each false alarm buys.

Two corrections came out of the simulation, and both are measurements the first
version of the model got wrong:

1. **A judge-driven retry succeeds at the measured rate (51 %), not at the
   escalation target's rate on the category.** The turns that reach it are the
   ones a cheaper route already failed, which is what makes them harder than
   average.
2. **A detected-but-unfixed failure still costs the stakes.** The router
   escalates once; pricing the residual as if a third attempt were waiting made
   the model prefer a weaker checked route over a stronger free one, and the
   simulation showed that trade losing accuracy at easy difficulties. With both
   corrections the judge never moves a choice that it cannot improve.

### 14.4 Chains of 3-4 calls, with and without the judge

`experiments/cascade.py`. **A simulation, and labelled as one.** Measured
inputs: per-model success by category and difficulty (§1), the catch, false
alarm and fix rates above, the measured per-route and judge latencies. Assumed:
a wrong step is fatal to the chain unless it is caught, steps are independent
given model and difficulty, difficulty is constant along the chain. The first
assumption is the pessimistic reading of Scott's point - it is what makes the
*no-judge* arm look bad - so a reader who thinks later steps often repair
earlier mistakes should treat the gap as an upper bound. 6,000 chains per cell.

Chains that finish **entirely** correct:

| | 1 call | 2 | 3 | 4 | cost of the 4-chain | wall time, 4-chain |
|---|---:|---:|---:|---:|---:|---:|
| coding, hard (d=0.70), no judge | 44.0 % | 20.2 % | 8.8 % | **3.6 %** | $0 (free route) | 341 s |
| coding, hard (d=0.70), with judge | 68.4 % | 47.4 % | 32.2 % | **22.3 %** | $0.0077 | 79 s |
| maths, moderate (d=0.45), no judge | 98.8 % | 96.8 % | 95.5 % | **94.8 %** | $0 | 20 s |
| maths, moderate (d=0.45), with judge | 99.4 % | 98.4 % | 97.6 % | **97.1 %** | $0.0016 | 62 s |
| coding easy (d=0.45) and maths hard (d=0.70) | — | — | — | *unchanged* | — | — |

This is the shape of Scott's objection, and it is real: without a check, a
four-call chain on hard coding work finishes correctly 3.6 % of the time even
though each individual call succeeds 44 % of the time. Catching 85 % of the
failures at the step where they happen takes that to 22 %. The gap *widens*
with chain length, which is exactly the property a single-hop benchmark hides.

The last row matters as much as the others: where the router already sends the
turn to a route the judge may not grade, the judge changes nothing at all -
not the choice, not the cost, not the latency. It is an addition to the cheap
tier, not a new tax on every turn.

### 14.5 Honest limits of this cycle

- **192 answers, 4 routes, 2 checkable categories.** The maths threshold rests
  on 11 wrong answers, and the per-category fix rate on 7. The coding numbers
  are the ones with enough mass to lean on.
- **The escalation measurement is single-pass.** Each flagged answer was
  escalated once; no repeated sampling, so the 51 % fix rate carries binomial
  noise of roughly +/-8 points.
- **The cascade table is a simulation**, not a measured agent run. No chain of
  three or four real routed calls was executed end to end.
- **The judge's price is an assumption.** Its token counts are measured; the
  dollars per token are a stand-in for a small model's public rate, which is
  why `verify.judge_usd` is a configuration value.
- **Two of 80 escalation calls failed** at the provider and are excluded; the
  arms are therefore 39 rows each rather than 40.
- **Nothing here says the judge improves frontier answers.** It was never asked
  to grade one, and the gate exists precisely to keep it from trying.

## Limits

- Cells hold 4–6 tasks; task difficulty for real traffic is a proxy (calls per turn).
- The held-out set is small by design (27 tasks, plus a 60-task pre-registered
  supplement). It separates categories; it does not rank frontier models, and no
  quality claim is made from a category with fewer than 10 valid *pairs*. With
  the supplement every category clears that floor — and the answer it gives is
  that the routing policy is level or slightly behind a single fixed route, with
  no cash saving to set against it (§12).
- **Neither run is fully verifiable by a reader.** The configuration half of the
  supplement's registration can only be checked by whoever holds the
  credential-bearing config it names; `evidence_verify.py` re-derives the rest
  and prints that limit every time. Nothing in the published evidence
  establishes that the recorded runtime config existed or produced the frozen
  policy/catalog identity.
- The design grader is structural. A page can satisfy every rule and still look bad.
- The simulator treats a failed turn as a whole-turn redo and assumes partially correlated retries.
- Latency is not modelled well; free-tier routes are slower (F's turns take longer in the replay).
- Plan usage in percent depends on a single week's conversion from list-price dollars.
- The subscription paths (§13) are n=1 per path, and the plan-with-headroom case
  was simulated with a stand-in usage reader.
