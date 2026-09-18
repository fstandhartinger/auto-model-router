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
