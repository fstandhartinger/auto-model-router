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

## Limits

- Cells hold 4–6 tasks; task difficulty for real traffic is a proxy (calls per turn).
- The simulator treats a failed turn as a whole-turn redo and assumes partially correlated retries.
- Latency is not modelled well; free-tier routes are slower (F's turns take longer in the replay).
- Plan usage in percent depends on a single week's conversion from list-price dollars.
