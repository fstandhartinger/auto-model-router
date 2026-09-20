# auto-model-router

A cost-, cache- and quota-aware LLM router. It sits in front of any number of
OpenAI-compatible providers and picks a model per user turn so that tasks get
solved at the lowest expected cost. It can also route *whole jobs* to a coding
agent's official CLI - including one running on a flat-rate subscription - so
that work lands on a plan you already pay for instead of a per-token bill. What
the vendors allow there is quoted, with links, in [`TERMS.md`](TERMS.md).

Status: experimental, measured. Full method and numbers: [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Vision

This router is one link in a longer chain, and it is built so the other links can
be plugged in without changing the routing logic.

1. **Evidence comes from a benchmark API.** Which models and providers exist,
   how capable each is *per topic*, what a task actually costs there, and what
   each provider charges for a cache read or write — served by a benchmark API in
   the [benchmarkheaven.com](https://benchmarkheaven.com) format, cached locally
   with a TTL, and every number carrying its basis and how strong that basis is
   ([`auto_router/bench.py`](auto_router/bench.py)). Local measurements override
   it, because headline scores mis-rank specific models and effort levels.
2. **A Jev-class model classifies the request.** Local Laya on CPU is the
   sample configuration's default; hosted [Jev](https://docs.typesafe.ai) and
   a no-model heuristic are selectable. Topic, difficulty,
   whether it needs tools or a long context, whether it builds on the previous
   turn, and what a wrong answer would cost. Hosted Jev took about 0.6 s; the
   current local CPU backend is much slower (measured below). Both receive only
   a scrubbed and truncated summary of the turn, never the raw transcript.
3. **Expected cost decides where it goes.** Call cost at the route's real cache
   state, times the measured chance of success, plus the price of a failure.
4. **The targets are deliberately heterogeneous.** Free tiers, metered APIs,
   flat-rate subscriptions through their own official clients — and, as a
   provider like any other, a **peer-to-peer network of volunteered GPUs**:
   browsers running a quantised model on WebGPU, reached over an
   OpenAI-compatible endpoint. Such a network is cheap and slow, which is exactly
   the shape of route the expected-cost rule is good at placing: it will send an
   easy turn there and keep a hard one on a strong paid model, because it prices
   the chance of failure rather than only the call.

Nothing in the router privileges a route by name. A peer-to-peer endpoint enters
the catalog with a price, a context length, a cache rule and a per-category
capability basis, and competes on those.

## Results in short

Eight models on 78 graded tasks, a replay of one week of real coding-agent traffic
(1,638 sessions, 57,696 calls, 8.7B input tokens, 96 % of them cache reads), and a live run
of the router server.

> **Read the table as a simulation, because it is one.** Every dollar figure below
> is *replay arithmetic*: real traffic and measured per-model success rates, priced at
> public list prices. **No money was saved and none was measured.** No invoice was
> compared, no A/B test was run against production, and the replay knows the whole week
> in advance in a way a live router does not. The numbers rank policies against each
> other under one set of assumptions; they are not a cash result and must not be quoted
> as one. Live, paired, *measured* results are in [`EXPERIMENTS.md`](EXPERIMENTS.md).

Cross-validated replay, public list prices, no subscription:

| policy | tasks solved | cost / week | cost per solved turn |
|---|---:|---:|---:|
| A: Claude Opus 5 for everything | 92.8 % | $12,917 | $6.42 |
| B: cheapest model predicted to succeed | 88.7 % | $5,472 | $2.85 |
| C: B + expected-value downgrade rule | 88.7 % | $5,478 | $2.85 |
| D: start cheap, escalate on failure, downgrade on cache expiry | 87.7 % | $2,995 | $1.58 |
| **F: minimise expected cost (default)** | **89.0 %** | **$1,341** | **$0.70** |
| F with higher stakes (10× turn cost) | 90.0 % | $2,236 | — |

With free-tier models and a Claude plan that may only serve Claude Code's own sessions, F
kept success at today's level (91.0 % vs 91.6 %) while cutting simulated plan use from 127 % to
24 % of the weekly quota for $485/week of API spend, or to 31 % for $0 with free models only.

**Checking cheap answers (18 Sep).** On 192 answers from four cheap routes, Jev caught 85 %
of wrong coding answers (10 % false alarms) and 73 % of wrong maths answers (no false alarms)
in 0.69 s; re-running the flagged ones on a stronger route lifted correctness from 76.0 % to
85.4 % for $0.027 a rescue. In a *simulated* four-call chain of hard coding turns, chains that
finish entirely correct go from 3.6 % to 22.3 % — the cascade effect a single-hop benchmark
hides. Details and limits: [`EXPERIMENTS.md` section 14](EXPERIMENTS.md).

The four things that mattered most:

1. **Measured success rates.** Capability read from benchmark headlines mis-ranks specific
   models and effort levels; with those beliefs F loses 3 points and triples spend. A
   78-task calibration run (about $20) fixes it — see `experiments/calibrate.py`.
2. **Price failure, not just calls.** Always starting on the cheapest model and escalating
   (D) sends hard turns through two paid attempts. F starts each turn on the model with the
   lowest *expected* cost including the chance and price of a retry.
3. **Route at user turns, not API calls.** Agent traffic is long tool loops over 100k+
   cached prefixes; switching mid-loop throws away a seconds-old cache. Sessions average
   1.3 user turns, so downgrade rules for warm conversations barely matter.
4. **Caching is a property of the route.** Read shares ranged from 0 % to 99 % for the same
   kind of model on different hosts, and Claude Code on a plan keeps a one-hour cache.

## The problem in one paragraph

Model prices span two orders of magnitude, and most turns in an agent session
do not need the strongest model. But every switch abandons the provider's
prompt cache: a cached prefix is read at ~0.1x the input price, while the same
prefix on a different model is written from scratch at 1x (or 1.25x on
Anthropic). On a 50k-token prefix that is a 10–12x difference on the largest
part of the bill. A router that ignores this churns and loses money; a router
that never switches strands easy work on expensive models.

## What it does

For each **user turn** (not each tool call — inside an agent's tool loop the
router stays on the turn's model):

1. **Classify** the turn with [Jev](https://docs.typesafe.ai): category, difficulty,
   needs tools / vision / long context, whether it builds on the previous turn,
   and how costly a wrong answer would be. One ~0.6 s call.
2. **Look up** each configured model: list prices including cache read and cache
   write, context length, per-category capability and a benchmaxxing penalty
   from a benchmark API (Benchmark Heaven format), cached locally with a TTL.
3. **Price** every eligible model for this turn with its actual cache state
   (warm tokens, TTL, minimum cacheable prefix, measured hit rate).
4. **Choose** with the configured policy (below), then escalate on failure
   signals. A *capability* failure (repeated failing tool results, an inadequate
   answer) escalates to a clearly stronger route; an *availability* failure
   (5xx, a broken connection, an answer the provider says it never finished)
   falls back sideways to the next usable route, because a 503 is not evidence
   that the model was too weak.
5. **Check the answer — but only a cheap one.** When the answering route is in
   the cheap tier, Jev is asked one typed question about what came back, and a
   rejected answer is re-run on a stronger route. See below.
6. **Record** the decision as four separate objects, plus the verdict — see below.

### Checking a cheap answer before returning it

The router used to choose a model and never look at what came back. For a
frontier route that is the only honest option: the judge is not smarter than
the thing it would be grading. For the cheap tier the relation is the other way
round — Jev answers one narrow, typed question about an answer a much smaller
model produced — and the measurement in EXPERIMENTS.md section 4 says it works
there. The calibration for this feature (EXPERIMENTS.md section 14, 192 cheap
answers) put numbers on it: **85 % of wrong coding answers caught for a 10 %
false-alarm rate**, **73 % of wrong maths answers for none at all**, in a
**0.69 s** median call.

So [`auto_router/verify.py`](auto_router/verify.py) checks exactly that tier:

- the answering route is **cheap** (free, `:free`, or a blended list price at or
  below `verify.max_price_per_mtok`) **and** its capability for this category
  stays below `verify.max_capability` — a free frontier-class route is not
  cheap in the sense that matters here;
- the request is **self-contained**: a question about a pasted document is
  skipped, because the judge is not shown the document and would flag every
  answer (the measured case: 24 out of 24);
- a judge is configured at all.

Everything else is recorded as "not verified", with the reason, and behaves
exactly as it did before. A verdict is one Noul — P(the answer fully and
correctly addresses the request), against task-specific criteria — plus a
choice naming the failure (`wrong`, `incomplete`, `off_topic`, `fine`), both in
one ~0.6 s call. Below `verify.thresholds[category]` the turn escalates to the
next route by the policy's own expected-cost ranking that is clearly stronger,
and the conversation's difficulty floor is raised so the *next* turn does not
start too low again.

What that buys, measured on the same 192 answers: 46 wrong before, 28 after -
**76.0 % -> 85.4 % correct** - for 39 second calls at $0.027 each, +0.7 s on
every checked turn and +84 s on the one in five that escalates. The escalation
target fixed **51 %** of the answers it was handed; it broke none.

And in a chain, which is where it matters (simulated, EXPERIMENTS.md 14.4): a
four-call chain of hard coding turns finishes entirely correct **3.6 %** of the
time without the judge and **22.3 %** with it. Each individual call succeeds
44 % of the time either way - the difference is that a wrong early answer stops
being something the next three calls build on.

Three deliberate limits:

- **A judge that fails never escalates.** An outage in the checker must not
  become an escalation storm that costs more than the failures it catches; the
  turn is recorded as unverified and the answer stands.
- **The second attempt gets a clean shot by default.** Handing the stronger
  model the failed answer helps when the failure was *incomplete* and anchors
  it when the failure was *wrong*; `verify.carry_failed_attempt` turns it on.
- **A stream is not re-written.** Its bytes are already on the wire, so the
  verdict arrives in a final `x_router` chunk and in the decision record, and
  the second answer is only appended for a client that asked for it
  (`"x_router": {"stream_escalate": true}`, or `AUTO_ROUTER_STREAM_ESCALATE=1`).

The expected-cost policy prices all of this: the judge's own cost, the second
calls its false alarms buy, and the failures it catches that the user would
not have. That is what makes a cheap route *more* attractive than it was —
see "How F decides, plainly".

### An answer the provider says it never finished

A route that exhausts its output budget returns HTTP 200 and a well-formed
body, so it used to be recorded as a success. It is not one, and it is the only
route difference the held-out run measured that no capability score predicted:
on the two hardest web-design tasks one route burned a 12,000-token budget
without finishing the page while a route the design-arena evidence rated
*lower* finished the same prompt in about 2,200 tokens.

So a completion the provider itself flags as a length stop
(`finish_reason: length`, `stop_reason: max_tokens`, or a gateway's
`native_finish_reason` equivalent) is recorded as `status: "truncated"` — an
observed failed attempt — and the non-streaming OpenAI surface then takes the
same sideways safe fallback a 5xx takes, by default for **one** extra route
(`AUTO_ROUTER_TRUNCATION_RETRIES`, `0` disables the retry and keeps the label).
Three limits are deliberate:

- **Only the provider's own machine-readable flag.** Never the answer text;
  "it reads as cut off" is an inference this router will not make, and acting
  on it would mean reading content the decision record deliberately excludes
  ([`auto_router/truncation.py`](auto_router/truncation.py)).
- **Nothing is written back into capability.** No score or success estimate is
  invented from the observation; it only reaches the next *selection*, below.
- **A stream is recorded, never retried.** Its bytes are already on the wire.
  The `/v1/messages` surface has no attempt loop, so it records too.

The tokens a truncated attempt spent were really billed, so it is committed and
metered like any other call; `total_requests` therefore counts attempts, not
turns, on a turn that truncated.

**Repeated truncation changes the next comparable request.** The retry above
fixes one turn; without memory the next similar request would go straight back
to the route that just ran out of budget. The router therefore keeps a small
in-process memory of *observed* statuses
([`auto_router/outcome_memory.py`](auto_router/outcome_memory.py)): per route
and per *comparable request* - same category, same output-budget bucket
(`<=1024`, `<=4096`, `<=16384`, `<=65536`, larger, or no `max_tokens`). At the
start of a turn, after the policy has chosen, the choice is replaced only when:

- that route has at least **2** provider-flagged length stops for this key, and
  they are at least **50 %** of its last **8** observed `ok`/`truncated`
  outcomes within **24 h** (`policy.truncation_memory`: `min_truncations`,
  `min_rate`, `window`, `ttl_s`, `enabled`); and
- a usable route exists that is not itself flagged. It is the policy's own best
  remaining route by expected cost. With none, the policy's choice stands.

The decision record says which happened in `selection.truncation_memory`
(route, key, counts, basis, fallback, applied) - counts and route names only.
Errors, `not_taken` records and launched jobs are not counted; a tool loop feeds
the memory but never switches mid-loop; the memory starts empty, so a fresh
router routes exactly as before. It is not persisted across restarts.

## Four things the router never mixes up

A router that stores "we picked X and it cost $0.004" has already lost the
ability to tell you whether it was right. Every decision is kept as four
separate objects ([`auto_router/decision.py`](auto_router/decision.py)), served
at `GET /v1/router/decisions` and appended to a JSONL ledger when
`AUTO_ROUTER_LEDGER` is set:

| object | what it is | never contains |
|---|---|---|
| `classification` | what the task *is*: category, difficulty, confidence, source | a model name |
| `selection` | what the router *decided*: candidates, evidence, cache decision, chosen route, fallback | an outcome |
| `estimated_outcome` | what it *expected*: cost, `p_success`, tokens, and the basis | anything measured |
| `observed_outcome` | what *happened*: status (`ok`, `truncated`, `upstream_error`, `transport_error`), latency, provider-reported tokens, cost when a price basis exists | an estimate standing in for a measurement |
| `verification` | what the *judge* said about the answer, or why it was not asked: `p_adequate`, the threshold, the failure type, where the turn escalated to | the answer it judged |

A cost with no measurement basis is recorded as `null` with the reason, never
as a zero — a free or subscription route must not later read as a measured
saving. The whole record is prompt-free by construction: no prompt text, no
response text, no tool arguments, no credentials. That is enforced by a test,
not by convention.

## Specialised routes come from evidence, not from names

Nothing in the router says "model X is the design model". Capability per
category is read from whatever the configured benchmark API serves, and each
number carries its basis and how strong that basis is:

```
design       80.12  designarena frontend+fullstack elo 1320 over 2813 battles   direct
coding       76.20  aa_coding_index                                             direct
summarisation 68.57 mean(long_context, knowledge, aa_intelligence_index x 1.5)  derived
```

That matters because the ranks disagree. On data served on 18 September 2026,
three routes sat within 0.3 points of each other on coding (76.2–76.5) and
spread across **46.5 to 80.1** on design. A router that reasons only about
"how good is this model" cannot see that; one that reads a per-category
evidence basis can.

When evidence is missing, thin or stale, `policy.success.evidence_discount`
pulls the score toward a neutral prior in proportion to how weak it is, the
decision records `evidence_confidence` and a `safe_fallback` reason, and the
response carries `X-Router-Safe-Fallback`. Setting the discount to `0`
reproduces the previous behaviour exactly.

Subscription models are priced with a shadow price from quota pacing
([`quota.py`](auto_router/quota.py)): free while the weekly quota is projected
to stay well below a reserve line, increasingly expensive as it approaches it,
closed above it or when the short session window is nearly full.

## Policies

| | policy | idea |
|---|---|---|
| A | `A_static` | one strong model for everything |
| B | `B_naive` | cheapest model that clears a success bar, per turn, cache ignored |
| C | `C_ev_switch` | B's target; escalate freely, downgrade only if horizon savings beat the risk |
| D | `D_escalate` | start cheap, escalate on failure, remember it; downgrade when the cache expired or savings beat risk |
| E | `E_escalate_sub` | D plus the subscription tier with quota pacing |
| F | `F_expected` | minimise expected cost: call cost + P(fail) × (retry or stakes), cache- and quota-aware, with difficulty memory (default) |

### How F decides, plainly

For each model that could take the turn it estimates three numbers: what the turn costs on
that model given what is already cached there; how likely the model is to get it right
(measured success rate for this category and difficulty); and what a failure costs — a retry
on a stronger model if the failure would be noticed, or the stakes of a wrong answer if not.
It picks the model with the lowest sum. Easy turns land on free or very cheap models because
their failure chance is tiny; hard turns go straight to the model with the best success per
dollar; a failed turn raises the conversation's difficulty memory so the next follow-up does
not start too low. Subscription models cost nothing while the weekly quota is on pace, and
their price rises to list price as usage approaches the reserve line.

With the answer judge configured, one term of that sum changes. "Would a failure be
noticed?" stops being a property of the user and becomes a property of the route: a checked
route's failures are noticed at the measured catch rate whether or not anyone is paying
attention, and its wrong answers therefore cost a retry rather than the stakes. The judge is
charged for on both sides — its own price per answer, and the second calls its false alarms
buy — so a jumpy judge makes a cheap route *less* attractive, not more. A route the judge
may not grade is priced exactly as before, and so is every route in a deployment with no Jev
key.

## Running

One-line local install (Python 3.10+, CPU-only wheels; no compiler or GPU):

```bash
curl -fsSL https://raw.githubusercontent.com/fstandhartinger/auto-model-router/main/scripts/install-local.sh | sh
```

The first local request downloads about 1.7 GB of Laya weights. On the measured
x86-64 Linux machine it peaked at 2.9 GB RAM. Then:

```bash
cp examples/config.example.yaml my.local.yaml   # describe your providers
export AUTO_ROUTER_CONFIG=my.local.yaml
auto-model-router
pytest -q
python experiments/calibrate.py --matrix runs/matrix.jsonl --out success.json   # after run_matrix.py
```

Choose the routing classifier under `policy.classifier`:

```yaml
classifier: {backend: local, model: convaiinnovations/laya, threads: 4}
# classifier: {backend: hosted}    # set TYPESAFE_API_KEY
# classifier: {backend: heuristic} # zero inference cost and latency
```

Local means the scrubbed routing input stays on the machine. Hosted uses the
existing Jev API path. If local inference or model loading fails, routing falls
back cautiously instead of failing the user's LLM call.

### Measured classifier comparison (20 September 2026)

Exact replay through `F_expected` on the router's own 78-task calibration set;
the chosen model's frozen per-task result and list-price cost are used, so no
new LLM answers were generated. These are small-sample results, not production
traffic savings.

| classifier | solved | added latency p50 / p95 | classifier cost | end-to-end cost | cost / solved turn |
|---|---:|---:|---:|---:|---:|
| local Laya 421M, 4 CPU threads | 66/78 (84.6%) | 15.34 / 21.35 s | $0 marginal API cost | $1.5135 | $0.02293 |
| hosted Jev | 65/78 (83.3%) | 0.62 / 0.84 s | $0.00312 | $0.1748 | $0.00269 |
| heuristic | 68/78 (87.2%) | 0 / 0 s | $0 | $0.2640 | $0.00388 |
| always Opus 5 | 71/78 (91.0%) | 0 / 0 s | $0 | $9.0790 | $0.12787 |

The local model did not lose solve rate to hosted Jev on this replay, but it
misclassified task category badly (19.2% versus Jev's 84.6%) and compensated by
sending 26 tasks to the expensive Sol route (Jev sent 4). It was therefore
8.7x as expensive end-to-end and about 25x slower at p50 than hosted Jev. Local
is useful for privacy/offline routing, but hosted Jev remains the practical
learned default; the heuristic was strongest on this small exact replay. The
reproducible aggregate is in
[`evaluation/jev-router-20260920.json`](evaluation/jev-router-20260920.json),
and the replay command is documented there.

A measured success table from our run ships as `examples/success.measured.json`; point
`policy.success.table` at it or at your own.

Point OpenAI clients at `http://127.0.0.1:8787/v1`, or Claude Code at it with
`ANTHROPIC_BASE_URL=http://127.0.0.1:8787`.

| Variable | Meaning |
|---|---|
| `AUTO_ROUTER_CONFIG` | provider/model config (YAML or JSON) |
| `AUTO_ROUTER_POLICY` | policy name if the config does not set one (default `F_expected`) |
| `AUTO_ROUTER_BENCH_URL` | benchmark API base URL (default `https://benchmarkheaven.com`) |
| `AUTO_ROUTER_BENCH_OFFLINE` | `1` = use cached benchmark data only |
| `AUTO_ROUTER_CACHE_DIR` | where benchmark responses are cached |
| `AUTO_ROUTER_SUBSCRIPTION_MODE` | `passthrough_only` (default) or `route_others`; see "Two ways to use a plan" |
| `AUTO_ROUTER_REWRITE_MODEL` | allow replacing Claude Code's requested model on passthrough (off) |
| `AUTO_ROUTER_PLAN_MODELS` | models your own plan includes; bounds the rewrite on subscription traffic |
| `AUTO_ROUTER_ALLOW_UPSTREAM_HOSTS` | extra hosts a subscription credential may reach (default: none) |
| `AUTO_ROUTER_LEDGER` | JSONL file for decision records; unset disables the ledger |
| `TYPESAFE_API_KEY` | Hosted Jev classifier key; not needed for local or heuristic mode |

Responses carry `X-Router-Model`, `X-Router-Category`, `X-Router-Difficulty`,
`X-Router-Reason`, `X-Router-Conversation`, `X-Router-Turn-Start`,
`X-Router-Decision` (the id of the decision record), `X-Router-Evidence`
(0–1 evidence confidence), `X-Router-Cache` and, when one applies,
`X-Router-Safe-Fallback`.

## Four ways to use a flat-rate plan

A coding subscription is the cheapest strong model most people have, and it is
also the one a router cannot simply call: it is sold for use through its own
client. There are four honest ways to put a router next to it, and this
repository implements all four. [`TERMS.md`](TERMS.md) quotes the vendor
documentation behind each, including the parts that say no.

| | how it works | plan's login touches the router? | Anthropic's position | pick it when |
|---|---|---|---|---|
| **1. `route-run`** | decide per job which official CLI to start | no | ordinary use of the client | whole jobs, scripts, agents |
| **2. gateway** | `ANTHROPIC_BASE_URL` → router; plan turns forwarded unchanged, others answered by cheaper models (`route_others`) | yes, in flight (forwarded, never stored) | forwarding documented; other models "not supported" | your own machine, if you accept maintaining it |
| **3. switch mode** | one conversation moves between *cheap mode* (router, its own credential) and *plan mode* (Claude Code direct to Anthropic) | **no** | both modes documented | **interactive Claude Code — the recommended way** |
| **4. delegate tool** | the plan model stays in charge; an MCP tool hands sub-tasks to cheap models | no | ordinary MCP use | long plan sessions with big, separable sub-tasks; also Codex |

Measured on 19 Sep 2026 on three small coding tasks (details in
[`EXPERIMENTS.md`](EXPERIMENTS.md) §14): every mode finished every task; the
cheap-side modes spent **no** plan quota and took 2–5× longer; the delegate tool
did not reduce plan use on small tasks; and **switching one conversation to the
plan half-way cost 3.4× the plan use of simply having stayed on the plan**,
because the plan reads the whole conversation cold. Decide early, switch rarely.

### 1. Route the job, not the request (`route-run`)

The launcher decides *which program* should do a piece of work and then starts
that program, unmodified, signed in the way its vendor documents:

```bash
cp examples/launcher.example.yaml my.local.yaml    # add a runner block per route
export AUTO_ROUTER_CONFIG=my.local.yaml TYPESAFE_API_KEY=...
scripts/route-run --list                           # what can run a job here
scripts/route-run --dry-run "fix the flaky upload test"   # decide, run nothing
scripts/route-run "fix the flaky upload test"             # decide and run it
```

An easy job lands on a free model; a hard one goes to a plan through
`claude -p --model …` or `codex exec -m …` while that plan has headroom, and to
a metered API model when it does not. Nothing intercepts the client's traffic,
nobody's login is read, and the choice of model is made the way a person makes
it — with the client's own flag.

Each run appends the same four-part decision record the HTTP surface writes, so
launched jobs and routed turns end up in one ledger. What a launched client
reports back is its exit status and how long it took, so that is what the record
contains: no token counts are invented, and a subscription run has no per-token
cost to claim.

Two details that are worth more than they look:

- **`clear_env`** empties the credential variables that would move the run from
  the plan to per-token billing (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`,
  `OPENAI_API_KEY`). A single exported key otherwise turns a "free" job into an
  invoice, silently.
- **`launch_only: true`** marks a plan that can only be reached by launching its
  own client — a ChatGPT plan, for instance. Such a route is a candidate for the
  launcher and is removed from the HTTP catalog, because no HTTP request from
  another client can ever be served from it.

Quota pacing decides when a plan is "in budget". `usage_command` runs **your**
usage reader and parses percentages; the router never reads a login token to ask
a vendor how full a plan is, and a reader that fails or goes stale closes the
plan rather than opening it on a guess.

### 2. A gateway in front of Claude Code (`ANTHROPIC_BASE_URL`)

Anthropic documents this case directly:

> Setting only that variable, without a gateway credential, doesn't replace the
> subscription. Requests still route through the gateway, but a saved claude.ai
> login remains the active credential, so its usage limits and billing apply.
> Gateways that pass this traffic on to Anthropic must forward the OAuth
> capability in `anthropic-beta`.
> — [code.claude.com/docs/en/llm-gateway](https://code.claude.com/docs/en/llm-gateway)

So run the router and point Claude Code at it, with **no** gateway credential
set:

```bash
uvicorn auto_router.server:app --host 127.0.0.1 --port 8787
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude       # ANTHROPIC_API_KEY unset
```

The turn is forwarded to Anthropic byte for byte — every `anthropic-*` header
verbatim, the system array untouched, the stream and its keep-alive pings
relayed as they arrive — and it is billed to the plan. Set a gateway credential
instead and the same endpoint meters the traffic against that credential; both
are legitimate, and which one is in force is not a guess: the OAuth capability
on the request says so ([`auto_router/plan_auth.py`](auto_router/plan_auth.py)).

What the router does with a subscription-authenticated turn is a setting:

| `AUTO_ROUTER_SUBSCRIPTION_MODE` | behaviour |
|---|---|
| `passthrough_only` (default) | Forward every turn unchanged and record the decision the policy *would* have made as `not_taken`. Claude Code behaves exactly as it does with no gateway, and the ledger still answers "how much of this week's plan use could have gone somewhere cheaper". |
| `route_others` | Additionally let the policy serve a turn from another provider on **your own** API key. It works — but Anthropic "doesn't support routing Claude Code to non-Claude models through any gateway", so you maintain it yourself. |

Use `route_others` only for **your own login on your own machine**: in this mode
the plan's token passes through the router on every request (forwarded, never
read or stored), which is the configuration closest to Anthropic's rule that
developers "may not collect, store, or intermediate Claude.ai credentials or
session tokens". Switch mode (3) gets the same result without that. "Not
supported" is also not an empty word: a real session on 19 Sep 2026 broke twice
until the translation learned that Claude Code now sends `role: "system"`
messages mid-conversation, and that a tool-call id such as `functions.Write:0`
from another model makes every later plan turn fail with a `400`. Both are
fixed; the next such change arrives with a Claude Code release.

Three refusals are built in, and they are code rather than advice:

- A credential that identifies as a claude.ai login is forwarded to
  `api.anthropic.com` and to nothing else, whatever the configuration says.
- A turn served from another provider is authenticated with that provider's own
  key; the client's `Authorization` header is not forwarded there.
- The model Claude Code asked for is not replaced by default. With
  `AUTO_ROUTER_REWRITE_MODEL=1` it may be, and on subscription traffic only for
  a model listed in `AUTO_ROUTER_PLAN_MODELS` — a plan grants particular models,
  and a gateway should not ask it for one the developer could not have picked.

Nothing is logged that could identify a credential: `redact()` covers every
header dict that reaches a log line, and the decision records carry no prompt
text and no token.

### 3. Switch mode: cheap by default, your plan when it matters

```bash
uvicorn auto_router.server:app --host 127.0.0.1 --port 8787 &   # the gateway
export AUTO_ROUTER_CONFIG=router.local.yaml   # cheap routes + a `subscription: claude` route
python -m auto_router.switch                  # instead of `claude`; any claude flags work
python -m auto_router.switch -p "fix the flaky test"   # headless works the same way
```

Claude Code starts in **cheap mode**: `ANTHROPIC_BASE_URL` points at the router
and a gateway credential is set, so "the credential replaces the subscription
login for that session, and the subscription's usage limits don't apply". The
router answers each request from free or metered routes on your own keys.

A `UserPromptSubmit` hook asks the router, before any model sees a prompt,
whether it belongs on the plan. If it does, the hook blocks the prompt (which
"erases it from context"), the wrapper stops Claude Code and starts it again in
**plan mode** — no base URL, no credential variable, signed in with your own
claude.ai login, talking to Anthropic directly — with
`--resume <session> "<your prompt>"`. The conversation continues where it was,
including everything the cheap side did. The router is not in the plan's path
at all, so it never sees the plan's login.

- **Escalation from inside a turn.** When the cheap route gets stuck (by
  default three failing tool results in a row) and the policy's next rung is
  the plan, the gateway ends the turn with a one-line notice and hands the
  conversation over the same way. The wrapper tags cheap-mode requests with a
  random token (`ANTHROPIC_CUSTOM_HEADERS`) so the gateway knows which session
  to hand over; without that header a gateway credential can never reach a plan
  route.
- **Overrides.** Start a prompt with `~plan` or `~cheap` to force the mode.
  (`!` would be Claude Code's own shell mode.)
- **Switching is not free.** The side you switch to reads the whole
  conversation without a warm cache. The hook therefore leaves the plan only
  for a clearly easy prompt (`AUTO_ROUTER_SWITCH_STICKY`, default 0.35), and a
  tie between the plan and a free route goes to the free route.
- **Failing open.** If the router is down or misconfigured, the hook lets the
  prompt through in the current mode; if the gateway is down, the wrapper
  starts on the plan and says so.
- Claude Code only. Codex has no cheap mode yet: it needs an OpenAI Responses
  API endpoint, which this router does not implement.

### 4. The delegate tool: the plan orchestrates, cheap models do the legwork

```bash
claude mcp add auto-router-delegate -- env AUTO_ROUTER_CONFIG=$PWD/router.local.yaml python -m auto_router.delegate
codex mcp add auto-router-delegate -- env AUTO_ROUTER_CONFIG=$PWD/router.local.yaml python -m auto_router.delegate
```

The official client runs unchanged on its plan. It gains one tool,
`delegate(task, cwd)`, which runs the job launcher restricted to non-plan routes
(`--no-plans`): the cheapest route expected to do the job starts its own agent
CLI in `cwd` and returns what it printed. The plan model then checks the result.
Codex needs `default_tools_approval_mode = "approve"` on the server to call it in
`codex exec`. Measured: with a soft hint the model never delegated a small
task; with an explicit "you are the orchestrator" instruction it did, and plan
use stayed about the same — the tool pays off only when the delegated part is
large compared with Claude Code's own fixed prompt.

### Install it with a coding agent

Paste this into Claude Code or Codex in the directory you want it in:

> Clone https://github.com/fstandhartinger/auto-model-router and set it up for
> me. Read TERMS.md first and keep to it. Then: create a virtualenv, install
> requirements.txt, run `pytest -q` and stop if anything fails. Copy
> `examples/launcher.example.yaml` to `router.local.yaml` and edit it for this
> machine — one route per model I can actually reach, each with a `runner`
> block that starts its official CLI, API keys referenced by environment
> variable name only and never pasted in. Mark any plan that is only reachable
> through its own CLI `launch_only: true`. For every subscription route set
> `clear_env` to the credential variables that would otherwise bill me per
> token. If I have a command that reports my plan usage as percentages, wire it
> up as `usage_command`; if not, leave the subscription tier closed rather than
> guessing. Then show me `scripts/route-run --list`, a `--dry-run` for one easy
> and one hard task, and tell me in three lines what it decided and why.

## Verifying a checkout

```bash
pytest -q                          # full suite, including the HTTP smoke test
python scripts/smoke_http.py       # the smoke test on its own, with its report
python experiments/sandbox.py      # prove the execution sandbox really isolates
python experiments/heldout.py preregister   # write the task set and analysis plan
python experiments/heldout.py verify        # confirm the task set has not changed
python experiments/evidence_verify.py --dir <a finished run directory>
```

`scripts/smoke_http.py` starts a stub upstream and the router itself on
ephemeral loopback ports, with the benchmark API pointed at an unroutable host
and no `TYPESAFE_API_KEY`, so the benchmark-outage and classifier-outage
fallbacks are what is exercised. It tears both processes down and then checks
that neither port still accepts a connection, so no service is left behind.

## Verifying a finished run, and what you cannot verify

`experiments/evidence_verify.py` re-derives a finished run's **public identity**
from its registration and this checkout — the task file, the registered task ids
and counts, the analysis plan, the frozen policy/catalog identity, and every
registered experiment- and product-code digest. It never opens a config, so it
gives the same answer on every machine, and it prints the limit of what it
established.

That limit is real and is not a formality. A run's registration also records the
path and digest of the **runtime config** it was measured against. That file
carries provider credentials, is not in this repository and cannot be
reconstructed from the registration, so the only tool that can check it —
`experiments/supplement.py verify --config <that exact file>`, which refuses when
the file is missing or has drifted — can only be run by whoever already holds it.
**A reader cannot verify the configuration half of a published run**, and
`evidence_verify.py` says so on every run rather than letting a pass be read as
more than it is. A run registered from now on should also freeze a
credential-free projection of its config
(`evidence_verify.py --dir <run> --freeze-public-config <config>`), which closes
that gap for the next run and cannot close it retroactively for an old one.

## Layout

| Path | Purpose |
|---|---|
| `auto_router/catalog.py` | model, price and cache-rule types |
| `auto_router/bench.py` | benchmark API client with disk cache and offline fallback |
| `auto_router/config.py` | provider config → catalog |
| `auto_router/economics.py` | cache-aware cost model and success model |
| `auto_router/policies.py` | policies A–F |
| `auto_router/verify.py` | which answers are checked, the verdict, and what it costs |
| `auto_router/quota.py` | subscription quota pacing |
| `auto_router/jev.py` | Jev classifier and adequacy judge, with credential scrubbing |
| `auto_router/decision.py` | classification / selection / estimate / observation, kept apart |
| `auto_router/ledger.py` | append-only JSONL decision ledger |
| `auto_router/router.py` | live routing state |
| `auto_router/server.py`, `shim.py` | HTTP API and Claude Code passthrough |
| `auto_router/plan_auth.py` | which credential is on a request, and where it may go |
| `auto_router/launcher.py`, `scripts/route-run` | job-level launcher: pick the tool, start its own client |
| `auto_router/switch.py` | switch mode: one Claude Code conversation between cheap mode and plan mode |
| `auto_router/delegate.py` | MCP `delegate` tool: a plan session hands sub-tasks to cheap routes |
| `auto_router/translate.py`, `stream_translate.py` | Anthropic ↔ OpenAI translation |
| `experiments/sandbox.py` | Bubblewrap isolation for executing model-produced code |
| `experiments/heldout.py` | pre-registered held-out evaluation across six categories |
| `experiments/supplement.py` | the pre-registered supplement: its own frozen registration (policy and catalog included), a concurrent runner, a cumulative budget guard, the combined report |
| `experiments/evidence_verify.py` | offline re-derivation of a finished run's public identity, with the limit of that guarantee printed every time |
| `experiments/pairing.py` | valid-**pair** accounting — a pair counts only when both arms were graded |
| `experiments/graders.py` | deterministic graders (no grader calls a model) |
| `experiments/verify_calibrate.py` | answer, judge, escalate, sweep: where the verify thresholds come from |
| `experiments/cascade.py` | chains of 3–4 routed calls, with and without the judge (simulation) |
| `experiments/` | task set, evaluation harness, simulator |
| `scripts/smoke_http.py` | local HTTP smoke test including the outage paths |

## What the evaluation actually shows

Ten valid **paired** tasks per category, across two separately pre-registered
runs (27 + 60 tasks), against the real normal routing policy — `F_expected`
with the live classifier, the same object the server builds:

- The routing policy is **never ahead** of a single fixed route in any of the
  six categories. It ties in coding, research and cache-repeat, and is behind by
  one, two and three discordant pairs in design, maths and summarisation. None
  of those is significant at this size, and none of the ties is evidence of
  equality either — the conservative paired interval on a twelve-pair tie still
  runs ±0.265.
- **No saving was measured and none is claimed.** The routing policy keeps
  choosing free routes, which is the correct decision and also why there is no
  cash contrast: both free arms spent $0.0000. The metered comparator spent
  $0.9457 for pass rates that are equal or one pair better.
- **No quality claim is supported in any category**, and that is structural:
  "at least as often" is a non-inferiority statement and no non-inferiority
  margin was pre-registered. What ten valid pairs per category *do* buy is a
  ceiling — a routing advantage large enough to show up at this sample size is
  not there.

The combined report prints eleven named limitations next to those numbers,
including that the supplement is an adaptive sample, that its design tasks are
mostly compact components, and that the cache-repeat category is warm-prefix
repeats rather than independent tasks. `EXPERIMENTS.md` §12 has the tables.

## Privacy

Only a scrubbed, truncated summary of a turn ever reaches Jev, and scrubbing
runs *before* truncation so a secret cannot survive by sitting past the cut.
`auto_router.jev.scrub` removes private-key blocks, `Authorization`-style
strings, credential-shaped assignments, credentials embedded in URLs, a list of
well-known key shapes (Anthropic, OpenAI, GitHub, Slack, AWS, Stripe, Google,
Hugging Face, JWT) and the literal value of any environment variable whose name
looks credential-like.

This is best effort and fails toward removing too much: prose about
configuration (`lines of key = value`) is redacted as if it were a credential.
Arbitrary sensitive text with no recognisable shape — a customer name, a
private document — is **not** detected and must not be sent in the first place.
`AUTO_ROUTER_JEV_REQUEST_CHARS` and its siblings cap how much is sent at all.

## Data and terms

Benchmark numbers served by the default benchmark API include third-party data
whose terms restrict use in competing commercial products. Fine for personal
and internal experiments; a commercial deployment needs its own licensed or
self-measured capability data.

A subscription is for your own sessions on your own plan, through the vendor's
official client. [`TERMS.md`](TERMS.md) quotes what each vendor documents, with
links and with the dates they were read; the two sentences that matter are that
Anthropic describes a gateway in front of Claude Code with a claude.ai login as
a configuration where "its usage limits and billing apply", and that plan OAuth
is reserved for ordinary use of the unmodified client - a developer may not
offer Claude.ai login in their own product, route other people's requests
through plan credentials, or collect, store or intermediate those credentials.
This router keeps to both: it forwards a subscription credential to Anthropic
and to nowhere else, never reads or stores one, and paces plan use well below
the plan's own limits. An earlier version of this file called the gateway a
grey area; the documentation quoted in `TERMS.md` is clearer than that, in both
directions.

A ChatGPT plan is reachable only through the Codex CLI, so it is configured as a
`launch_only` route: the launcher starts that CLI, and the HTTP surface never
offers it.

## Licence

MIT
