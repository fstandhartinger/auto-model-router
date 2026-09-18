# auto-model-router

A cost-, cache- and quota-aware LLM router. It sits in front of any number of
OpenAI-compatible providers (and, optionally, a flat-rate Claude subscription
through Claude Code) and picks a model per user turn so that tasks get solved
at the lowest expected cost.

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
2. **[Jev](https://docs.typesafe.ai) classifies the request.** Topic, difficulty,
   whether it needs tools or a long context, whether it builds on the previous
   turn, and what a wrong answer would cost. One ~0.6 s call, on a scrubbed and
   truncated summary of the turn, never the raw prompt.
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
   that the model was too weak. A Jev adequacy judge (`jev.judge`) is measured
   in EXPERIMENTS.md and works well for self-contained coding and math answers;
   it is not yet wired into the server.
5. **Record** the decision as four separate objects — see below.

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
- **Nothing is written back into capability.** The observation is recorded and
  goes no further; no score is invented from it.
- **A stream is recorded, never retried.** Its bytes are already on the wire.
  The `/v1/messages` surface has no attempt loop, so it records too.

The tokens a truncated attempt spent were really billed, so it is committed and
metered like any other call; `total_requests` therefore counts attempts, not
turns, on a turn that truncated.

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

## Running

```bash
pip install -r requirements.txt
cp examples/config.example.yaml my.local.yaml   # describe your providers
export AUTO_ROUTER_CONFIG=my.local.yaml TYPESAFE_API_KEY=...
uvicorn auto_router.server:app --host 127.0.0.1 --port 8787
pytest -q
python experiments/calibrate.py --matrix runs/matrix.jsonl --out success.json   # after run_matrix.py
```

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
| `AUTO_ROUTER_REWRITE_MODEL` | allow replacing Claude Code's requested model on passthrough |
| `AUTO_ROUTER_LEDGER` | JSONL file for decision records; unset disables the ledger |
| `TYPESAFE_API_KEY` | Jev classifier; without it a cautious default is used |

Responses carry `X-Router-Model`, `X-Router-Category`, `X-Router-Difficulty`,
`X-Router-Reason`, `X-Router-Conversation`, `X-Router-Turn-Start`,
`X-Router-Decision` (the id of the decision record), `X-Router-Evidence`
(0–1 evidence confidence), `X-Router-Cache` and, when one applies,
`X-Router-Safe-Fallback`.

### Verifying a checkout

```bash
pytest -q                          # full suite, including the HTTP smoke test
python scripts/smoke_http.py       # the smoke test on its own, with its report
python experiments/sandbox.py      # prove the execution sandbox really isolates
python experiments/heldout.py preregister   # write the task set and analysis plan
python experiments/heldout.py verify        # confirm the task set has not changed
```

`scripts/smoke_http.py` starts a stub upstream and the router itself on
ephemeral loopback ports, with the benchmark API pointed at an unroutable host
and no `TYPESAFE_API_KEY`, so the benchmark-outage and classifier-outage
fallbacks are what is exercised. It tears both processes down and then checks
that neither port still accepts a connection, so no service is left behind.

## Layout

| Path | Purpose |
|---|---|
| `auto_router/catalog.py` | model, price and cache-rule types |
| `auto_router/bench.py` | benchmark API client with disk cache and offline fallback |
| `auto_router/config.py` | provider config → catalog |
| `auto_router/economics.py` | cache-aware cost model and success model |
| `auto_router/policies.py` | policies A–F |
| `auto_router/quota.py` | subscription quota pacing |
| `auto_router/jev.py` | Jev classifier and adequacy judge, with credential scrubbing |
| `auto_router/decision.py` | classification / selection / estimate / observation, kept apart |
| `auto_router/ledger.py` | append-only JSONL decision ledger |
| `auto_router/router.py` | live routing state |
| `auto_router/server.py`, `shim.py` | HTTP API and Claude Code passthrough |
| `auto_router/translate.py`, `stream_translate.py` | Anthropic ↔ OpenAI translation |
| `experiments/sandbox.py` | Bubblewrap isolation for executing model-produced code |
| `experiments/heldout.py` | pre-registered held-out evaluation across six categories |
| `experiments/graders.py` | deterministic graders (no grader calls a model) |
| `experiments/` | task set, evaluation harness, simulator |
| `scripts/smoke_http.py` | local HTTP smoke test including the outage paths |

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

Subscription passthrough is for your own sessions on your own plan, through the
vendor's official client and within its terms. Anthropic's Claude Code terms reserve plan
OAuth for ordinary use of the unmodified Claude Code binary, forbid routing requests through
plan credentials on behalf of others, and forbid intermediating those credentials. A local
proxy that forwards Claude Code's own requests is a grey area under that wording: the router
never adds traffic to a plan, only moves Claude Code's own turns off it, and passthrough is
only active when you configure a `subscription: claude` model. Check the current terms
before enabling it. Plans such as Codex have no API route and are out of scope for the
proxy; choose them at the job level instead.

## Licence

MIT
