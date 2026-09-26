# auto-model-router

A cost-, cache- and quota-aware LLM router. It sits in front of any number of
OpenAI-compatible providers and picks a model per user turn so that tasks get
solved at the lowest expected cost. As a local gateway it sits in front of
Claude Code (Codex, opencode and Cursor have no gateway mode here yet; see
"Four ways to use a flat-rate plan"). It can also route *whole jobs* to a coding
agent's official CLI, and it offers an MCP tool through which Claude Code, Codex,
opencode, Cursor, OpenClaw, Hermes Agent or GitHub Copilot can hand bounded
sub-tasks to cheaper routed workers. The
vendor documentation and its stated limits are linked in [`TERMS.md`](TERMS.md).

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

**Measured in Claude Code (26 Sep 2026, paired A/B).** 20 coding tasks (7 easy, 7 medium,
6 hard; Python/JS modules with hidden tests and single-file web apps), each run in Claude Code
once with Opus 5.5 for every turn and once through this router (v0.5.1 settings, answer check
on). At list prices the router arm cost **2.4× less** ($3.21 against $7.73; median task 10×),
with a **noticeable quality drop**: mean judge score 8.19 against 9.16 out of 10, hidden tests
172/179 against 177/179, and both blind judges preferred the Opus run on 13 of 20 tasks. With
the answer check off the router was 13× cheaper but much worse (117/179 tests). The answer
check rejected 34 of 43 final answers, mostly short summaries of work done in tools that the
judge cannot see; its escalations were about 80 % of the router arm's spend. Raw data, charts
and limits: <https://whichmodel.app.mintapis.com/evidence>.

Everything below this paragraph is older and mostly simulated:
Eight models on 78 graded tasks, a replay of one week of real coding-agent traffic
(1,638 sessions, 57,696 calls, 8.7B input tokens, 96 % of them cache reads), and a live run
of the router server.

> **Read the table as a simulation, because it is one.** Every dollar figure below
> is *replay arithmetic*: real traffic and measured per-model success rates, priced at
> public list prices. **No money was saved and none was measured.** No invoice was
> compared, no A/B test was run against production traffic, and the replay knows the whole week
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
5. **Check the answer — when the route is cheap or below a reference model.**
   When the answering route is in the cheap tier, or its benchmark intelligence
   is below a configured reference (GPT-5.6 Terra in `examples/models.yaml`),
   a Jev-class judge is asked one typed question about what came back, and a
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
  The exception is the intelligence-threshold rule below, which buffers.

The expected-cost policy prices all of this: the judge's own cost, the second
calls its false alarms buy, and the failures it catches that the user would
not have. That is what makes a cheap route *more* attractive than it was —
see "How F decides, plainly".

### Checking any answer from a model below a reference

A second, independent rule (`policy.verify.intelligence_threshold`): when the
answering model is **less intelligent than a reference model** on Benchmark
Heaven, its finished answer is graded the same way, whatever it costs.

```yaml
verify:
  judge: {backend: local-jev, base_url: http://127.0.0.1:8081/v1}   # or hosted (own TYPESAFE_API_KEY)
  intelligence_threshold:
    reference_model: gpt-5.6-terra::max   # AA Intelligence Index 42.1 on 25 Sep 2026
    value: 42.1                           # used only if the reference is not in the data
    per_category: true                    # compare the request's category when both sides are measured
    unknown: check                        # a route with no intelligence data is checked
  max_escalations: 1
  buffer_streams: true
```

- **The comparison.** Per category when both the route and the reference have
  a directly measured score for the request's category (for coding:
  `aa_coding_index`, Terra 76.7), otherwise on the headline
  `aa_intelligence_index` (Terra 42.1). The reference's numbers come from the
  benchmark data (network, disk cache or bundled snapshot); the configured
  `value` is used only when the reference is not in the data. On the example
  model list, Claude Sonnet 5, GPT-6 Luna, GLM-5.3 Flash (41.8) and Bonsai 2 are
  below; Claude Opus 5.5 and GPT-6 Astra are not.
- **What happens.** Pass: the answer is returned. Fail: the turn is re-run on
  the cheapest route that is clearly stronger, preferring one that is not below
  the threshold, and *that* answer goes back to the coding agent. Judge
  unreachable: the original answer is returned and the reason is recorded.
  Intermediate tool-call steps of an agent loop are not graded, only final
  answers. `max_escalations` above 1 grades a second route again if it is
  itself still below the threshold.
- **Streaming.** On the gateway (OpenAI and Anthropic surfaces) a checked
  route's stream is held back until the answer is graded, so an escalated
  answer *replaces* the rejected one. The trade-off is latency: on these routes
  only, nothing reaches the client until the whole first answer is generated and
  graded (plus a whole second answer when it escalates). Every other route
  streams as before. `buffer_streams: false` restores stream-then-check.
- **What is logged.** Every check lands in the decision record's
  `verification` block: rule, model, its intelligence, the threshold, the basis
  and source of the threshold, verdict, probability, judge backend, judge
  latency and escalation target. Because the ledger line is written when the
  outcome is observed, a checked decision is written a second time with
  `"event": "verification"` and the same id; keep the last line per id.
- **Assumptions.** The rule is the operator's choice, not a calibration: the
  catch and false-flag rates in EXPERIMENTS.md were measured on the cheap tier
  and are assumed, not measured, for these stronger routes. Answers served
  through a plan's own login are passed through unchanged and never graded.

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
# classifier: {backend: local-route-head, model: benchmarkheaven/weiche-395m, variant: fp16}
# classifier: {backend: local-jev, base_url: http://127.0.0.1:8081/v1, model: jevk5}
# classifier: {backend: hosted}    # set TYPESAFE_API_KEY
# classifier: {backend: heuristic} # zero inference cost and latency
```

Local means the scrubbed routing input stays on the machine. Hosted uses the
existing Jev API path. If local inference or model loading fails, routing falls
back cautiously instead of failing the user's LLM call.

**Weiche** (`local-route-head`, optional) is a 395M routing model trained for this router
([model card](https://huggingface.co/benchmarkheaven/weiche-395m),
[training code](https://github.com/fstandhartinger/weiche)). It is ModernBERT-large with typed
heads and runs on CPU through ONNX Runtime; it needs no PyTorch
(`pip install "auto-model-router[route-head]"`). One pass returns the same fields as Jev
(category, difficulty, needs_*, follow_up, stakes). It also returns measured-outcome
P(success) for a small, a mid and a strong tier, kept in the classification's `raw`.
On held-out data it got the category right 88.5 % of the time (hosted Jev 85.8 %, local
Laya 9.3 %). On this repository's own 70 held-out tasks it got 94.9 % (Jev 79.7 %, Laya
33.9 %). On the same busy 4-thread CPU it took 386 ms per decision in fp16 (288 ms in int4);
local Laya took 11.1 s. These are development measurements from the model card, not
production traffic results.

**A local open Jev-class model** (`local-jev`) is any decision model served
behind an OpenAI-compatible endpoint by `llama-server` or LM Studio. The
backend asks one question per call for a single option letter and turns the
letters' log-probabilities into probabilities (the published JevK5 v0.2 recipe:
its ChatML prompt, temperature 1.532); `prompt_format: chat` uses
`/chat/completions` instead, for servers without raw completions. The same
backend can grade answers (`verify.judge: same-as-classifier`). Open candidates
and licences (JevBench v1.4.2 notes): JevK5 v0.2 (Apache-2.0, official GGUF
Q4_K_M 2.71 GB; a separate local test measured 43/50 on a public 50-item sample
at 189 ms median on an RTX 3070) and decider-4b v2 (Apache-2.0, BF16 only,
~8.4 GB). No model is
downloaded by the router and no non-commercially licensed model is a default.
The prompt recipe is ported, not re-measured through this backend.

### Models, local targets and benchmark data

[`examples/models.yaml`](examples/models.yaml) is a complete model list:
Claude Opus 5.5 and Claude Sonnet 5 (plan pass-through and metered API),
GPT-6 Luna and GPT-6 Astra (OpenAI API), GLM-5.3 Flash on TensorX
(`https://api.tensorx.ai/v1`, `z-ai/glm-5.3-flash`, $0.20 / $0.50, cache read
$0.05, 1,048,576 tokens) and Bonsai 2 on llama.cpp, plus a local Jev-class
classifier and judge. Switch models on and off with `enabled: [...]` in the
file or `AUTO_ROUTER_MODELS=a,b,c` (which wins); an unknown name is an error.

**Free local targets.** A provider whose `base_url` is on loopback - LM Studio
(`http://127.0.0.1:1234/v1`), `llama-server` (`http://127.0.0.1:8080/v1`),
Ollama (`http://127.0.0.1:11434/v1`) - is `local`: no key is sent and its models
are priced at zero (~zero cost; local electricity and hardware are not
counted), whatever a hosted offer for the same weights costs.

**Bonsai 2** (`prism-ml/Ternary-Bonsai-2-27B-gguf`) is listed on Benchmark Heaven
without scores, so its capability is **assumed**, not measured:
`capability_like: qwen3.8-27b::medium` borrows Qwen3.8 27B's numbers. Every
basis then reads "assumed like ...", the evidence strength is at most weak,
the decision record carries a note and the response an
`X-Router-Capability-Assumed-From` header.

**Benchmark data.** The client reads `GET /api/models/{id}`,
`GET /api/benchmaxxing?report={id}` and, for cost per task only,
`efficiency.global_io_ratio` from `GET /api/price-comparison` on
benchmarkheaven.com (checked 25 Sep 2026). It uses, in order: the network, the
disk cache (24 h fresh, 7 days stale), then a small bundled snapshot
(`auto_router/data/bench-snapshot.json`, minimal fields, with fetch date and
source, provenance `bundled-snapshot`, always treated as stale). Measured
**output tokens per benchmark task** feed the cost model: each route's
expected output is scaled by its tokens per task relative to the catalog
median (bounded 0.25x-4x; `policy.token_appetite: false` turns it off), which
is the "token appetite" part of Benchmark Heaven's cost per task.

```bash
python -m auto_router.bench --show gpt-5.6-terra::max   # scores, prices, tokens and cost per task
python -m auto_router.bench --refresh                   # re-fetch every id in $AUTO_ROUTER_CONFIG
python -m auto_router.bench --write-snapshot --config examples/models.yaml
```

**Prompt cache.** Each route is priced at its real cache state: a warm prefix
is read at the cache-read price, new or expired tokens are written (1.25x input
on Anthropic's 5-minute cache, plain input elsewhere), the warmth expires with
the provider's TTL measured from the start of the request, and switching model
starts cold on the new one. `tests/test_cache_warmth.py` follows this through a
conversation.

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
| `AUTO_ROUTER_BENCH_SNAPSHOT` | `0` = never fall back to the bundled benchmark snapshot |
| `AUTO_ROUTER_MODELS` | comma list of routable model names; overrides `enabled:` |
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
`X-Router-Safe-Fallback` and `X-Router-Capability-Assumed-From`.

## Four ways to use a flat-rate plan

A coding subscription is the cheapest strong model most people have, and it is
also the one a router cannot simply call: it is sold for use through its own
client. There are four honest ways to put a router next to it, and this
repository implements all four. [`TERMS.md`](TERMS.md) quotes the vendor
documentation behind each, including the parts that say no.

| | how it works | plan's login touches the router? | vendor documentation and limits | pick it when |
|---|---|---|---|---|
| **1. `route-run`** | decide per job which official CLI to start | no | ordinary use of the client | whole jobs, scripts, agents |
| **2. gateway** | `ANTHROPIC_BASE_URL` → router; default forwards plan-authenticated turns unchanged; `route_others` is a separate opt-in | yes, in flight | Anthropic documents the billing behavior, says non-Claude routing is unsupported, and restricts developers from intermediating Claude.ai credentials; see [`TERMS.md`](TERMS.md) | only after reviewing those limits for your own login and machine |
| **3. switch mode** | one conversation moves between *cheap mode* (router, its own credential) and *plan mode* (Claude Code direct to Anthropic) | **no** | both modes documented | **interactive Claude Code — the recommended way** |
| **4. delegate tool** | the plan model stays in charge; an MCP tool hands sub-tasks to cheap models | no | documented MCP integration; follow your account and organisation policies | long plan sessions with big, separable sub-tasks; also Codex |

Measured on 19 Sep 2026 on three small coding tasks (details in
[`EXPERIMENTS.md`](EXPERIMENTS.md) §15): every mode finished every task; the
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
relayed as they arrive — and it is billed to the plan. The local gateway checks
the credential type and destination while it is in flight; it does not write the
credential to the routing ledger or logs. Set a gateway credential instead and
the same endpoint meters the traffic against that credential; the OAuth
capability on the request distinguishes the modes
([`auto_router/plan_auth.py`](auto_router/plan_auth.py)).

What the router does with a subscription-authenticated turn is a setting:

| `AUTO_ROUTER_SUBSCRIPTION_MODE` | behaviour |
|---|---|
| `passthrough_only` (default) | Forward every turn unchanged and record the decision the policy *would* have made as `not_taken`. Claude Code behaves exactly as it does with no gateway, and the ledger still answers "how much of this week's plan use could have gone somewhere cheaper". |
| `route_others` | Additionally let the policy serve a turn from another provider on **your own** API key. Anthropic says it does not support routing Claude Code to non-Claude models through a gateway; this mode can break as Claude Code changes and is not the recommended path. |

The gateway mode is an explicit local setup for **your own login on your own
machine**. The router inspects the authorization header only to identify the
credential type and enforce the Anthropic-only upstream rule; it does not persist
the credential and redacts it from logs. Anthropic's legal page also says
developers may not collect, store, or intermediate Claude.ai credentials or
session tokens. This repository does not claim that its local gateway is legally
cleared for every user or organisation. Switch mode (3) avoids sending the plan
login through the router. "Not
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

### 4. Use it as a subagent layer

Keep a strong model as planner and reviewer, and route bounded mechanical work
to workers. The MCP server (`auto-router-delegate`) exposes:

- `delegate(task, context, cwd, tier="cheap"|"auto"|"strong", parallel=n, timeout_s)`
  for one brief, optionally as `n` independent attempts;
- `delegate_many(tasks, context, cwd, tier, parallel, timeout_s)` for up to 32
  different independent briefs.

Workers never run on a subscription route (`--no-plans`). The tier is applied
**among the routes the policy itself allows** - the same tool, context-window
and quota filters - so it can never launch a route the policy ruled out:
`cheap` (the default) prefers the lowest list price for work the classifier
rates easy and keeps the expected-cost choice for hard work, `auto` keeps the
policy's choice, `strong` prefers the most capable allowed worker. The decision
record says which tier chose the route; it is not recorded as an operator
override.

Every response names the selected model, wall time, brief size and the router's
pre-run cost estimate, labelled as an estimate. `cost_usd` is always `null`: no
launched agent CLI reports its token usage back to the launcher, so no actual
cost is measured.

What the server enforces, because a worker is usually a cheap third-party model
with a shell:

- **Arguments** are checked against the tool schema (types, ranges, at most 32
  briefs, a list of strings for `delegate_many`). A malformed call gets a
  structured error and the server keeps serving. The brief follows `--` on the
  launcher's command line, so a brief such as `--list` is a task, not an option.
- **Environment.** A launched worker gets an allowlist, not your environment:
  `PATH`, `HOME`, the locale, `TERM`, `TMPDIR`, the XDG directories and
  `AUTO_ROUTER_DELEGATE_COPY` (the copy a delegated worker runs in), plus what
  its route names (`env_pass: [NAME]`, `env_from: {NAME: SOURCE}`, `env: {...}`)
  and what you add for every route under `launcher.env_allow`. Provider keys,
  `SSH_AUTH_SOCK`, `DBUS_*` and `XDG_RUNTIME_DIR` are not passed unless you name
  them. `launcher.inherit_env: true` restores the old pass-everything behaviour;
  it is not recommended. This covers environment variables only: a worker still
  runs as your user and can read files your user can read, including agent
  logins under `HOME`. Run workers in a container or sandbox if that matters.
- **Working directory.** `cwd` must lie inside the delegation root:
  `AUTO_ROUTER_DELEGATE_ROOT`, else the directory the server was started in
  (the project your client opened). A root of `/`, `$HOME` or a parent of it is
  refused. `route-run` itself accepts `launcher.cwd_root` for the same purpose.
- **Timeouts** end the whole job: the launcher runs in a session of its own and
  the agent in a process group of its own, and on a timeout, `Ctrl-C` or a stop
  signal everything in them gets `SIGTERM`, then `SIGKILL`. A background process
  an agent leaves behind after exiting is stopped too. A descendant that
  detaches with `setsid()` after its parent exited cannot be seen by a process;
  containing that needs a cgroup or sandbox.
  A `Ctrl-C`, `SIGTERM` or `SIGHUP` that arrives while a job is being started
  waits until the job is recorded as running and then ends it like any other
  stop. A job stays recorded until the background processes it left behind
  after exiting are stopped too, so a stop signal that arrives while they are
  being stopped still ends them, with `SIGKILL` for one that outlasts the
  `SIGTERM`. This holds for the stop handlers of `route-run` and the delegate
  server; code that calls the library with a handler of its own that raises
  twice in a row during that cleanup can still cut it short.
- **Parallel work never shares a tree.** With more than one worker, each works in
  its own disposable copy of `cwd` (without `.git`, virtualenvs,
  `node_modules` and caches; at most 200 MB / 50,000 files, see
  `AUTO_ROUTER_DELEGATE_COPY_LIMIT_MB`). The result carries each copy's path and a
  diff against the starting state; nothing is applied to `cwd` and nothing a
  worker wrote is executed by the server. You review the diffs and apply what you
  accept. A single worker runs in `cwd` itself, and tool calls are served one at
  a time. If the server is stopped (`SIGTERM`/`SIGHUP`) or interrupted while
  workers run, queued briefs are not started and the running workers are
  stopped. The copies are deleted only after that, because no reply will name
  them. A worker that cannot be stopped within 30 s keeps its copies on disk
  (`$TMPDIR/auto-router-delegate-*`), since they are not deleted under a writer.
  A second `SIGTERM`/`SIGHUP` while the server is stopping is ignored, so it
  cannot cut that cleanup short; `SIGKILL` still ends the server at once and
  leaves the copies, and its running workers go on running (each is in a
  session of its own) with no timeout applied. `Ctrl-C` runs the same cleanup,
  and a further `Ctrl-C`, `SIGTERM` or `SIGHUP` during it is ignored as well.
  In `route-run`, a further `Ctrl-C`, `SIGTERM` or `SIGHUP` after the first
  `Ctrl-C` is ignored, so the agent keeps its full grace period before
  `SIGKILL`, and `route-run` exits as interrupted by `Ctrl-C`. Without a
  `Ctrl-C` first, a `SIGTERM`/`SIGHUP` still ends `route-run` promptly, after
  stopping the agent with a shorter grace period (0.5 s).
- **Copies left behind are reclaimed when the next server starts, only if
  nothing can still write to them.** Each run's scratch directory holds an
  owner record (`.auto-router-owner.json`: the server's pid and start time,
  the boot id and the pid namespace), which the server keeps locked while it
  runs and deletes when a result hands the copies over. When the server starts,
  it deletes an `$TMPDIR/auto-router-delegate-*` directory only if all of
  these hold: it is a directory (not a link) owned by you; its record names
  that very path and comes from this boot and this pid namespace; no process
  has the recorded pid and start time and the record's lock is free; no
  process visible in `/proc` holds anything inside it (working or root
  directory, executable, open or mapped file); no process started after that
  server was started with `AUTO_ROUTER_DELEGATE_COPY` naming that directory in
  its environment (every worker in a copy gets it, and whatever it starts
  inherits it, so a descendant that detached with `setsid()`, left the copy and
  holds nothing inside it still counts); and no process whose references
  cannot be read (another user's, or a non-dumpable one) started after that
  server did. Everything else stays and is named, with the reason, on the
  server's stderr: copies a result handed over (they have no record; delete
  them yourself once applied), directories without a readable record, copies
  from before a reboot or from another container, and copies still in use,
  which a later start reclaims once they are not. The record is deleted last,
  so an interrupted reclaim is finished by the next start; a server killed in
  the moment between creating the directory and writing the record leaves an
  empty directory (at most an empty record in it) that is never reclaimed. What the check cannot see is a
  process that holds nothing inside the copy at that moment, later writes into
  it by path, and no longer has the variable in the environment it was started
  with: one started with a cleaned environment (by the worker's agent CLI for
  its tools, or by an `env -i`/`exec` of its own), or one that got the path
  some other way (a file, a socket). /proc keeps no other link from such a
  process to the copy once its parent has exited; only running the workers in
  a cgroup or sandbox contains that. Nothing is reclaimed while a server runs,
  only when one starts.
- **Symlinks in copies.** A copy keeps a symlink only if its target is relative
  and stays inside the copy, both as written and after following every link on
  the way. Absolute links (even into the project, which would lead back to the
  original), links whose `..` climbs out and chains that end outside are left
  out of every copy and listed in the result's `copy_skipped`, so editing a
  file in a copy cannot write through a link into `cwd` or elsewhere. FIFOs,
  sockets and devices are skipped too. Entries are opened with `O_NOFOLLOW`
  relative to their parent directory, and links are vetted on the finished,
  private copy before any worker starts, so a tree changing during the copy
  cannot slip a link through. This protects against writing *through a copied
  link*; it is not a sandbox. A worker still runs as your user and can create
  its own links or write to any absolute path your user may write. A link a
  worker creates, retargets or removes in its copy, including a link to a
  directory, appears in the diff as `-> target`, and each result's
  `changes.links_leaving_copy` names the added or retargeted links that lead
  out of the copy; check those before applying a copy's changes.
- **What the diff reads from a copy is capped; the copy itself is not.** The
  200 MB / 50,000-file limit applies to `cwd` before it is copied. A worker may
  then write as much as your disk allows into its copy. The server never reads
  more than 1 MiB of a changed file into memory: a larger one is still listed in
  `added`/`modified`/`deleted` and in `changes.not_diffed`, with no patch.
  Once the patch passes its 60,000-character cap (`patch_truncated`), the
  remaining changed files are listed in `not_diffed` too and are not read.
  Not capped: the number of paths listed, and comparing two large files of equal
  size to decide whether one was modified (streamed, so it takes time but not
  memory). Files a worker writes under an ignored name (`.git`, `node_modules`,
  `__pycache__` and the other names left out of copies) do not appear in the
  diff at all, and a copy whose only changes are there is deleted as unchanged.

The included `plan-with-cheap-workers` skill tells the main model to retain
judgement, security-sensitive work and final review; send only task-local context;
parallelise only independent work; and verify every worker result.

**Evaluation: no saving.** The only numbers are derived from the 19 September
runs in [`EXPERIMENTS.md`](EXPERIMENTS.md) §15, re-tabulated on 21 September in
[`DELEGATION_EVALUATION.md`](DELEGATION_EVALUATION.md); nothing new was run, and
the tier, parallel and isolation code described above did not exist yet when
they were measured. On three small agentic coding tasks, quality stayed 3/3 with
and without delegation; the plan model (Claude Sonnet 5) used 1.7 % more at
API-equivalent list prices, and wall time was 4.1× longer. On one repeated Codex
task, quality stayed 1/1, plan tokens rose 6.6 %, and wall time was 2.4× longer.
The worker brief was *estimated* by the router at 179 prompt tokens; the worker's
actual usage was not reported. Which free model the Codex run's worker used is
recorded inconsistently (the ledger says Qwen3.8 27B, the run's notes say Kimi
K3); see the evaluation file. Delegation may fit larger separable work; this
sample does not show that.

**Install.** Pick a commit you have reviewed, read the installer at that commit,
then run it with that commit's full SHA. It refuses a branch name or short SHA,
never overwrites an existing MCP entry, skill, Cursor rule or command link
without `--force` (and backs up what it replaces), writes JSON settings
atomically, refuses JSONC it cannot round-trip, and reads or writes no
credential:

```bash
REF=<full 40-character commit SHA you reviewed>
git clone https://github.com/fstandhartinger/auto-model-router.git && cd auto-model-router
git checkout --detach "$REF" && less scripts/install-delegation.sh scripts/install-delegation.py
sh scripts/install-delegation.sh claude "$REF" --config ~/router.local.yaml
# or: codex, opencode, cursor, openclaw, hermes, copilot in place of claude
```

`--config` (or `AUTO_ROUTER_CONFIG` set when you install) is recorded in the MCP
entry only if that file exists; without it the entry records no path and the
server reads `AUTO_ROUTER_CONFIG` from the environment it starts in. Claude Code
and Codex get a user-level skill and MCP entry through their own `mcp` commands;
opencode gets its skill and an entry in `~/.config/opencode/opencode.json`;
Cursor gets an entry in `~/.cursor/mcp.json`, and the project rule only with
`--project DIR`.

The OpenClaw, Hermes Agent and GitHub Copilot targets write these files; none
of them runs the agent's own CLI:

| Target | Command | File and key written | Skill / instructions |
| --- | --- | --- | --- |
| OpenClaw | `sh scripts/install-delegation.sh openclaw "$REF" --config ~/router.local.yaml` | `~/.openclaw/openclaw.json`, `mcp.servers.auto-router-delegate` (`command`, `args`, `env`); `OPENCLAW_CONFIG_PATH` / `OPENCLAW_STATE_DIR` are honoured | `~/.openclaw/skills/plan-with-cheap-workers/SKILL.md` |
| Hermes Agent | `sh scripts/install-delegation.sh hermes "$REF" --config ~/router.local.yaml` | `~/.hermes/config.yaml` (or `$HERMES_HOME`), `mcp_servers.auto-router-delegate` (`command`, `args`, `env`) | `~/.hermes/skills/plan-with-cheap-workers/SKILL.md` |
| Copilot CLI | `sh scripts/install-delegation.sh copilot "$REF" --config ~/router.local.yaml` | `~/.copilot/mcp-config.json` (or `$COPILOT_HOME`), `mcpServers.auto-router-delegate` (`type: local`, `command`, `args`, `tools: ["*"]`, `env`) | none |
| Copilot in VS Code | `sh scripts/install-delegation.sh copilot "$REF" --project DIR --config ~/router.local.yaml` | `DIR/.vscode/mcp.json`, `servers.auto-router-delegate` (`type: stdio`, `command`, `args`, `env`) | `DIR/.github/copilot-instructions.md`, only if absent |

OpenClaw's settings are JSON5: a file with comments or unquoted keys is refused
and the entry printed, together with the equivalent `openclaw mcp set`
command; restart the OpenClaw gateway afterwards. Hermes's YAML is edited as
text so your comments stay: the entry goes right below an existing
`mcp_servers:` block or is appended as a new one, the file is backed up, and
the result must parse to your old settings plus this one entry. An inline
`mcp_servers: {}` or a differing `auto-router-delegate` entry is refused with
the snippet to paste, even with `--force`. Hermes passes a stdio server only
`PATH`, `HOME` and a few other variables plus the entry's `env`, so pass
`--config` there, and add the API-key variables your routes name under `env`
yourself if you want them (the installer writes no credential). An existing
`.github/copilot-instructions.md` is never replaced, not even with `--force`;
the text to add is printed instead. The formats come from
[docs.openclaw.ai](https://docs.openclaw.ai/gateway/config-extensions), Hermes
Agent's own documentation (`website/docs/user-guide/features/mcp.md`,
checked at commit `b20cc5f`),
[GitHub's Copilot CLI docs](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-mcp-servers)
and [VS Code's MCP reference](https://code.visualstudio.com/docs/copilot/reference/mcp-configuration),
checked on 26 September 2026; the installer has not been run against a real
OpenClaw, Hermes or Copilot.

The shell script installs into its own virtualenv under
`~/.auto-router`; Python dependencies come from PyPI at the versions
`pyproject.toml` allows.

What has been tested: `tests/test_install_delegation_e2e.py` runs both entry
points end to end, through their own argument parsing, in a disposable HOME
with fake `git`, `python3 -m venv`, `pip`, `claude`, `codex`, `cursor`,
`opencode`, `openclaw`, `hermes` and `copilot` commands that only record their
calls; `tests/test_install_delegation_targets.py` covers the OpenClaw, Hermes
and Copilot files in a temporary HOME. That proves the control flow:
the order of steps, what is written where, and that a dirty checkout, a
fetched commit that differs from the pin, a foreign `auto-router-delegate`
link, a missing `--config` file, JSONC settings and an existing MCP entry all
stop the install without changing what was there. It does **not** show that the
installer works with the real git, pip, PyPI, Claude Code or Codex CLIs; no
real installation has been run. The installer checks every refusal (a
differing skill, settings file, MCP entry or Cursor rule) before it writes
anything, so a refused install leaves no skill or entry behind. A step that
fails while it is writing (for example `claude mcp add` exiting with an error
after the skill was copied) can still leave a partial install. Running the
same install again stops at the existing Claude Code or Codex MCP entry
instead of reporting it as current, because the installer does not parse
those CLIs' output; use `--force` to replace it.

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
| `auto_router/delegate.py` | MCP `delegate` tools: a plan session hands sub-tasks to cheap routes, with argument checks, a cwd root and per-worker copies |
| `auto_router/procs.py` | runs a launched agent so a timeout ends its whole process group or session |
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

A plan-backed job is launched through the user's own official client. In the
optional Claude gateway mode, Anthropic documents that a saved login stays active
and plan billing and limits apply when only `ANTHROPIC_BASE_URL` is set. Its
gateway guide says routing Claude Code to non-Claude models is unsupported; its
legal page restricts developers from handling or intermediating session
credentials. This router checks the authorization header in memory, forwards a
subscription credential only to Anthropic, and does not persist it or include it
in logs. Those implementation controls do not decide whether the mode fits a
particular user's agreement or organisation policy. It is opt-in; switch mode is
recommended when users do not want the router in the plan-authenticated path.

A ChatGPT plan is reachable through the user's Codex CLI in this router, so it is
configured as a `launch_only` route: the launcher starts that CLI, and the HTTP
surface never offers it. The Codex custom-provider proxy setting is not
implemented or live-tested here.

## Licence

MIT
