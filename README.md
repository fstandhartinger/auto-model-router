# auto-model-router

A cost-, cache- and quota-aware LLM router. It sits in front of any number of
OpenAI-compatible providers (and, optionally, a flat-rate Claude subscription
through Claude Code) and picks a model per user turn so that tasks get solved
at the lowest expected cost.

Status: experimental. Measurements and the policy comparison are in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

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
   signals: upstream errors, repeated failing tool results, or a Jev adequacy
   judgement.

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
| F | `F_expected` | minimise expected cost: call cost + P(fail) × (retry or stakes), cache- and quota-aware, with difficulty memory |

## Running

```bash
pip install -r requirements.txt
cp examples/config.example.yaml my.local.yaml   # describe your providers
export AUTO_ROUTER_CONFIG=my.local.yaml TYPESAFE_API_KEY=...
uvicorn auto_router.server:app --host 127.0.0.1 --port 8787
pytest -q
```

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
| `TYPESAFE_API_KEY` | Jev classifier; without it a cautious default is used |

Responses carry `X-Router-Model`, `X-Router-Category`, `X-Router-Difficulty`,
`X-Router-Reason`, `X-Router-Conversation` and `X-Router-Turn-Start`.

## Layout

| Path | Purpose |
|---|---|
| `auto_router/catalog.py` | model, price and cache-rule types |
| `auto_router/bench.py` | benchmark API client with disk cache and offline fallback |
| `auto_router/config.py` | provider config → catalog |
| `auto_router/economics.py` | cache-aware cost model and success model |
| `auto_router/policies.py` | policies A–F |
| `auto_router/quota.py` | subscription quota pacing |
| `auto_router/jev.py` | Jev classifier and adequacy judge |
| `auto_router/router.py` | live routing state |
| `auto_router/server.py`, `shim.py` | HTTP API and Claude Code passthrough |
| `auto_router/translate.py`, `stream_translate.py` | Anthropic ↔ OpenAI translation |
| `experiments/` | task set, evaluation harness, simulator |

## Data and terms

Benchmark numbers served by the default benchmark API include third-party data
whose terms restrict use in competing commercial products. Fine for personal
and internal experiments; a commercial deployment needs its own licensed or
self-measured capability data.

Subscription passthrough is for your own sessions on your own plan, through the
vendor's official client and within its terms. Do not route other people's
traffic through a personal subscription.

## Licence

MIT
