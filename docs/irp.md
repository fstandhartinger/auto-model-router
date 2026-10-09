# Inference Routing Protocol

The router exposes IRP `0.3.0-draft` **suggest-only** endpoints:

* `GET /v1/routing/models`: the models the configured success model can score.
* `POST /v1/routing/rank`: rank 1–512 caller-supplied candidates without forwarding their inference request.

Model ids use explicit `irp_model_id` in a model config entry when supplied. Otherwise a provider `upstream_id` in author/slug form is used; bare provider-local names have the stable id `system1models.ai/<configured-name>`. Set `irp_model_id` to the model's OpenRouter id where one exists, keeping seller-local `upstream_id` separate. Models are grouped by cross-vendor id; candidates remain separate even when they share a model. Unknown models may be omitted; an entirely unscorable list returns 422.

`cost_quality_tradeoff` defaults to 5, accepts integers 0–10, and ranks by `(1-t)*quality - t*cost/max_candidate_cost`, where `t=tradeoff/10`. Ties prefer lower cost, then higher quality, then candidate id. At 0 quality wins; at 10 cost wins. For any two preferences, adding the two winner optimality inequalities proves that the higher preference cannot choose a higher expected cost. Neither usage nor quality is changed by the preference.

Quality uses the existing configured success model, and the configured classifier when available. Without a classifier the existing conservative category/difficulty prior applies. Usage includes all supplied messages and tools using a fixed character-based token estimate, with the configured output appetite and the supplied output limit (reasoning tokens included). Caller-known cached tokens are a lower bound on total input; the estimate never predicts fewer input tokens than these. Costs apply each caller's `input`, `cache_read`, and `output` USD-per-million prices directly. Estimates are advisory, not measurements or guarantees. Multiple requested completions multiply the output estimate. Reasoning effort is omitted in this release because the existing success data cannot predict effort-specific quality/usage; no support is guessed from a model name.

Custom data lives only in `extra["system1models.ai"]`. Unknown fields and namespaces are ignored. Errors use RFC 9457 and the four IRP §8 types (400, 402, 422, 503); a host's existing auth/payment policy can be integrated with `endpoints(get_router, guard)`. The router library does not introduce a new fee. Hosts should put the routes behind their existing authentication if they use one. The body limit is 2 MB, including chunked uploads.

## Public example

The public endpoint is https://whichmodel.app.mintapis.com. It uses the playground catalog and success model with a local conservative prior, without calling a paid classifier. It is free, with its own bounded IP/network allowance. Existing playground spending and behavior remain separate.

```bash
curl -sS https://whichmodel.app.mintapis.com/v1/routing/models
curl -sS https://whichmodel.app.mintapis.com/v1/routing/rank \
  -H 'Content-Type: application/json' \
  -d '{"request":{"messages":[{"role":"user","content":"Write a short Python function."}],"max_tokens":300},"routing":{"cost_quality_tradeoff":5,"candidates":[{"id":"luna@my-provider","model":"openai/gpt-5.6-luna","pricing":{"input":0.5,"cache_read":0.05,"output":2}},{"id":"opus@my-provider","model":"anthropic/claude-opus-5","pricing":{"input":15,"cache_read":1.5,"output":75}}]}}'
```

These are example **caller-supplied** prices, not a price quote from our service. Fetch the model list to check which ids are currently scorable.

## Proxy scope

The existing `/v1/chat/completions` library endpoint remains the legacy router API; it is not advertised as an IRP §6 proxy. The public playground's execution path trims conversation, caps output, substitutes unaffordable frontier models, converts provider reasoning dialects and emits custom SSE events. It cannot relay an arbitrary Chat Completions request/response unchanged. Adding §6 cleanly needs candidate-restricted execution, preservation of all request fields, actual-model/candidate identity on every standard SSE chunk, and a suitable authenticated inference budget. This release therefore claims §5 only, instead of presenting playground output as a compliant §6 proxy.

## Conformance

`tests/test_irp.py` covers server requirements in §2, §3, §4, §5.1, §5.2 and §8. Client MUSTs in §3.4/§5.3 and proxy MUSTs in §6 are outside this suggest-only server's scope. Required fields, candidate bounds/uniqueness, model identity, JSON shapes, finite prices, cache arithmetic, ignored fields/model/stream, 422 instead of an empty result, all problem types, supported reasoning efforts, statelessness and the monotonic preference rule are exercised. The shared ranking code never forwards candidate inference or updates proxy conversation state.
