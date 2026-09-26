"""Provider configuration and catalog assembly.

The router ships with no providers. You describe your own OpenAI-compatible
endpoints (and, optionally, a flat-rate subscription tier) in a YAML or JSON
file and point ``AUTO_ROUTER_CONFIG`` at it. See ``examples/config.example.yaml``.

Schema (YAML shown)::

    providers:
      my-host:
        base_url: https://api.example.com/v1
        api_key_env: EXAMPLE_API_KEY          # name of the env var, never the key
        cache: openai                          # key into DEFAULT_CACHE_RULES
        extra_headers: {User-Agent: "..."}
        max_tokens_field: max_completion_tokens   # when the endpoint rejects max_tokens
    subscriptions:
      claude:
        usage_command: [my-usage-reader, --json]   # your own reader, percentages only
        budget_file: ~/.agent-budget.json      # optional cached usage source
        weekly_reserve: 0.65                   # see quota.py
    models:
      - name: cheap-coder
        provider: my-host
        upstream_id: vendor/model-x
        bench_id: model-x::default             # capability + list price lookup
        success_key: model-x                   # name in the measured success table
        bench_offer: {platform: OpenRouter, provider: SomeHost}
        prices: {input: 0.1, output: 0.4, cache_read: 0.01}   # overrides list price
        free: true                              # shorthand for all-zero prices
        cache: {ttl_seconds: 300, hit_rate: 0.93}             # measured values
        capability: {coding: 55}                # overrides benchmark data
        vision: false
        tools: true
        context_tokens: 128000
        launch_only: true                       # only reachable by launching its own client
        runner:                                 # optional: how route-run starts this route
          cmd: [the-official-cli, --model, vendor/model-x]
          stdin: true                           # pass the task on stdin
          env: {SOME_API_KEY: ""}               # cleared, never a literal secret
        capability_like: other-model::medium    # borrow another model's benchmark data
                                                # (an assumption; recorded as such)
        cache_write_premium: false              # this endpoint bills cache writes as input
    enabled: [cheap-coder]                      # optional: only these models are routable

``AUTO_ROUTER_MODELS=a,b,c`` overrides ``enabled`` without editing the file.
A provider whose ``base_url`` is on loopback (LM Studio, llama.cpp, Ollama) is
``local`` unless the config says otherwise; its models need no key.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import urllib.parse
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .bench import (BenchmarkClient, capability_evidence, context_length, intelligence_index,
                    pick_offer, task_tokens)
from .catalog import CONFIG_OVERRIDE, DEFAULT_CACHE_RULES, CacheRules, Catalog, ModelInfo, Prices


@dataclass
class Provider:
    name: str
    base_url: str
    api_key_env: str | None = None
    cache: str = "generic"
    extra_headers: dict[str, str] = field(default_factory=dict)
    #: "openai" (chat completions) or "anthropic" (messages passthrough).
    api: str = "openai"
    #: Runs on this machine (LM Studio, llama.cpp server, Ollama). No key is
    #: needed and nothing is billed; see ``is_loopback``.
    local: bool = False
    #: Name of the output-budget field this endpoint accepts. OpenAI's
    #: reasoning models reject ``max_tokens`` and take ``max_completion_tokens``.
    max_tokens_field: str = "max_tokens"

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


log = logging.getLogger("auto_router.config")


def is_loopback(base_url: str) -> bool:
    host = (urllib.parse.urlparse(base_url).hostname or "").lower()
    return host in ("localhost", "::1") or host.startswith("127.")


@dataclass
class RouterConfig:
    providers: dict[str, Provider]
    catalog: Catalog
    subscriptions: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    #: The model the intelligence-threshold check compares against, resolved
    #: from benchmark data at load time (see ``resolve_reference``). None when
    #: no reference is configured.
    intelligence_reference: dict[str, Any] | None = None
    #: Routes that are switched off (``enabled``) but still serve as a plan
    #: route's list-price reference. Prices only, never candidates.
    reference_models: Catalog | None = None


def _load_file(path: str | Path) -> dict:
    text = Path(path).expanduser().read_text()
    if str(path).endswith((".yaml", ".yml")):
        import yaml  # optional dependency, only needed for YAML configs
        return yaml.safe_load(text) or {}
    return json.loads(text)


def build_model(entry: dict, providers: dict[str, Provider],
                bench: BenchmarkClient | None) -> ModelInfo:
    provider = providers.get(entry.get("provider", ""))
    cache_family = (entry.get("cache_family") or (provider.cache if provider else "generic"))
    rules = DEFAULT_CACHE_RULES.get(cache_family, DEFAULT_CACHE_RULES["generic"])
    if isinstance(entry.get("cache"), dict):
        rules = replace(rules, **entry["cache"])

    prices: Prices | None = None
    capability: dict[str, float] = {}
    basis: dict[str, str] = {}
    strength: dict[str, str] = {}
    ctx: int | None = None
    benchmaxxing = 0.0
    cap_source = "none"
    evidence: dict = {"source": "none"}
    stale = True

    intelligence: float | None = None
    tokens_per_task: float | None = None
    bench_id = entry.get("bench_id")
    if bench and bench_id:
        doc, prov = bench.model_with_provenance(bench_id)
        evidence = {"bench_id": bench_id, **prov.to_dict()}
        stale = prov.stale
        if doc:
            ev = capability_evidence(doc)
            capability = {k: round(e.value, 2) for k, e in ev.items()}
            basis = {k: e.basis for k, e in ev.items()}
            strength = {k: e.strength for k, e in ev.items()}
            cap_source = "bench" if capability else "none"
            evidence["release_date"] = doc.get("release_date")
            intelligence = intelligence_index(doc)
            tokens = task_tokens(doc)
            if tokens is not None:
                tokens_per_task = tokens.output
                evidence["task_tokens"] = tokens.to_dict()
            ctx = context_length(doc)
            offer_sel = entry.get("bench_offer") or {}
            offer = pick_offer(doc, offer_sel.get("platform"), offer_sel.get("provider"))
            if offer:
                prices = Prices(
                    input=float(offer["input_per_1m"]),
                    output=float(offer.get("output_per_1m") or 0.0),
                    cache_read=offer.get("cache_read_per_1m"),
                    cache_write=offer.get("cache_write_per_1m"),
                )
                evidence["offer"] = {"platform": offer.get("platform"),
                                     "provider": offer.get("provider")}
        # Fetched separately from the model document and able to be stale on
        # its own, so it carries its own provenance and a stale score is not
        # allowed to silently move a capability that is otherwise current.
        score, bm_prov = bench.benchmaxxing_with_provenance(bench_id)
        evidence["benchmaxxing"] = bm_prov.to_dict()
        if score is not None and not bm_prov.stale:
            benchmaxxing = score
        elif score is not None:
            evidence["benchmaxxing_ignored"] = "stale benchmaxxing report; penalty not applied"

    assumed_from = entry.get("capability_like")
    if assumed_from:
        # The route's own model was never measured, so its capability is
        # *borrowed* from a model the operator says it resembles (a quantised
        # build of the same base model, typically). That is an assumption, not
        # evidence: every basis says so, the strength is at most "weak", and
        # the decision record carries the alias.
        assumed_from = str(assumed_from)
        alias_doc, alias_prov = (bench.model_with_provenance(assumed_from) if bench
                                 else (None, None))
        evidence = {**evidence, "assumed_capability": True,
                    "capability_assumed_from": assumed_from,
                    "capability_assumption": (f"capability assumed equal to {assumed_from}; "
                                              "not measured for this model")}
        if alias_prov is not None:
            evidence["capability_like_provenance"] = alias_prov.to_dict()
        if alias_doc:
            ev = capability_evidence(alias_doc)
            capability = {k: round(e.value, 2) for k, e in ev.items()}
            basis = {k: f"assumed like {assumed_from}: {e.basis}" for k, e in ev.items()}
            strength = {k: "weak" for k in ev}
            cap_source = f"assumed:{assumed_from}" if capability else "none"
            intelligence = intelligence_index(alias_doc)
            if tokens_per_task is None:
                alias_tokens = task_tokens(alias_doc)
                if alias_tokens is not None:
                    tokens_per_task = alias_tokens.output
                    evidence["task_tokens"] = {**alias_tokens.to_dict(),
                                               "assumed_from": assumed_from}
            if not ctx:
                ctx = context_length(alias_doc)
            stale = bool(alias_prov.stale)
        else:
            capability, basis, strength, cap_source = {}, {}, {}, "none"
            intelligence = None
            stale = True

    if isinstance(entry.get("prices"), dict):
        p = entry["prices"]
        prices = Prices(float(p["input"]), float(p["output"]), p.get("cache_read"), p.get("cache_write"))
    if entry.get("cache_write_premium") is False and prices is not None:
        # The benchmark offer may list a write premium another endpoint of the
        # same model charges; this endpoint bills cache writes as plain input.
        prices = replace(prices, cache_write=None)
    local = bool(entry.get("local", provider.local if provider else False))
    if entry.get("free") or entry.get("subscription"):
        prices = Prices.free()
    elif local and not isinstance(entry.get("prices"), dict):
        # A model on the operator's own machine costs nothing per token. The
        # benchmark document's offers are some hosted provider's prices for the
        # same weights and say nothing about this endpoint. Electricity and
        # hardware are real but not counted; the README says so.
        prices = Prices.free()
    if prices is None:
        raise ValueError(f"model {entry.get('name')!r}: no prices (set prices, free, or a bench_id with offers)")

    if isinstance(entry.get("capability"), dict):
        # An explicit config number is a deliberate statement by the operator,
        # so it counts as direct evidence and is never discounted as stale.
        overrides = {k: float(v) for k, v in entry["capability"].items()}
        capability = {**capability, **overrides}
        basis = {**basis, **{k: CONFIG_OVERRIDE for k in overrides}}
        strength = {**strength, **{k: "direct" for k in overrides}}
        cap_source = "config" if cap_source == "none" else cap_source + "+config"
        if cap_source == "config":
            stale = False
            evidence = {"source": "config", "stale": False}
    if "benchmaxxing" in entry:
        benchmaxxing = float(entry["benchmaxxing"])

    return ModelInfo(
        name=entry["name"],
        provider=entry.get("provider", ""),
        upstream_id=entry.get("upstream_id", entry["name"]),
        prices=prices,
        cache=rules,
        context_tokens=int(entry.get("context_tokens") or ctx or 128_000),
        max_output_tokens=int(entry.get("max_output_tokens") or 32_000),
        vision=bool(entry.get("vision", False)),
        tools=bool(entry.get("tools", True)),
        capability=capability,
        benchmaxxing=benchmaxxing,
        subscription=entry.get("subscription"),
        capability_source=cap_source,
        capability_basis=basis,
        capability_strength=strength,
        evidence_stale=bool(stale and cap_source != "config"),
        evidence=evidence,
        bench_id=bench_id,
        latency_s=float(entry.get("latency_s", 5.0)),
        success_key=entry.get("success_key"),
        runner=entry.get("runner"),
        launch_only=bool(entry.get("launch_only", False)),
        intelligence_index=intelligence,
        task_tokens=tokens_per_task,
        capability_assumed_from=assumed_from or None,
        local=local,
    )


#: Bounds on the token-appetite multiplier. Measured tokens per task vary by
#: more than 10x across models; the bound keeps one outlier measurement from
#: making a route look free or unaffordable on its own.
APPETITE_BOUNDS = (0.25, 4.0)


def apply_token_appetite(models: list[ModelInfo], enabled: bool = True) -> list[ModelInfo]:
    """Scale each route's expected output by its measured tokens per task.

    The reference is the median over the routes in this catalog that have a
    measurement, so the multiplier compares routes with each other and a
    catalog with a single measured route is left alone. Routes with no
    measurement keep 1.0: no data is not evidence of a small appetite.
    """
    measured = [m.task_tokens for m in models if m.task_tokens]
    if not enabled or len(measured) < 2:
        return models
    median = statistics.median(measured)
    lo, hi = APPETITE_BOUNDS
    out = []
    for m in models:
        if not m.task_tokens:
            out.append(m)
            continue
        factor = round(max(lo, min(hi, m.task_tokens / median)), 3)
        evidence = {**m.evidence, "output_appetite": {
            "factor": factor, "tokens_per_task": round(m.task_tokens, 1),
            "catalog_median": round(median, 1),
            "basis": "measured output tokens per benchmark task / catalog median"}}
        out.append(m.with_(output_appetite=factor, evidence=evidence))
    return out


def enabled_names(raw: dict) -> list[str] | None:
    """Which models are routable: ``AUTO_ROUTER_MODELS`` beats ``enabled:``; None means all."""
    env = os.environ.get("AUTO_ROUTER_MODELS", "").strip()
    if env:
        return [n.strip() for n in env.split(",") if n.strip()]
    listed = raw.get("enabled")
    if listed is None:
        return None
    if isinstance(listed, str):
        listed = [n.strip() for n in listed.split(",")]
    return [str(n) for n in listed if str(n).strip()]


def select_models(raw: dict) -> list[dict]:
    """The model entries that are switched on. An unknown name is an error, not a no-op.

    Subscription entries' ``list_price_model`` references stay loadable: a
    plan route is shadow-priced at another route's list price, which is a
    price, not a candidate.
    """
    entries = list(raw.get("models") or [])
    names = enabled_names(raw)
    if names is None:
        return entries
    known = {e.get("name") for e in entries}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"enabled models not in the config: {', '.join(unknown)} "
                         f"(known: {', '.join(sorted(str(k) for k in known))})")
    wanted = set(names)
    return [e for e in entries if e.get("name") in wanted]


def resolve_reference(policy: dict, bench: BenchmarkClient | None) -> dict | None:
    """The intelligence reference for ``policy.verify.intelligence_threshold``.

    Looked up in the benchmark data (network, disk cache or bundled snapshot,
    whichever ``BenchmarkClient`` finds first). When the reference model is not
    in the data at all, the configured numeric ``value`` is used and the
    record says so. With neither, the rule has no threshold and checks nothing.
    """
    conf = ((policy or {}).get("verify") or {}).get("intelligence_threshold")
    if not isinstance(conf, dict) or conf.get("enabled") is False:
        return None
    ref = conf.get("reference_model")
    fallback = conf.get("value")
    out: dict[str, Any] = {"reference_model": ref, "configured_value": fallback,
                           "intelligence_index": None, "capability": {}, "source": "none"}
    if ref and bench is not None:
        doc, prov = bench.model_with_provenance(str(ref))
        out["provenance"] = prov.to_dict()
        if doc:
            out["resolved_id"] = doc.get("id")
            out["intelligence_index"] = intelligence_index(doc)
            out["capability"] = {k: {"value": round(e.value, 2), "strength": e.strength}
                                 for k, e in capability_evidence(doc).items()}
            out["source"] = f"benchmark:{prov.source}"
    if out["intelligence_index"] is None and fallback is not None:
        out["intelligence_index"] = float(fallback)
        out["source"] = "configured-value"
        out["note"] = (f"{ref or 'the reference model'} has no intelligence index in the "
                       "benchmark data; the configured value is used")
    return out


def for_http(config: RouterConfig) -> RouterConfig:
    """The same configuration, minus routes that only a launched client can reach.

    A subscription that is served exclusively through its own CLI is not an
    endpoint: no HTTP request from another client can be answered from it. It
    stays in the catalog for the launcher and disappears here.
    """
    return replace(config, catalog=Catalog(config.catalog.http_routable()))


def load_config(path: str | Path | None = None, *, bench: BenchmarkClient | None = None,
                use_bench: bool = True) -> RouterConfig:
    path = path or os.environ.get("AUTO_ROUTER_CONFIG")
    if not path:
        return RouterConfig(providers={}, catalog=Catalog([]))
    raw = _load_file(path)
    providers = {
        name: Provider(name=name, base_url=p["base_url"].rstrip("/"),
                       api_key_env=p.get("api_key_env"), cache=p.get("cache", "generic"),
                       extra_headers=p.get("extra_headers") or {}, api=p.get("api", "openai"),
                       local=bool(p.get("local", is_loopback(p["base_url"]))),
                       max_tokens_field=str(p.get("max_tokens_field") or "max_tokens"))
        for name, p in (raw.get("providers") or {}).items()
    }
    if use_bench and bench is None:
        bench = BenchmarkClient(offline=os.environ.get("AUTO_ROUTER_BENCH_OFFLINE") == "1")
    entries = select_models(raw)
    # A plan route's list-price reference must stay resolvable even when the
    # reference route itself is switched off; it is loaded but not routable.
    names = {e.get("name") for e in entries}
    refs = {e.get("list_price_model") for e in entries if e.get("list_price_model")} - names
    policy = raw.get("policy") or {}
    client = bench if use_bench else None
    models = apply_token_appetite([build_model(m, providers, client) for m in entries],
                                  enabled=policy.get("token_appetite", True) is not False)
    if len(entries) != len(raw.get("models") or []):
        log.info("routable models: %s", ", ".join(m.name for m in models))
    config = RouterConfig(providers=providers, catalog=Catalog(models),
                          subscriptions=raw.get("subscriptions") or {},
                          policy=policy, raw=raw,
                          intelligence_reference=resolve_reference(policy, client))
    if refs:
        config.reference_models = Catalog([build_model(e, providers, client)
                                           for e in raw.get("models") or []
                                           if e.get("name") in refs])
    return config
