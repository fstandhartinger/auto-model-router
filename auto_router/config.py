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
    subscriptions:
      claude:
        budget_file: ~/.agent-budget.json      # optional usage source
        weekly_reserve: 0.65                   # see quota.py
    models:
      - name: cheap-coder
        provider: my-host
        upstream_id: vendor/model-x
        bench_id: model-x::default             # capability + list price lookup
        bench_offer: {platform: OpenRouter, provider: SomeHost}
        prices: {input: 0.1, output: 0.4, cache_read: 0.01}   # overrides list price
        free: true                              # shorthand for all-zero prices
        cache: {ttl_seconds: 300, hit_rate: 0.93}             # measured values
        capability: {coding: 55}                # overrides benchmark data
        vision: false
        tools: true
        context_tokens: 128000
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .bench import BenchmarkClient, capability_evidence, context_length, pick_offer
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

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


@dataclass
class RouterConfig:
    providers: dict[str, Provider]
    catalog: Catalog
    subscriptions: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


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

    if isinstance(entry.get("prices"), dict):
        p = entry["prices"]
        prices = Prices(float(p["input"]), float(p["output"]), p.get("cache_read"), p.get("cache_write"))
    if entry.get("free") or entry.get("subscription"):
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
    )


def load_config(path: str | Path | None = None, *, bench: BenchmarkClient | None = None,
                use_bench: bool = True) -> RouterConfig:
    path = path or os.environ.get("AUTO_ROUTER_CONFIG")
    if not path:
        return RouterConfig(providers={}, catalog=Catalog([]))
    raw = _load_file(path)
    providers = {
        name: Provider(name=name, base_url=p["base_url"].rstrip("/"),
                       api_key_env=p.get("api_key_env"), cache=p.get("cache", "generic"),
                       extra_headers=p.get("extra_headers") or {}, api=p.get("api", "openai"))
        for name, p in (raw.get("providers") or {}).items()
    }
    if use_bench and bench is None:
        bench = BenchmarkClient(offline=os.environ.get("AUTO_ROUTER_BENCH_OFFLINE") == "1")
    models = [build_model(m, providers, bench if use_bench else None) for m in raw.get("models") or []]
    return RouterConfig(providers=providers, catalog=Catalog(models),
                        subscriptions=raw.get("subscriptions") or {},
                        policy=raw.get("policy") or {}, raw=raw)
