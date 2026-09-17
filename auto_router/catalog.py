"""Model catalog: prices, cache semantics, capabilities.

Nothing in here is a hard-coded price list for a particular deployment. A
catalog is assembled at runtime from three layers, later layers winning:

1. ``DEFAULT_CACHE_RULES`` - documented provider cache behaviour (TTL, minimum
   cacheable prefix, whether writes carry a premium). Public vendor docs.
2. Benchmark data (``bench.BenchmarkClient``) - per-category capability,
   per-offer list prices including cache read / cache write prices, context
   length, and a benchmaxxing penalty. Cached on disk with a TTL; the router
   keeps working from the last good copy or from the config alone when the API
   is down.
3. The user's provider config (``config.load_config``) - which models exist on
   which OpenAI-compatible endpoint, price overrides (e.g. a free tier), and
   *measured* cache hit rates. Hit rates are never assumed: measure them per
   model and region, because cross-region inference profiles can miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

#: Capability categories the router reasons about.
CATEGORIES = ("coding", "agentic", "math", "knowledge", "long_context", "tool_use", "general")


@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens.

    ``cache_read`` None means the provider gives no cache discount.
    ``cache_write`` None means cache writes are billed as plain input (OpenAI,
    DeepSeek and most OpenAI-compatible hosts); Anthropic bills 1.25x input for
    the 5-minute cache and 2x for the 1-hour cache.
    """

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None

    @property
    def read(self) -> float:
        return self.input if self.cache_read is None else self.cache_read

    @property
    def write(self) -> float:
        return self.input if self.cache_write is None else self.cache_write

    @property
    def is_free(self) -> bool:
        return self.input == 0 and self.output == 0

    @staticmethod
    def free() -> "Prices":
        return Prices(0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CacheRules:
    ttl_seconds: int = 300
    min_tokens: int = 1024
    #: True when every hit restarts the TTL (Anthropic, OpenAI).
    refresh_on_hit: bool = True
    #: Probability that a warm prefix is actually read back. Measure it.
    hit_rate: float = 0.9


#: Documented cache behaviour by provider family. Sources:
#:   anthropic: docs.claude.com prompt-caching - 5 min TTL refreshed on use,
#:              writes 1.25x input, reads 0.1x input, min 1024 tokens (Opus-class 512 on
#:              recent models; we use the conservative 1024 unless configured).
#:   openai:    platform.openai.com prompt-caching - automatic for prompts >= 1024
#:              tokens, no write premium, in-memory retention 5-10 min (up to 1 h
#:              off-peak), extended retention up to 24 h on supporting models.
#:   deepseek:  api-docs.deepseek.com context caching - disk cache, no write premium,
#:              kept for hours; we use 1 h as a conservative planning value.
#:   generic:   OpenAI-compatible hosts that do prefix caching without publishing a TTL.
DEFAULT_CACHE_RULES: dict[str, CacheRules] = {
    "anthropic": CacheRules(ttl_seconds=300, min_tokens=1024, hit_rate=0.95),
    "openai": CacheRules(ttl_seconds=300, min_tokens=1024, hit_rate=0.8),
    "deepseek": CacheRules(ttl_seconds=3600, min_tokens=64, hit_rate=0.9),
    "google": CacheRules(ttl_seconds=300, min_tokens=2048, hit_rate=0.7),
    "generic": CacheRules(ttl_seconds=300, min_tokens=1024, hit_rate=0.5),
    "none": CacheRules(ttl_seconds=0, min_tokens=10**9, hit_rate=0.0),
}


@dataclass(frozen=True)
class ModelInfo:
    """One routable model on one endpoint."""

    name: str
    provider: str
    upstream_id: str
    prices: Prices
    cache: CacheRules = field(default_factory=CacheRules)
    context_tokens: int = 128_000
    max_output_tokens: int = 32_000
    vision: bool = False
    tools: bool = True
    #: 0..100 per category; missing categories fall back to "general".
    capability: dict[str, float] = field(default_factory=dict)
    #: Signed benchmaxxing gap in capability points. Positive means headline
    #: benchmarks overstate held-out performance; only the positive part is
    #: charged as a penalty.
    benchmaxxing: float = 0.0
    #: "claude" / "codex" when calls consume a flat-rate subscription quota
    #: instead of money. Zero marginal dollars, but not free: see quota.py.
    subscription: str | None = None
    #: Where the capability numbers came from ("bench", "config", "none").
    capability_source: str = "none"
    bench_id: str | None = None
    #: Seconds to first token under normal load; used as a latency tie-breaker.
    latency_s: float = 5.0

    def cap(self, category: str, *, benchmaxxing_weight: float = 0.5) -> float:
        base = self.capability.get(category)
        if base is None:
            base = self.capability.get("general", 50.0)
        return base - benchmaxxing_weight * max(0.0, self.benchmaxxing)

    def with_(self, **kw) -> "ModelInfo":
        return replace(self, **kw)


class Catalog:
    def __init__(self, models: list[ModelInfo]):
        self._models = {m.name: m for m in models}

    def __contains__(self, name: str) -> bool:
        return name in self._models

    def __getitem__(self, name: str) -> ModelInfo:
        return self._models[name]

    def get(self, name: str | None) -> ModelInfo | None:
        return self._models.get(name) if name else None

    def all(self) -> list[ModelInfo]:
        return list(self._models.values())

    def eligible(self, *, needs_vision: bool = False, needs_tools: bool = False,
                 prompt_tokens: int = 0, exclude: set[str] | None = None) -> list[ModelInfo]:
        out = []
        for m in self._models.values():
            if exclude and m.name in exclude:
                continue
            if needs_vision and not m.vision:
                continue
            if needs_tools and not m.tools:
                continue
            if prompt_tokens and prompt_tokens > m.context_tokens * 0.9:
                continue
            out.append(m)
        return out
