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

#: Capability categories the router reasons about. ``design`` (web/UI work) and
#: ``summarisation`` are specialised categories: which model is best at them is
#: read from the configured evidence, never hard-coded here.
CATEGORIES = ("coding", "agentic", "math", "knowledge", "long_context", "tool_use",
              "design", "summarisation", "general")

#: Basis string for a capability the operator stated directly in the config.
#: It is current by definition and is never aged out by benchmark staleness.
CONFIG_OVERRIDE = "config override"

#: How much a capability score is trusted, by the strength of its evidence.
#: A derived number (a neighbouring index standing in for a category nobody
#: measured) and a stale one are both pulled toward the pool mean so the policy
#: does not act on them as if they were measured. See ``ModelInfo.cap``.
EVIDENCE_WEIGHT = {"direct": 1.0, "derived": 0.6, "weak": 0.4, "none": 0.0}


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
    #: category -> human-readable basis for that capability number.
    capability_basis: dict[str, str] = field(default_factory=dict)
    #: category -> "direct" | "derived" | "weak"; missing means "none".
    capability_strength: dict[str, str] = field(default_factory=dict)
    #: True when the benchmark document behind these numbers was stale or absent.
    #: It applies per category: a hand-written config override is a deliberate
    #: current statement by the operator and is never demoted by a stale
    #: benchmark document for a *different* category on the same model.
    evidence_stale: bool = False
    #: Compact provenance record for the routing explanation (no prompt text).
    evidence: dict = field(default_factory=dict)
    bench_id: str | None = None
    #: Seconds to first token under normal load; used as a latency tie-breaker.
    latency_s: float = 5.0
    #: Name this route's model is *measured* under, when that differs from the
    #: route name. The same model reached two ways - through a flat-rate plan's
    #: client and through a metered API - is one model as far as capability and
    #: measured success rates go, and splitting them would quietly drop a route
    #: back onto the fitted curve. Defaults to ``name``.
    success_key: str | None = None
    #: How to run a whole job on this route through its own official client,
    #: for the job-level launcher: ``{cmd, env, env_from, stdin, timeout_s}``.
    #: ``None`` means the route is reachable over HTTP only. See launcher.py.
    runner: dict | None = None
    #: True when the *only* way to reach this route is to launch its own
    #: client. A flat-rate plan tied to one vendor's CLI is the case that
    #: matters: it cannot serve another client's HTTP request at all, and a
    #: router that offered it as an HTTP route would either fail the request or
    #: invite an implementation to send one vendor's login to another. Such a
    #: route is a candidate for the launcher and for nothing else.
    launch_only: bool = False

    @property
    def measurement_key(self) -> str:
        """The name measurements about this model are filed under."""
        return self.success_key or self.name

    def evidence_strength(self, category: str) -> str:
        """How well this model's capability in ``category`` is evidenced."""
        if category not in self.capability:
            return "none"
        if self.capability_basis.get(category) == CONFIG_OVERRIDE:
            return self.capability_strength.get(category, "direct")
        if self.evidence_stale:
            # A stale document may still be right, but it is no longer current
            # evidence, so nothing derived from it counts as direct.
            return "weak" if self.capability_strength.get(category) == "direct" else "none"
        return self.capability_strength.get(category, "derived")

    def evidence_basis(self, category: str) -> str:
        if category not in self.capability:
            return f"fallback: general ({self.capability_basis.get('general', 'none')})"
        return self.capability_basis.get(category, "unrecorded")

    def cap(self, category: str, *, benchmaxxing_weight: float = 0.5,
            evidence_discount: float = 0.0, neutral: float = 50.0) -> float:
        """Capability score, optionally shrunk toward ``neutral`` by evidence strength.

        With ``evidence_discount`` at 0 this is the raw score and behaves as it
        always has. Above 0 a number the router cannot fully vouch for - a
        derived stand-in, a thin sample, a stale document - is pulled toward a
        neutral prior in proportion to how weak it is. A confident cheap route
        therefore needs real evidence before the policy will trust it with a
        specialised task.
        """
        base = self.capability.get(category)
        if base is None:
            base = self.capability.get("general", neutral)
        score = base - benchmaxxing_weight * max(0.0, self.benchmaxxing)
        if evidence_discount > 0.0:
            weight = EVIDENCE_WEIGHT.get(self.evidence_strength(category), 0.0)
            shrink = evidence_discount * (1.0 - weight)
            score = score * (1.0 - shrink) + neutral * shrink
        return score

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

    def http_routable(self) -> list[ModelInfo]:
        """Routes an HTTP client can actually be served from."""
        return [m for m in self._models.values() if not m.launch_only]

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
